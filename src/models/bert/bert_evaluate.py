import torch
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification,DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import load_from_disk
import os

def to_python_serializable(obj):
    """Recursively convert NumPy types to native Python for JSON serialization."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_python_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_python_serializable(v) for k, v in obj.items()}
    return obj

class BERTEvaluator:
    def __init__(self, model_path="./artifacts/bert_ckpt/best_model", config_path="src/configs/bert_hparams.yaml"):
        """BERT model evaluator for comprehensive testing"""
        
        # Resolve paths robustly relative to this file
        current_file_dir = os.path.dirname(__file__)
        src_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
        project_dir = os.path.abspath(os.path.join(src_dir, '..'))
        self.src_dir = src_dir
        self.project_dir = project_dir

        # Resolve config path
        default_config_path = os.path.join(src_dir, 'configs', 'bert_hparams.yaml')
        alt_config_path = os.path.join(project_dir, 'src', 'configs', 'bert_hparams.yaml')
        if not os.path.exists(config_path):
            if os.path.exists(default_config_path):
                config_path = default_config_path
            elif os.path.exists(alt_config_path):
                config_path = alt_config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Resolve model path
        default_model_path = os.path.join(project_dir, 'artifacts', 'bert_ckpt', 'best_model')
        alt_model_path = os.path.join('duyguanalizi', 'artifacts', 'bert_ckpt', 'best_model')
        if not os.path.exists(model_path):
            if os.path.exists(default_model_path):
                model_path = default_model_path
            elif os.path.exists(alt_model_path):
                model_path = alt_model_path

        # Load tokenizer and model
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Label mappings: prefer model-config mapping if available
        label_map = None
        try:
            id2label = getattr(self.model.config, 'id2label', None)
            if id2label:
                parsed = {}
                for k, v in id2label.items():
                    try:
                        parsed[int(k)] = str(v)
                    except Exception:
                        # Fall back if keys are already ints or unexpected
                        if isinstance(k, int):
                            parsed[k] = str(v)
                if parsed:
                    label_map = dict(sorted(parsed.items(), key=lambda kv: kv[0]))
        except Exception:
            pass
        if label_map is None:
            # Default mapping consistent with dataset used in this project
            label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        self.label_map = label_map
        
        print("✅ Model loaded successfully!")
    
    def evaluate_dataset(self, dataset_path="data/processed/bert_val"):
        """Evaluate model on validation dataset"""
        print(f"\n📊 Loading validation dataset from: {dataset_path}")
        
        try:
            # Resolve dataset path
            candidate_paths = [
                dataset_path,
                os.path.join(self.project_dir, dataset_path),
            ]
            resolved_path = None
            for cand in candidate_paths:
                if os.path.exists(cand):
                    resolved_path = cand
                    break
            if resolved_path is None:
                raise FileNotFoundError(f"Directory {dataset_path} not found")

            val_dataset = load_from_disk(resolved_path)
            print(f"Dataset size: {len(val_dataset)} samples")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
        
        # Get predictions
        predictions = []
        true_labels = []
        confidences = []
        
        print("🔮 Making predictions...")
        batch_size = 16
        column_names = set(val_dataset.column_names)
        has_text_column = 'text' in column_names
        label_key = 'label' if 'label' in column_names else 'labels'
        
        for i in range(0, len(val_dataset), batch_size):
            batch = val_dataset[i:i+batch_size]
            
            # Prepare model inputs
            if has_text_column:
                # Tokenize from raw text if available
                inputs = self.tokenizer(
                    batch['text'],
                    truncation=True,
                    padding=True,
                    max_length=self.config['model']['max_length'],
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Use pre-tokenized inputs from the dataset with proper padding
                inputs = self.tokenizer.pad(
                    {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']},
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
            
            predictions.extend(batch_predictions)
            true_labels.extend(batch[label_key])
            confidences.extend(batch_confidences)
            
            # Progress
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(val_dataset))}/{len(val_dataset)} samples")
        
        # Recover texts for reporting if possible
        if has_text_column:
            texts = val_dataset['text']
        else:
            try:
                texts = self.tokenizer.batch_decode(val_dataset['input_ids'], skip_special_tokens=True)
            except Exception:
                texts = [""] * len(val_dataset)

        return {
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'texts': texts
        }
    
    def calculate_metrics(self, results):
        """Calculate comprehensive metrics"""
        predictions = results['predictions']
        true_labels = results['true_labels']
        confidences = results['confidences']
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # Per-class metrics
        ordered_label_names = [self.label_map[i] for i in sorted(self.label_map.keys())]
        report = classification_report(true_labels, predictions,
                                       target_names=ordered_label_names,
                                       output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Confidence statistics
        avg_confidence = np.mean(confidences)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'average_confidence': avg_confidence,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {name: report[name] for name in ordered_label_names}
        }
        
        return metrics
    

    
    def save_results(self, metrics, results, output_dir="./artifacts/bert_evaluation"):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics JSON
        with open(f"{output_dir}/bert_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(to_python_serializable(metrics), f, indent=2, ensure_ascii=False)
        
        # Save detailed results CSV
        df = pd.DataFrame({
            'text': results['texts'],
            'true_label': results['true_labels'],
            'predicted_label': results['predictions'],
            'confidence': results['confidences'],
            'true_sentiment': [self.label_map[label] for label in results['true_labels']],
            'predicted_sentiment': [self.label_map[label] for label in results['predictions']],
            'correct': np.array(results['true_labels']) == np.array(results['predictions'])
        })
        df.to_csv(f"{output_dir}/bert_predictions.csv", index=False, encoding='utf-8')
        
        # Plot confusion matrix
        self.plot_confusion_matrix(metrics['confusion_matrix'], output_dir)
        
        print(f"📁 Results saved to: {output_dir}")
        return output_dir
    
    def plot_confusion_matrix(self, cm, output_dir):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        ordered_label_names = [self.label_map[i] for i in sorted(self.label_map.keys())]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=ordered_label_names,
                    yticklabels=ordered_label_names)
        plt.title('BERT Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bert_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_metrics(self, metrics):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("🎯 BERT MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  F1 (Macro):    {metrics['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Avg Confidence: {metrics['average_confidence']:.4f}")
        
        print(f"\n📈 Per-Class Performance:")
        for sentiment, metrics_dict in metrics['per_class_metrics'].items():
            print(f"  {sentiment:8s}: Precision={metrics_dict['precision']:.3f}, "
                  f"Recall={metrics_dict['recall']:.3f}, F1={metrics_dict['f1-score']:.3f}")
        
        print(f"\n🔢 Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        labels = list(self.label_map.values())
        print(f"          {'':>10s} {'Predicted':>30s}")
        print(f"          {'':<10s} {' '.join(f'{label:>8s}' for label in labels)}")
        for i, label in enumerate(labels):
            print(f"  True {label:>8s} {' '.join(f'{cm[i][j]:>8d}' for j in range(len(labels)))}")
        
        print("="*60)

def main():
    """Run complete BERT evaluation"""
    print("🚀 Starting BERT Model Evaluation...")
    
    try:
        evaluator = BERTEvaluator()
        
        # Evaluate on validation set
        results = evaluator.evaluate_dataset()
        if results is None:
            return
        
        # Calculate metrics
        print("\n📊 Calculating metrics...")
        metrics = evaluator.calculate_metrics(results)
        
        # Print results
        evaluator.print_metrics(metrics)
        
        # Save results
        output_dir = evaluator.save_results(metrics, results)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📈 Confusion matrix plot: {output_dir}/bert_confusion_matrix.png")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
