import torch
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from datasets import load_from_disk
import os
import warnings
warnings.filterwarnings('ignore')

# Import your custom model
from bert_mlp_classifier import BertMLPClassifier, BertMLPConfig

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

class BertMLPEvaluator:
    def __init__(self, model_path="./artifacts/bert_mlp_ckpt/best_model", config_path="src/configs/bert_hparams.yaml"):
        """BERT+MLP model evaluator with advanced analysis capabilities"""
        
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
        
        # Load tokenizer and custom model
        print(f"Loading BERT+MLP model from: {model_path}")
        
        # Try to load the custom model first
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create custom model with same config
            if self.config['mlp']['use_mlp']:
                mlp_config = BertMLPConfig(
                    hidden_sizes=self.config['mlp']['hidden_sizes'],
                    dropout_rate=self.config['mlp']['dropout_rate'],
                    activation=self.config['mlp']['activation'],
                    use_batch_norm=self.config['mlp']['use_batch_norm']
                )
                
                # Load BERT base model name from config
                bert_model_name = self.config['model']['name']
                self.model = BertMLPClassifier(
                    model_name=bert_model_name,
                    num_classes=self.config['model']['num_classes'],
                    mlp_config=mlp_config
                )
                
                # Try to load the trained weights - handle both safetensors and pytorch formats
                model_file = None
                if os.path.exists(os.path.join(model_path, 'model.safetensors')):
                    model_file = os.path.join(model_path, 'model.safetensors')
                    print("📁 Loading weights from Safetensors format...")
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                elif os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
                    model_file = os.path.join(model_path, 'pytorch_model.bin')
                    print("📁 Loading weights from PyTorch format...")
                    state_dict = torch.load(model_file, map_location=self.device)
                else:
                    raise FileNotFoundError("No model weights found (neither safetensors nor pytorch_model.bin)")
                
                # Load state dict
                self.model.load_state_dict(state_dict)
                print("✅ Custom BERT+MLP model loaded successfully!")
            else:
                raise ValueError("MLP is enabled in config but model loading failed")
                
        except Exception as e:
            print(f"❌ Error loading custom model: {e}")
            print("Falling back to standard HuggingFace model...")
            # Fallback to standard model if custom loading fails
            from transformers import AutoModelForSequenceClassification
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    ignore_mismatched_sizes=True,
                    trust_remote_code=True
                ).to(self.device)
                print("✅ Standard HuggingFace model loaded as fallback")
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                print("Trying to load from BERT base model...")
                # Last resort: load from base BERT model
                bert_model_name = self.config['model']['name']
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    bert_model_name,
                    num_labels=self.config['model']['num_classes']
                ).to(self.device)
                print("✅ Base BERT model loaded (untrained)")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Label mappings
        self.label_map = {'negatif': 0, 'pozitif': 1, 'notr': 2}  # Make sure this matches your data exactly
        self.id2label = {v: k for k, v in self.label_map.items()}  # Reverse mapping for predictions

        print("✅ Model loaded successfully!")
    
    def evaluate_dataset(self, dataset_path="data/processed/bert_test"):
        """Evaluate model on validation dataset with detailed analysis"""
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
        
        # Get predictions with detailed analysis
        predictions = []
        true_labels = []
        confidences = []
        all_probabilities = []
        mlp_outputs = []
        
        print("🔮 Making predictions with detailed analysis...")
        batch_size = 8  # Smaller batch for detailed analysis
        
        column_names = set(val_dataset.column_names)
        has_text_column = 'text' in column_names
        label_key = 'label' if 'label' in column_names else 'labels'
        
        for i in range(0, len(val_dataset), batch_size):
            batch = val_dataset[i:i+batch_size]
            
            # Prepare model inputs
            if has_text_column:
                inputs = self.tokenizer(
                    batch['text'],
                    truncation=True,
                    padding=True,
                    max_length=self.config['model']['max_length'],
                    return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.tokenizer.pad(
                    {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']},
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict with detailed outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
                batch_probabilities = probabilities.cpu().numpy()
                
                # Get MLP layer outputs if available
                if hasattr(self.model, 'mlp') and hasattr(outputs, 'hidden_states'):
                    # This would require modifying your model to return intermediate outputs
                    batch_mlp_outputs = logits.cpu().numpy()  # Use logits as proxy
                else:
                    batch_mlp_outputs = logits.cpu().numpy()
            
            predictions.extend(batch_predictions)
            true_labels.extend(batch[label_key])
            confidences.extend(batch_confidences)
            all_probabilities.extend(batch_probabilities)
            mlp_outputs.extend(batch_mlp_outputs)
            
            # Progress
            if (i // batch_size + 1) % 20 == 0:
                print(f"  Processed {min(i + batch_size, len(val_dataset))}/{len(val_dataset)} samples")
        
        # Recover texts for reporting
        if has_text_column:
            texts = val_dataset['text']
        else:
            try:
                texts = self.tokenizer.batch_decode(val_dataset['input_ids'], skip_special_tokens=True)
            except Exception:
                texts = [""] * len(val_dataset)
        # Coerce to integer IDs
        true_labels = [
            self.label_map[l] if isinstance(l, str) else int(l)
            for l in true_labels
        ]
        predictions = [int(p) for p in predictions]
        return {
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'probabilities': all_probabilities,
            'mlp_outputs': mlp_outputs,
            'texts': texts
        }
    
    def calculate_advanced_metrics(self, results):
        """Calculate comprehensive metrics with consistent int labels."""
        predictions = list(map(int, results['predictions']))
        true_labels = list(map(int, results['true_labels']))
        confidences = results['confidences']
        probabilities = np.array(results['probabilities'])

        labels = [0, 1, 2]
        target_names = [self.id2label[i] for i in labels]

        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro', labels=labels)
        f1_weighted = f1_score(true_labels, predictions, average='weighted', labels=labels)

        # Debug (optional)
        print("🔍 DEBUG (after coercion):")
        print("   true_labels unique values:", sorted(set(true_labels)))
        print("   predictions unique values:", sorted(set(predictions)))
        print("   target_names:", target_names)

        report = classification_report(
            true_labels, predictions,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )

        cm = confusion_matrix(true_labels, predictions, labels=labels)
        avg_confidence = float(np.mean(confidences))

        negative_class_metrics = self._analyze_negative_class(predictions, true_labels, probabilities, confidences)
        bias_analysis = self._analyze_model_bias(predictions, true_labels, probabilities)

        # Build per-class metrics dict keyed by readable names
        per_class = {target_names[i]: report[target_names[i]] for i in range(len(labels))}

        return {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'average_confidence': avg_confidence,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class,
            'negative_class_analysis': negative_class_metrics,
            'bias_analysis': bias_analysis
        }

    def _analyze_negative_class(self, predictions, true_labels, probabilities, confidences):
        """Detailed analysis of negative class performance"""
        # Convert to numpy arrays to ensure proper operations
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        probabilities = np.array(probabilities)
        confidences = np.array(confidences)
        
        negative_mask = (true_labels == 0)
        negative_predictions = predictions[negative_mask]
        negative_confidences = confidences[negative_mask]
        negative_probabilities = probabilities[negative_mask]
        
        # Count misclassifications
        negative_correct = (negative_predictions == 0).sum()
        negative_total = negative_mask.sum()
        negative_incorrect = negative_total - negative_correct
        
        # Analyze where negative samples are misclassified
        misclassified_as = {}
        for pred in negative_predictions[negative_predictions != 0]:
            pred_label = self.id2label[pred]
            misclassified_as[pred_label] = misclassified_as.get(pred_label, 0) + 1
        
        # Confidence analysis for negative class
        negative_confidence_stats = {
            'mean_confidence': float(np.mean(negative_confidences)),
            'std_confidence': float(np.std(negative_confidences)),
            'min_confidence': float(np.min(negative_confidences)),
            'max_confidence': float(np.max(negative_confidences))
        }
        
        # Probability distribution analysis
        negative_prob_0 = negative_probabilities[:, 0]  # Probability of being negative
        negative_prob_stats = {
            'mean_prob_negative': float(np.mean(negative_prob_0)),
            'std_prob_negative': float(np.std(negative_prob_0)),
            'samples_with_low_confidence': int((negative_prob_0 < 0.6).sum())
        }
        
        return {
            'total_negative_samples': int(negative_total),
            'correctly_classified': int(negative_correct),
            'incorrectly_classified': int(negative_incorrect),
            'negative_recall': float(negative_correct / negative_total),
            'misclassified_as': misclassified_as,
            'confidence_statistics': negative_confidence_stats,
            'probability_statistics': negative_prob_stats
        }
    
    def _analyze_model_bias(self, predictions, true_labels, probabilities):
        """Analyze model bias toward different classes"""
        # Class distribution in predictions vs true labels
        pred_distribution = np.bincount(predictions, minlength=3)
        true_distribution = np.bincount(true_labels, minlength=3)
        
        # Bias toward each class
        bias_scores = {}
        for i in range(3):
            pred_ratio = pred_distribution[i] / len(predictions)
            true_ratio = true_distribution[i] / len(true_labels)
            bias_scores[self.id2label[i]] = float(pred_ratio - true_ratio)
        
        # Average probability for each class
        avg_probabilities = {}
        for i in range(3):
            avg_probabilities[self.id2label[i]] = float(np.mean(probabilities[:, i]))
        
        return {
            'prediction_distribution': pred_distribution.tolist(),
            'true_distribution': true_distribution.tolist(),
            'bias_scores': bias_scores,
            'average_probabilities': avg_probabilities
        }
    
    def save_results(self, metrics, results, output_dir="./artifacts/bert_evaluation"):
        """Save evaluation results with enhanced analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics JSON
        with open(f"{output_dir}/bert_mlp_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(to_python_serializable(metrics), f, indent=2, ensure_ascii=False)
        
        # Save detailed results CSV
        df = pd.DataFrame({
            'text': results['texts'],
            'true_label': results['true_labels'],
            'predicted_label': results['predictions'],
            'confidence': results['confidences'],
            'true_sentiment': [self.id2label[label] for label in results['true_labels']],
            'predicted_sentiment': [self.id2label[label] for label in results['predictions']],
            'correct': np.array(results['true_labels']) == np.array(results['predictions']),
            'prob_negative': [probs[0] for probs in results['probabilities']],
            'prob_neutral': [probs[2] for probs in results['probabilities']],
            'prob_positive': [probs[1] for probs in results['probabilities']]
        })
        df.to_csv(f"{output_dir}/bert_mlp_predictions.csv", index=False, encoding='utf-8')
        
        # Create enhanced visualizations
        self._create_enhanced_plots(metrics, results, output_dir)
        
        print(f"📁 Results saved to: {output_dir}")
        return output_dir
    
    def _create_enhanced_plots(self, metrics, results, output_dir):
        """Create enhanced visualization plots"""
        # 1. Confusion Matrix
        self.plot_confusion_matrix(metrics['confusion_matrix'], output_dir)
        
        # 2. Negative Class Analysis
        self.plot_negative_class_analysis(metrics['negative_class_analysis'], output_dir)
        
        # 3. Bias Analysis
        self.plot_bias_analysis(metrics['bias_analysis'], output_dir)
        
        # 4. Confidence Distribution
        self.plot_confidence_distribution(results, output_dir)
    
    def plot_confusion_matrix(self, cm, output_dir):
        """Plot and save confusion matrix"""
        # Convert to numpy array if it's a list
        cm = np.array(cm)
        
        plt.figure(figsize=(10, 8))
        ordered_label_names = [self.id2label[i] for i in sorted(self.id2label.keys())]
        
        # Create heatmap with percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=ordered_label_names,
                    yticklabels=ordered_label_names)
        plt.title('BERT+MLP Model - Confusion Matrix (%)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bert_mlp_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_negative_class_analysis(self, negative_analysis, output_dir):
        """Plot negative class performance analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Misclassification breakdown
        misclassified = negative_analysis['misclassified_as']
        if misclassified:
            labels = list(misclassified.keys())
            values = list(misclassified.values())
            ax1.bar(labels, values, color=['#ff6b6b', '#4ecdc4'])
            ax1.set_title('Negative Samples Misclassified As', fontweight='bold')
            ax1.set_ylabel('Count')
        else:
            ax1.text(0.5, 0.5, 'No misclassifications!', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Negative Samples Misclassified As', fontweight='bold')
        
        # 2. Confidence distribution for negative class
        ax2.hist(negative_analysis['confidence_statistics']['mean_confidence'], bins=20, alpha=0.7, color='skyblue')
        ax2.set_title('Negative Class Confidence Distribution', fontweight='bold')
        ax2.set_xlabel('Confidence')
        ax3.set_ylabel('Frequency')
        
        # 3. Performance summary
        performance_data = [
            negative_analysis['correctly_classified'],
            negative_analysis['incorrectly_classified']
        ]
        performance_labels = ['Correct', 'Incorrect']
        colors = ['#2ecc71', '#e74c3c']
        
        ax3.pie(performance_data, labels=performance_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Negative Class Classification Results', fontweight='bold')
        
        # 4. Probability statistics
        prob_stats = negative_analysis['probability_statistics']
        ax4.bar(['Mean Prob', 'Low Conf Samples'], 
                [prob_stats['mean_prob_negative'], prob_stats['samples_with_low_confidence']],
                color=['#9b59b6', '#f39c12'])
        ax4.set_title('Negative Class Probability Analysis', fontweight='bold')
        ax4.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/negative_class_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_bias_analysis(self, bias_analysis, output_dir):
        """Plot model bias analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Distribution comparison
        labels = list(self.label_map.values())
        x = np.arange(len(labels))
        width = 0.35
        
        ax1.bar(x - width/2, bias_analysis['true_distribution'], width, label='True Distribution', alpha=0.8)
        ax1.bar(x + width/2, bias_analysis['prediction_distribution'], width, label='Predicted Distribution', alpha=0.8)
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Count')
        ax1.set_title('True vs Predicted Class Distribution', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        
        # 2. Bias scores
        bias_values = list(bias_analysis['bias_scores'].values())
        colors = ['red' if x > 0 else 'blue' for x in bias_values]
        ax2.bar(labels, bias_values, color=colors, alpha=0.7)
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Bias Score')
        ax2.set_title('Model Bias Toward Each Class', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bias_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_distribution(self, results, output_dir):
        """Plot confidence distribution across classes"""
        plt.figure(figsize=(12, 8))
        
        # Separate confidence by true class
        for i, label_name in self.label_map.items():
            mask = np.array(results['true_labels']) == i
            if mask.sum() > 0:
                class_confidences = np.array(results['confidences'])[mask]
                plt.hist(class_confidences, bins=20, alpha=0.6, label=f'{label_name} (n={mask.sum()})')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by True Class', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_enhanced_metrics(self, metrics):
        """Pretty print enhanced metrics with negative class focus"""
        print("\n" + "="*80)
        print("🎯 BERT+MLP MODEL EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  F1 (Macro):    {metrics['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Avg Confidence: {metrics['average_confidence']:.4f}")
        
        print(f"\n📈 Per-Class Performance:")
        for sentiment, metrics_dict in metrics['per_class_metrics'].items():
            print(f"  {sentiment:8s}: Precision={metrics_dict['precision']:.3f}, "
                  f"Recall={metrics_dict['recall']:.3f}, F1={metrics_dict['f1-score']:.3f}")
        
        print(f"\n🔍 NEGATIVE CLASS ANALYSIS (Label 0):")
        neg_analysis = metrics['negative_class_analysis']
        print(f"  Total Negative Samples: {neg_analysis['total_negative_samples']}")
        print(f"  Correctly Classified:   {neg_analysis['correctly_classified']}")
        print(f"  Incorrectly Classified: {neg_analysis['incorrectly_classified']}")
        print(f"  Negative Class Recall:  {neg_analysis['negative_recall']:.4f}")
        
        if neg_analysis['misclassified_as']:
            print(f"  Misclassified As:")
            for label, count in neg_analysis['misclassified_as'].items():
                print(f"    {label}: {count} samples")
        
        print(f"\n📊 Bias Analysis:")
        bias_analysis = metrics['bias_analysis']
        for class_name, bias_score in bias_analysis['bias_scores'].items():
            bias_direction = "over-predicts" if bias_score > 0 else "under-predicts"
            print(f"  {class_name}: {bias_direction} by {abs(bias_score):.3f}")
        
        print("="*80)

def main():
    """Run complete BERT+MLP evaluation"""
    print("🚀 Starting BERT+MLP Model Evaluation...")
    
    try:
        evaluator = BertMLPEvaluator()
        
        # Evaluate on validation set
        results = evaluator.evaluate_dataset()
        if results is None:
            return
        
        # Calculate metrics
        print("\n📊 Calculating enhanced metrics...")
        metrics = evaluator.calculate_advanced_metrics(results)
        
        # Print results
        evaluator.print_enhanced_metrics(metrics)
        
        # Save results
        output_dir = evaluator.save_results(metrics, results)
        
        print(f"\n✅ Enhanced evaluation completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📈 Visualizations created:")
        print(f"  - Confusion Matrix: {output_dir}/bert_mlp_confusion_matrix.png")
        print(f"  - Negative Class Analysis: {output_dir}/negative_class_analysis.png")
        print(f"  - Bias Analysis: {output_dir}/bias_analysis.png")
        print(f"  - Confidence Distribution: {output_dir}/confidence_distribution.png")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
