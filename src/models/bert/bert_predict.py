import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
class BERTPredictor:
    def __init__(self, model_path="./artifacts/bert_ckpt/best_model", config_path="src/configs/bert_hparams.yaml"):
        """BERT model predictor for sentiment analysis"""
        
        # Resolve paths relative to this file
        current_file_dir = os.path.dirname(__file__)
        src_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
        project_dir = os.path.abspath(os.path.join(src_dir, '..'))

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
        
        # Label mappings (consistent with dataset: 0=Negative, 1=Positive, 2=Neutral)
        self.label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        self.emoji_map = {0: "😞", 1: "😊", 2: "😐"}
        
        print("✅ Model loaded successfully!")
    
    def predict_single(self, text):
        """Predict sentiment for a single text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config['model']['max_length'],
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_label = torch.argmax(logits, dim=-1).item()
        
        # Format results
        confidence = probabilities[0][predicted_label].item()
        all_probs = probabilities[0].cpu().numpy()
        
        result = {
            'text': text,
            'predicted_label': predicted_label,
            'predicted_sentiment': self.label_map[predicted_label],
            'emoji': self.emoji_map[predicted_label],
            'confidence': confidence,
            'all_probabilities': {
                'Negative': all_probs[0],
                'Neutral': all_probs[1],    
                'Positive': all_probs[2]
            }
        }
        
        return result
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def print_prediction(self, result):
        """Pretty print prediction result"""
        print("\n" + "="*60)
        print(f"📝 Text: {result['text']}")
        print(f"🎯 Prediction: {result['predicted_sentiment']} {result['emoji']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print("\n📈 All Probabilities:")
        for sentiment, prob in result['all_probabilities'].items():
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {sentiment:8s}: {prob:.3f} |{bar}|")
        print("="*60)

def main():
    """Test the predictor with sample texts"""
    print("🚀 Loading BERT Sentiment Predictor...")
    
    try:
        predictor = BERTPredictor()
        
        # Test samples (Turkish)
        test_texts = [
            "Bu film gerçekten harika, çok beğendim!",
            "Kötü bir deneyimdi, hiç memnun kalmadım.",
            "Normal bir gün geçirdim, ne iyi ne kötü.",
            "Mükemmel bir hizmet, herkese tavsiye ederim!",
            "Berbat bir uygulama, hiç kullanmayın.",
            "İdare eder, fena değil ama süper de değil."
        ]
        
        print(f"\n🧪 Testing with {len(test_texts)} sample texts...\n")
        
        for text in test_texts:
            result = predictor.predict_single(text)
            predictor.print_prediction(result)
        
        print("\n✅ Testing completed!")
        print("\n💡 You can now use this predictor in your applications!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your model is trained and saved correctly.")

if __name__ == "__main__":
    main()
