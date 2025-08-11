import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

class BERTDemo:
    def __init__(self, model_path="./artifacts/bert_ckpt/best_model", config_path="src/configs/bert_hparams.yaml"):
        """Interactive BERT sentiment analysis demo"""
        
        print("🚀 Loading BERT Sentiment Analysis Demo...")
        
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
        self.color_map = {0: "\033[91m", 1: "\033[93m", 2: "\033[92m"}  # Red, Yellow, Green
        self.reset_color = "\033[0m"
        
        print("✅ Model loaded successfully!\n")
    
    def predict(self, text):
        """Predict sentiment for text"""
        if not text.strip():
            return None
            
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
        
        return {
            'label': predicted_label,
            'sentiment': self.label_map[predicted_label],
            'emoji': self.emoji_map[predicted_label],
            'confidence': confidence,
            'probabilities': {
                'Negative': all_probs[0],
                'Neutral': all_probs[1], 
                'Positive': all_probs[2]
            }
        }
    
    def print_result(self, text, result):
        """Pretty print result with colors"""
        if result is None:
            return
            
        label = result['label']
        color = self.color_map[label]
        
        print(f"\n📝 Text: {text}")
        print(f"🎯 Sentiment: {color}{result['sentiment']} {result['emoji']}{self.reset_color}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print("\n📈 Probability Distribution:")
        
        for sentiment, prob in result['probabilities'].items():
            bar_length = int(prob * 30)
            if sentiment == result['sentiment']:
                bar_color = color
            else:
                bar_color = ""
            
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {sentiment:8s}: {prob:.3f} |{bar_color}{bar}{self.reset_color}|")
    
    def show_examples(self):
        """Show some example predictions"""
        examples = [
            "Bu film gerçekten muhteşem, herkese tavsiye ederim!",
            "Çok kötü bir deneyimdi, hiç beğenmedim.",
            "Normal bir gün geçirdim, ne özel bir şey yok.",
            "Harika bir restoran, yemekler çok lezzetli!",
            "Berbat bir hizmet, bir daha gelmem.",
            "İdare eder, fena değil ama süper de değil."
        ]
        
        print("💡 Here are some example predictions:\n")
        print("="*70)
        
        for text in examples:
            result = self.predict(text)
            self.print_result(text, result)
            print("-" * 70)
    
    def interactive_mode(self):
        """Run interactive prediction mode"""
        print("🎮 Interactive Mode - Type sentences to analyze!")
        print("Commands:")
        print("  - Type any text to analyze sentiment")
        print("  - 'examples' to see example predictions")
        print("  - 'help' to show this help")
        print("  - 'quit' or 'exit' to quit")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n💬 Enter text (or command): ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  - Type any text to analyze sentiment")
                    print("  - 'examples' to see example predictions")
                    print("  - 'help' to show this help")
                    print("  - 'quit' or 'exit' to quit")
                    continue
                elif user_input.lower() == 'examples':
                    self.show_examples()
                    continue
                
                # Predict sentiment
                result = self.predict(user_input)
                self.print_result(user_input, result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def batch_mode(self, texts):
        """Process multiple texts at once"""
        print(f"📦 Batch Mode - Processing {len(texts)} texts...\n")
        print("="*70)
        
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}/{len(texts)}]")
            result = self.predict(text)
            self.print_result(text, result)
            print("-" * 70)

def main():
    """Main demo function"""
    try:
        demo = BERTDemo()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            # Batch mode with command line arguments
            texts = sys.argv[1:]
            demo.batch_mode(texts)
        else:
            # Interactive mode
            demo.interactive_mode()
            
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files.")
        print(f"Make sure you've trained your BERT model first!")
        print(f"Details: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
