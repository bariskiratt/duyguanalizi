# 🤖 BERT Sentiment Analysis - Usage Guide

Your BERT model is now trained and ready to use! Here's how to use all the tools I've created for you.

## 🎯 Quick Start

### 1. **Test Single Sentences** 
```bash
python src/models/bert/bert_predict.py
```
- Tests your model with sample Turkish sentences
- Shows confidence scores and probability distributions
- Perfect for quick testing

### 2. **Interactive Demo**
```bash
python src/models/bert/bert_demo.py
```
- Type sentences interactively and get instant predictions
- Supports commands: `examples`, `help`, `quit`
- Great for testing your own sentences

### 3. **Evaluate on Validation Set**
```bash
python src/models/bert/bert_evaluate.py
```
- Tests on your entire validation dataset
- Generates confusion matrix and detailed metrics
- Saves results to `artifacts/bert_evaluation/`

### 4. **Compare with Baseline**
```bash
python src/models/bert/bert_compare.py
```
- Compares BERT vs your baseline model performance
- Creates visualization plots
- Shows which model performs better

## 📁 Output Files

After running the scripts, you'll find results in:

```
artifacts/
├── bert_evaluation/
│   ├── bert_metrics.json          # Detailed performance metrics
│   ├── bert_predictions.csv       # All predictions with confidence
│   ├── bert_confusion_matrix.png  # Confusion matrix plot
│   ├── model_comparison.png       # BERT vs Baseline comparison
│   └── comparison_report.json     # Comprehensive comparison
└── bert_ckpt/
    └── best_model/                # Your trained model files
        ├── model.safetensors
        ├── config.json
        └── tokenizer files...
```

## 🚀 Advanced Usage

### Batch Predictions
```bash
# Predict multiple sentences at once
python src/models/bert/bert_demo.py "İlk cümle" "İkinci cümle" "Üçüncü cümle"
```

### Custom Model Path
```python
from bert_predict import BERTPredictor

# Use different model path
predictor = BERTPredictor(model_path="./custom/model/path")
result = predictor.predict_single("Test sentence")
```

## 📊 Understanding Results

### Sentiment Labels
- **0 (Negative)** 😞 - Olumsuz duygu
- **1 (Neutral)** 😐 - Nötr duygu  
- **2 (Positive)** 😊 - Olumlu duygu

### Key Metrics
- **Accuracy**: Overall correct predictions
- **F1 Score**: Balance of precision and recall
- **Confidence**: Model's certainty (0.0 to 1.0)

## 🛠️ Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   # Make sure you're in the right environment
   python -c "import torch; print('OK')"
   ```

2. **"Model files not found"**
   - Ensure training completed successfully
   - Check `artifacts/bert_ckpt/best_model/` exists

3. **CUDA out of memory**
   - Reduce batch size in evaluation scripts
   - Use CPU mode: comment out `.to(device)` lines

4. **Validation dataset not found**
   - Run data preparation first:
   ```bash
   python src/data/prepare_bert_data.py
   ```

## 🎮 Example Session

```bash
$ python src/models/bert/bert_demo.py

🚀 Loading BERT Sentiment Analysis Demo...
Using device: cuda
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
✅ Model loaded successfully!

🎮 Interactive Mode - Type sentences to analyze!
💬 Enter text: Bu film harika, çok beğendim!

📝 Text: Bu film harika, çok beğendim!
🎯 Sentiment: Positive 😊
📊 Confidence: 0.892

📈 Probability Distribution:
  Negative: 0.021 |██░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
  Neutral : 0.087 |████░░░░░░░░░░░░░░░░░░░░░░░░░░|
  Positive: 0.892 |██████████████████████████████|
```

## 🏆 Next Steps

1. **Production Deployment**: Use `bert_predict.py` as base for API
2. **Model Improvement**: Collect more training data
3. **Domain Adaptation**: Fine-tune on specific domains
4. **Performance Optimization**: Convert to ONNX for faster inference

Your BERT model is now ready for real-world sentiment analysis! 🎉
