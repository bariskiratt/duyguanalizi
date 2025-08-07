# Multinomial Logistic Regression - Duygu Analizi

Bu proje, Türkçe e-ticaret yorumları için multinomial logistic regression kullanarak duygu analizi yapar.

## 🎯 Özellikler

- **Multinomial Logistic Regression**: 3 sınıflı duygu analizi (pozitif, notr, negatif)
- **TF-IDF Vektörizasyonu**: Unigram ve bigram özellikleri
- **Hyperparameter Tuning**: Grid search ile otomatik parametre optimizasyonu
- **Model Kaydetme/Yükleme**: Eğitilmiş modeli kaydetme ve yeniden kullanma
- **Detaylı Analiz**: Confusion matrix, olasılık dağılımları, özellik önemleri
- **Batch Tahmin**: Tek tek veya toplu tahmin yapabilme

## 📁 Dosya Yapısı

```
src/models/
├── train_baseline.py          # Ana eğitim script'i
├── demo_multinomial_lr.py     # Demo script'i
└── README.md                  # Bu dosya
```

## 🚀 Kullanım

### 1. Model Eğitimi

```python
from train_baseline import train_baseline

# Basit eğitim
train_baseline(
    input_parquet="data/processed/clean.parquet",
    output_dir="artifacts/baseline",
    use_hyperparameter_tuning=False,
    save_model_flag=True
)

# Hyperparameter tuning ile eğitim
train_baseline(
    input_parquet="data/processed/clean.parquet",
    output_dir="artifacts/baseline_tuned",
    use_hyperparameter_tuning=True,
    save_model_flag=True
)
```

### 2. Model Yükleme ve Tahmin

```python
from train_baseline import load_model, predict_sentiment, predict_batch

# Modeli yükle
model, vectorizer = load_model("artifacts/baseline")

# Tek metin tahmini
result = predict_sentiment("Bu ürün harika!", model, vectorizer)
print(result)
# Çıktı: {'text': 'Bu ürün harika!', 'prediction': 'pozitif', 'probabilities': {...}}

# Toplu tahmin
texts = ["Ürün iyi", "Ürün kötü", "Ürün normal"]
results = predict_batch(texts, model, vectorizer)
```

### 3. Demo Script'i

```bash
cd src/models
python demo_multinomial_lr.py
```

## 🔧 Model Parametreleri

### Multinomial Logistic Regression
- **solver**: 'lbfgs' (multinomial için en iyi)
- **multi_class**: 'multinomial' (açıkça belirtiliyor)
- **max_iter**: 1000 (daha fazla iterasyon)
- **C**: 1.0 (regularization parametresi)
- **class_weight**: 'balanced' (sınıf dengesizliği için)

### TF-IDF Vektörizasyonu
- **ngram_range**: (1,2) (unigram + bigram)
- **max_features**: 5000 (en sık geçen 5000 kelime)

### Hyperparameter Tuning
- **C**: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
- **max_iter**: [500, 1000, 2000]
- **class_weight**: ['balanced', None]

## 📊 Çıktılar

### Metrikler (`metrics.json`)
```json
{
  "pozitif": {"precision": 0.85, "recall": 0.80, "f1-score": 0.82},
  "notr": {"precision": 0.75, "recall": 0.70, "f1-score": 0.72},
  "negatif": {"precision": 0.90, "recall": 0.85, "f1-score": 0.87},
  "macro avg": {"f1-score": 0.80}
}
```

### Görselleştirmeler
- `confusion_matrix.png`: Confusion matrix
- `probability_distribution.png`: Sınıf olasılık dağılımları

### Model Bilgileri (`model_info.json`)
```json
{
  "model_type": "Multinomial Logistic Regression",
  "solver": "lbfgs",
  "multi_class": "multinomial",
  "accuracy": 0.82,
  "macro_f1": 0.80
}
```

## 🎓 Multinomial Logistic Regression Nedir?

Multinomial logistic regression, ikiden fazla sınıf için kullanılan bir sınıflandırma algoritmasıdır.

### Matematiksel Temel
```
P(Y=k|X) = exp(βₖᵀX) / Σᵢexp(βᵢᵀX)
```

### Avantajları
- ✅ **Yorumlanabilirlik**: Katsayılar özellik önemini gösterir
- ✅ **Olasılık Çıktısı**: Her sınıf için olasılık verir
- ✅ **Hızlı Eğitim**: Diğer algoritmalara göre hızlı
- ✅ **Regularization**: Overfitting'i önler

### Dezavantajları
- ❌ **Lineer Sınırlar**: Karmaşık non-lineer ilişkileri yakalayamaz
- ❌ **Özellik Seçimi**: Manuel özellik mühendisliği gerekebilir

## 🔍 Özellik Analizi

Model, her sınıf için en önemli özellikleri (kelimeleri) gösterir:

```
pozitif sınıfı için en önemli özellikler:
  harika: 2.3456
  mükemmel: 2.1234
  tavsiye: 1.9876

negatif sınıfı için en önemli özellikler:
  kötü: -2.5678
  berbat: -2.3456
  iade: -1.9876
```

## 🚀 Performans İyileştirme

### 1. Veri Kalitesi
- Daha fazla veri toplayın
- Veri temizliğini iyileştirin
- Sınıf dengesizliğini kontrol edin

### 2. Özellik Mühendisliği
- Farklı n-gram kombinasyonları deneyin
- Kelime köklerini kullanın
- Sentiment sözlükleri ekleyin

### 3. Model Optimizasyonu
- Hyperparameter tuning kullanın
- Cross-validation ile değerlendirin
- Ensemble yöntemleri deneyin

## 📝 Örnek Kullanım Senaryoları

### Senaryo 1: Yeni Model Eğitimi
```python
# Veriyi hazırla ve modeli eğit
train_baseline(
    input_parquet="data/processed/clean.parquet",
    output_dir="artifacts/new_model",
    use_hyperparameter_tuning=True
)
```

### Senaryo 2: Mevcut Model ile Tahmin
```python
# Modeli yükle
model, vectorizer = load_model("artifacts/baseline")

# Yeni yorumları analiz et
new_reviews = ["Harika ürün!", "Kötü kalite", "Normal"]
results = predict_batch(new_reviews, model, vectorizer)
```

### Senaryo 3: Model Performansını Değerlendir
```python
# Model bilgilerini oku
import json
with open("artifacts/baseline/model_info.json", "r") as f:
    model_info = json.load(f)
    
print(f"Model doğruluğu: {model_info['accuracy']}")
print(f"Macro F1: {model_info['macro_f1']}")
```

## 🛠️ Sorun Giderme

### Yaygın Hatalar

1. **Model bulunamadı hatası**
   ```python
   # Çözüm: Önce eğitim yapın
   train_baseline(save_model_flag=True)
   ```

2. **Memory hatası**
   ```python
   # Çözüm: max_features'ı azaltın
   vectorizer = TfidfVectorizer(max_features=2000)
   ```

3. **Convergence hatası**
   ```python
   # Çözüm: max_iter'ı artırın
   clf = LogisticRegression(max_iter=2000)
   ```

## 📚 Referanslar

- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Multinomial Logistic Regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) 