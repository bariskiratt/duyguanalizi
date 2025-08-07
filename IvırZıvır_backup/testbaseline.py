import pandas as pd
import pickle 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Türkçe font desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

# Klasör oluştur
Path("artifacts/baseline").mkdir(parents=True, exist_ok=True)

with open("src/saves/trained_model.pkl", 'rb') as f:
    clf = pickle.load(f)

with open("src/saves/trained_vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

test_data = pd.read_parquet("data/processed/test_data.parquet")

y_true = test_data['label']
X_test_texts = test_data['review_text']

X_test_vectorized = vectorizer.transform(X_test_texts)

y_pred = clf.predict(X_test_vectorized)
y_pred_proba = clf.predict_proba(X_test_vectorized)

# ============================================================================
# 📊 METRİKLERİ HESAPLA VE TABLO HALİNE GETİR
# ============================================================================

accuracy = accuracy_score(y_true, y_pred)

# Classification report'u DataFrame'e çevir
report = classification_report(y_true, y_pred, output_dict=True)

# Sınıf bazlı metrikler
class_metrics = []
for class_name in ['pozitif', 'negatif', 'notr']:
    if class_name in report:
        metrics = report[class_name]
        class_metrics.append({
            'Sınıf': class_name,
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1-score']:.3f}",
            'Support': metrics['support']
        })

metrics_df = pd.DataFrame(class_metrics)

# Sadece classification report ve sınıf metriklerini yazdır
print("=" * 80)
print("🎯 DUYGU ANALİZİ MODELİ TEST SONUÇLARI")
print("=" * 80)

print(f"\n📊 GENEL PERFORMANS:")
print(f"   Accuracy: {accuracy:.3f} ({accuracy:.1%})")

print(f"\n📋 SINIF BAZLI METRİKLER:")
print(metrics_df.to_string(index=False))

print(f"\n📝 DETAYLI CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred))

# ============================================================================
# 🔍 NÖTR TAHMİNLERİ ANALİZ ET (SESSİZ)
# ============================================================================

neutral_predictions = []
for i, (text, pred, true_label) in enumerate(zip(X_test_texts, y_pred, y_true)):
    if pred == 'notr':
        neutral_predictions.append({
            'Cümle': text,
            'Tahmin': pred,
            'Gerçek': true_label,
            'Güven': f"{max(y_pred_proba[i]):.1%}",
            'Nötr_Olasılık': f"{y_pred_proba[i][1]:.1%}"
        })

neutral_df = pd.DataFrame(neutral_predictions)

# ============================================================================
# 📈 KELİME ÖNEM TABLOLARI (SESSİZ)
# ============================================================================

feature_names = vectorizer.get_feature_names_out()
coefficients = clf.coef_[1]  # nötr sınıfı için coefficient'lar

word_weights = pd.DataFrame({
    'Kelime': feature_names,
    'Ağırlık': coefficients
})

# En önemli pozitif ve negatif kelimeler
top_positive = word_weights.nlargest(15, 'Ağırlık')[['Kelime', 'Ağırlık']]
top_negative = word_weights.nsmallest(15, 'Ağırlık')[['Kelime', 'Ağırlık']]

# ============================================================================
# 🎨 CONFUSION MATRIX GÖRSELİ (SESSİZ)
# ============================================================================

labels_sorted = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_sorted, yticklabels=labels_sorted,
            cbar_kws={'label': 'Örnek Sayısı'})
plt.title("Confusion Matrix - Duygu Analizi Modeli", fontsize=14, pad=20)
plt.xlabel("Tahmin Edilen", fontsize=12)
plt.ylabel("Gerçek", fontsize=12)
plt.tight_layout()
plt.savefig("artifacts/baseline/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 💾 SONUÇLARI DOSYALARA KAYDET (SESSİZ)
# ============================================================================

# Ana metrikler
results = {
    'accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('artifacts/baseline/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Tabloları kaydet
metrics_df.to_csv('artifacts/baseline/classification_metrics.csv', index=False, encoding='utf-8')
neutral_df.to_csv('artifacts/baseline/neutral_predictions.csv', index=False, encoding='utf-8')
word_weights.to_csv('artifacts/baseline/word_weights.csv', index=False, encoding='utf-8')

# ============================================================================
# 🔬 CÜMLE ANALİZİ (SESSİZ)
# ============================================================================

def analyze_sentence_weights(sentence, vectorizer, clf, feature_names):
    """Belirli bir cümle için kelime ağırlıklarını analiz et"""
    sentence_vector = vectorizer.transform([sentence])
    word_indices = sentence_vector.indices
    word_weights_in_sentence = []
    
    for idx in word_indices:
        word = feature_names[idx]
        weight = clf.coef_[1][idx]  # nötr sınıfı için ağırlık
        word_weights_in_sentence.append((word, weight))
    
    return sorted(word_weights_in_sentence, key=lambda x: x[1], reverse=True)

# Örnek bir nötr cümle analizi (sessiz)
if len(neutral_df) > 0:
    sample_sentence = neutral_df.iloc[0]['Cümle']
    word_analysis = analyze_sentence_weights(sample_sentence, vectorizer, clf, feature_names)
    
    # Cümle analizini dosyaya kaydet
    sentence_analysis = {
        'sentence': sample_sentence,
        'word_weights': [(word, float(weight)) for word, weight in word_analysis]
    }
    
    with open('artifacts/baseline/sentence_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(sentence_analysis, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("✅ TEST TAMAMLANDI!")
print("=" * 80)
