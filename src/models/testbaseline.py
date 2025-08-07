import pandas as pd
import pickle 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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

print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Classification Report: ", classification_report(y_true, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_true, y_pred))

# Nötr olarak tahmin edilen cümleleri bul
neutral_predictions = []
for i, (text, pred, true_label) in enumerate(zip(X_test_texts, y_pred, y_true)):
    if pred == 'notr':
        neutral_predictions.append({
            'text': text,
            'predicted': pred,
            'true_label': true_label,
            'confidence': max(y_pred_proba[i]),
            'neutral_prob': y_pred_proba[i][1]  # nötr olasılığı
        })

print(f"\nNötr olarak tahmin edilen cümle sayısı: {len(neutral_predictions)}")

# Nötr cümleleri göster
print("\n=== NÖTR OLARAK TAHMİN EDİLEN CÜMLELER ===")
for i, item in enumerate(neutral_predictions[:20], 1):
    print(f"{i}. Cümle: {item['text']}")
    print(f"   Tahmin: {item['predicted']}")
    print(f"   Gerçek: {item['true_label']}")
    print(f"   Güven: {item['confidence']:.1%}")
    print(f"   Nötr Olasılık: {item['neutral_prob']:.1%}")
    print("-" * 60)



# Model'in feature importance'sini al
feature_names = vectorizer.get_feature_names_out()
coefficients = clf.coef_[1]  # nötr sınıfı için coefficient'lar

# Kelime ağırlıklarını DataFrame'e çevir
word_weights = pd.DataFrame({
    'word': feature_names,
    'weight': coefficients
})

# Nötr sınıfı için en önemli kelimeleri göster
print("\n=== NÖTR SINIFI İÇİN EN ÖNEMLİ KELİMELER ===")
print("Pozitif ağırlık (nötr sınıfına katkı):")
print(word_weights.nlargest(20, 'weight')[['word', 'weight']])

print("\nNegatif ağırlık (nötr sınıfından çıkarır):")
print(word_weights.nsmallest(20, 'weight')[['word', 'weight']])
    
    
    # Belirli bir nötr cümle için kelime ağırlıklarını analiz et
def analyze_sentence_weights(sentence, vectorizer, clf, feature_names):
    # Cümleyi vectorize et
    sentence_vector = vectorizer.transform([sentence])
    
    # Hangi kelimelerin kullanıldığını bul
    word_indices = sentence_vector.indices
    word_weights_in_sentence = []
    
    for idx in word_indices:
        word = feature_names[idx]
        weight = clf.coef_[1][idx]  # nötr sınıfı için ağırlık
        word_weights_in_sentence.append((word, weight))
    
    return sorted(word_weights_in_sentence, key=lambda x: x[1], reverse=True)

# Örnek bir nötr cümle analizi
if neutral_predictions:
    sample_sentence = neutral_predictions[0]['text']
    print(f"\n=== CÜMLE ANALİZİ: '{sample_sentence}' ===")
    
    word_analysis = analyze_sentence_weights(sample_sentence, vectorizer, clf, feature_names)
    print("Kelime ağırlıkları:")
    for word, weight in word_analysis:
        print(f"  {word}: {weight:.4f}")



# Nötr cümleleri DataFrame'e çevir ve kaydet
neutral_df = pd.DataFrame(neutral_predictions)
neutral_df.to_csv('neutral_predictions.csv', index=False)

# Kelime ağırlıklarını kaydet
word_weights.to_csv('neutral_word_weights.csv', index=False)

print(f"\nSonuçlar kaydedildi:")
print("- neutral_predictions.csv")
print("- neutral_word_weights.csv")

# confusion matrix görseli
labels_sorted = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=labels_sorted, yticklabels=labels_sorted)
plt.title("Confusion Matrix – Baseline")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
