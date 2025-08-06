import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open('trained_model.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('trained_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Test sentences (Turkish)
test_sentences = [
    "Bu ürün gerçekten harika",
    "Çok kötü bir deneyim yaşadım",
    "Orta karar bir ürün",
    "Mükemmel kalite ve hızlı teslimat",
    "Hiç beğenmedim, para israfı",
    "Fena değil ama daha iyi olabilir",
    "Kesinlikle tavsiye ederim",
    "Berbat kalite, hiç çalışmıyor",
    "Normal bir ürün, beklentilerimi karşıladı",
    "Çok güzel ve kaliteli bir ürün"
]

print("=== TÜRKÇE SENTIMENT ANALİZİ TEST SONUÇLARI ===\n")

# Test each sentence
for i, sentence in enumerate(test_sentences, 1):
    # Transform the sentence
    X_test = vectorizer.transform([sentence])
    
    # Get prediction and probabilities
    prediction = clf.predict(X_test)[0]
    probabilities = clf.predict_proba(X_test)[0]
    
    # Get confidence (highest probability)
    confidence = max(probabilities)
    
    print(f"Test {i}: '{sentence}'")
    print(f"  Tahmin: {prediction}")
    print(f"  Güven: {confidence:.1%}")
    print(f"  Olasılıklar: Pozitif={probabilities[2]:.1%}, Negatif={probabilities[0]:.1%}, Nötr={probabilities[1]:.1%}")
    print("-" * 60)