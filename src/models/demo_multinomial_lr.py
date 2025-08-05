#!/usr/bin/env python3
"""
Multinomial Logistic Regression Demo Script
Bu script, eğitilmiş multinomial logistic regression modelini nasıl kullanacağınızı gösterir.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_baseline import train_baseline, load_model, predict_sentiment, predict_batch

def demo_training():
    """
    Model eğitimi demo'su
    """
    print("=" * 50)
    print("MULTINOMIAL LOGISTIC REGRESSION EĞİTİMİ")
    print("=" * 50)
    
    # Model eğitimi
    train_baseline(
        input_parquet="data/processed/clean.parquet",
        output_dir="artifacts/baseline",
        use_hyperparameter_tuning=False,  # True yaparak hyperparameter tuning kullanabilirsiniz
        save_model_flag=True
    )

def demo_prediction():
    """
    Tahmin demo'su
    """
    print("\n" + "=" * 50)
    print("TAHMIN DEMO'SU")
    print("=" * 50)
    
    # Modeli yükle
    try:
        model, vectorizer = load_model("artifacts/baseline")
        print("Model başarıyla yüklendi!")
    except FileNotFoundError:
        print("Model bulunamadı! Önce eğitim yapın.")
        return
    
    # Test metinleri
    test_texts = [
        "Bu ürün gerçekten harika, çok memnunum!",
        "Ürün beklentilerimi karşılamadı, hayal kırıklığı yaşadım.",
        "Ürün normal, ne iyi ne kötü.",
        "Mükemmel kalite, kesinlikle tavsiye ederim!",
        "Çok kötü bir deneyim, paramı geri istiyorum."
    ]
    
    print("\nTek tek tahmin örnekleri:")
    print("-" * 30)
    
    for text in test_texts:
        result = predict_sentiment(text, model, vectorizer)
        print(f"Metin: {text}")
        print(f"Tahmin: {result['prediction']}")
        print(f"Olasılıklar: {result['probabilities']}")
        print()
    
    print("\nToplu tahmin örneği:")
    print("-" * 30)
    
    batch_results = predict_batch(test_texts, model, vectorizer)
    for result in batch_results:
        print(f"Metin: {result['text']}")
        print(f"Tahmin: {result['prediction']}")
        print(f"En yüksek olasılık: {max(result['probabilities'].items(), key=lambda x: x[1])}")
        print()

def demo_hyperparameter_tuning():
    """
    Hyperparameter tuning demo'su
    """
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING DEMO'SU")
    print("=" * 50)
    
    # Hyperparameter tuning ile eğitim
    train_baseline(
        input_parquet="data/processed/clean.parquet",
        output_dir="artifacts/baseline_tuned",
        use_hyperparameter_tuning=True,
        save_model_flag=True
    )

def main():
    """
    Ana demo fonksiyonu
    """
    print("Multinomial Logistic Regression Demo")
    print("Bu script size multinomial logistic regression modelinin nasıl kullanılacağını gösterir.")
    
    while True:
        print("\nSeçenekler:")
        print("1. Model eğitimi")
        print("2. Tahmin demo'su")
        print("3. Hyperparameter tuning demo'su")
        print("4. Çıkış")
        
        choice = input("\nSeçiminizi yapın (1-4): ").strip()
        
        if choice == "1":
            demo_training()
        elif choice == "2":
            demo_prediction()
        elif choice == "3":
            demo_hyperparameter_tuning()
        elif choice == "4":
            print("Demo sonlandırılıyor...")
            break
        else:
            print("Geçersiz seçim! Lütfen 1-4 arasında bir sayı girin.")

if __name__ == "__main__":
    main() 