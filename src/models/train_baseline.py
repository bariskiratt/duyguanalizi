import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from typing import Dict, Any
from typing import cast, Dict, Any
import numpy as np
import pickle

def save_model(model, vectorizer, output_dir, model_name="multinomial_lr_model"):
    """
    Eğitilmiş modeli ve vectorizer'ı kaydet
    """
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    vectorizer_path = os.path.join(output_dir, f"{model_name}_vectorizer.pkl")
    
    # Model ve vectorizer'ı birlikte kaydet
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model kaydedildi: {model_path}")
    print(f"Vectorizer kaydedildi: {vectorizer_path}")

def load_model(output_dir, model_name="multinomial_lr_model"):
    """
    Kaydedilmiş modeli ve vectorizer'ı yükle
    """
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    vectorizer_path = os.path.join(output_dir, f"{model_name}_vectorizer.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """
    Tek bir metin için duygu analizi yap
    """
    # Metni vektörize et
    text_vectorized = vectorizer.transform([text])
    
    # Tahmin yap
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Sonuçları formatla
    result = {
        'text': text,
        'prediction': prediction,
        'probabilities': {
            'pozitif': float(probabilities[0]),
            'notr': float(probabilities[1]),
            'negatif': float(probabilities[2])
        }
    }
    
    return result

def predict_batch(texts, model, vectorizer):
    """
    Birden fazla metin için duygu analizi yap
    """
    # Metinleri vektörize et
    texts_vectorized = vectorizer.transform(texts)
    
    # Tahminler yap
    predictions = model.predict(texts_vectorized)
    probabilities = model.predict_proba(texts_vectorized)
    
    # Sonuçları formatla
    results = []
    for i, text in enumerate(texts):
        result = {
            'text': text,
            'prediction': predictions[i],
            'probabilities': {
                'pozitif': float(probabilities[i][0]),
                'notr': float(probabilities[i][1]),
                'negatif': float(probabilities[i][2])
            }
        }
        results.append(result)
    
    return results

def hyperparameter_tuning(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """
    Multinomial Logistic Regression için hyperparameter tuning
    """
    print("Hyperparameter tuning başlatılıyor...")
    
    # Grid search parametreleri
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'max_iter': [500, 1000, 2000],
        'class_weight': ['balanced', None]
    }
    
    # Base model
    base_model = LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42
    )
    
    # Grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    # Eğitim
    grid_search.fit(X_train_tfidf, y_train)
    
    # En iyi parametreleri yazdır
    print(f"En iyi parametreler: {grid_search.best_params_}")
    print(f"En iyi cross-validation skoru: {grid_search.best_score_:.4f}")
    
    # En iyi model ile test
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test doğruluğu: {test_accuracy:.4f}")
    
    return best_model, grid_search.best_params_

def train_baseline(input_parquet="data/processed/clean.parquet", # Temizlenmiş ve Parquet formatında kaydedilmiş verinin yolu.
                   output_dir="artifacts/baseline",    # Model sonuçlarının (metrik, grafik) kaydedileceği klasör
                   use_hyperparameter_tuning=False,    # Hyperparameter tuning kullanılsın mı?
                   save_model_flag=True):              # Model kaydedilsin mi?
    
    
    os.makedirs(output_dir, exist_ok=True)
    # Veri yükle
    df = pd.read_parquet(input_parquet) # clean.parquet dosyasını DataFrame'e çevirir.
    X = df["review_text"]   # Bağımsız değişken, yani her bir yorumun metni.
    y = df["label"]   # Hedef değişken — modelin tahmin etmesini istediğimiz "negatif/ notr/ pozitif" etiketler.

    # Sınıf dağılımını kontrol et
    print("Sınıf dağılımı:")
    print(y.value_counts())
    print(f"Toplam örnek sayısı: {len(y)}")

    stratify_param = None
    counts = y.value_counts()
    if counts.min() >= 2:
        stratify_param = y
    else:
        print(" Stratify için bazı sınıflarda <2 örnek var,", 
              "stratify devre dışı bırakıldı.")
        
    
    # Split
    # Stratify parametresi, sınıf dağılımını korumak için kullanılır.
    # Eğer bazı sınıflarda örnek sayısı 2'den az ise, strat
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,   # Verinin %20'si test, %80'i eğitim.
        random_state=42, #Bölmeyi tekrarlanabilir yapar (aynı rastgelelik)
        stratify=stratify_param  #  Dengeli dağılım için
    )

    print(f"Eğitim seti boyutu: {len(X_train)}")
    print(f"Test seti boyutu: {len(X_test)}")

    # TF-IDF (unigram + bigram)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)  # Unigram ve bigram kullanarak TF-IDF vektörleştirme
    # max_features=5000 ile en sık geçen 5000 kelimeyi alır
    X_train_tfidf = vectorizer.fit_transform(X_train)  # Eğitim verisini TF-IDF ile dönüştürür
    X_test_tfidf = vectorizer.transform(X_test)  # Test verisini TF-IDF ile dönüştürür

    print(f"TF-IDF özellik sayısı: {X_train_tfidf.shape[1]}")

    # Model seçimi
    if use_hyperparameter_tuning:
        clf, best_params = hyperparameter_tuning(X_train_tfidf, y_train, X_test_tfidf, y_test)
        print("Hyperparameter tuning ile en iyi model kullanılıyor.")
    else:
        # Multinomial Logistic Regression - İyileştirilmiş parametreler
        # solver='lbfgs': Multinomial logistic regression için en iyi solver
        # multi_class='multinomial': Açıkça multinomial belirtiyoruz
        # max_iter=1000: Daha fazla iterasyon
        # C=1.0: Regularization parametresi (daha küçük = daha fazla regularization)
        # class_weight='balanced': Sınıf dengesizliği için
        clf = LogisticRegression(
            solver='lbfgs',           # Multinomial için en iyi solver
            multi_class='multinomial', # Açıkça multinomial belirtiyoruz
            max_iter=1000,            # Daha fazla iterasyon
            C=1.0,                    # Regularization parametresi
            class_weight='balanced',   # Sınıf dengesizliği için
            random_state=42           # Tekrarlanabilirlik için
        )
        
        print("Model eğitiliyor...")
        clf.fit(X_train_tfidf, y_train)  # Modeli eğitim verisi ile eğitir
    
    y_pred = clf.predict(X_test_tfidf)   # Tahmin
    y_pred_proba = clf.predict_proba(X_test_tfidf)  # Olasılık tahminleri

    # Model performansını değerlendir
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Genel doğruluk: {accuracy:.4f}")

    # Rapor & Confusion Matrix
    report = cast(Dict[str, Any], classification_report(y_test, y_pred, output_dict=True))
    #report = classification_report(y_test, y_pred, output_dict=True)
    with open(f"{output_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    

    
    print("Macro F1:", report["macro avg"]["f1-score"])
    # Sınıf yoksa 0.0 yaz
    for c in ["pozitif", "negatif", "notr"]:
        f1 = report[c]["f1-score"] if c in report else 0.0
        print(f"{c}: {f1}")

    # Model katsayılarını analiz et
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_
    
    # Her sınıf için en önemli özellikleri bul
    print("\nEn önemli özellikler (her sınıf için):")
    for i, class_name in enumerate(clf.classes_):
        top_indices = np.argsort(coefs[i])[-10:]  # En yüksek 10 katsayı
        print(f"\n{class_name} sınıfı için en önemli özellikler:")
        for idx in reversed(top_indices):
            print(f"  {feature_names[idx]}: {coefs[i][idx]:.4f}")

    # Confusion Matrix
        #seaborn.heatmap: Matris değerlerini renkli ısı haritası olarak çizer, hücre içine sayıyı yazar.
    cm = confusion_matrix(y_test, y_pred, labels=["pozitif","notr","negatif"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=["pozitif","notr","negatif"], 
                yticklabels=["pozitif","notr","negatif"])
    
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix - Multinomial Logistic Regression")
    plt.savefig(f"{output_dir}/confusion_matrix.png")  # Grafiği Kaydeder
    plt.close()

    # Olasılık dağılımını görselleştir
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(clf.classes_):
        plt.subplot(1, 3, i+1)
        plt.hist(y_pred_proba[:, i], bins=20, alpha=0.7)
        plt.title(f'{class_name} Olasılık Dağılımı')
        plt.xlabel('Olasılık')
        plt.ylabel('Frekans')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/probability_distribution.png")
    plt.close()

    # Model bilgilerini kaydet
    model_info = {
        "model_type": "Multinomial Logistic Regression",
        "solver": clf.solver,
        "multi_class": clf.multi_class,
        "max_iter": clf.max_iter,
        "C": clf.C,
        "class_weight": clf.class_weight,
        "feature_count": X_train_tfidf.shape[1],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "hyperparameter_tuning_used": use_hyperparameter_tuning
    }
    
    with open(f"{output_dir}/model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    # Modeli kaydet
    if save_model_flag:
        save_model(clf, vectorizer, output_dir)

    print(f"\nModel bilgileri {output_dir}/model_info.json dosyasına kaydedildi.")
    print("Multinomial Logistic Regression eğitimi tamamlandı!")

if __name__ == "__main__":
    # Normal eğitim
    train_baseline()
    
    # Hyperparameter tuning ile eğitim (isteğe bağlı)
    # train_baseline(use_hyperparameter_tuning=True)
