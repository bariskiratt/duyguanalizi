import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from typing import cast, Dict, Any

def train_baseline(input_parquet="data/processed/clean.parquet", # Temizlenmiş ve Parquet formatında kaydedilmiş verinin yolu.
                   output_dir="artifacts/baseline"):    # Model sonuçlarının (metrik, grafik) kaydedileceği klasör
    
    
    os.makedirs(output_dir, exist_ok=True)
    # Veri yükle
    df = pd.read_parquet(input_parquet) # clean.parquet dosyasını DataFrame’e çevirir.
    X = df["review_text"]   # Bağımsız değişken, yani her bir yorumun metni.
    y = df["label"]   # Hedef değişken — modelin tahmin etmesini istediğimiz “negatif/ notr/ pozitif” etiketler.

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
        test_size=0.2,   # Verinin %20’si test, %80’i eğitim.
        random_state=42, #Bölmeyi tekrarlanabilir yapar (aynı rastgelelik)
        stratify=stratify_param  #  Dengeli dağılım için
    )

    # TF-IDF (unigram + bigram)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)  # Unigram ve bigram kullanarak TF-IDF vektörleştirme
    # max_features=5000 ile en sık geçen 5000 kelimeyi alır
    X_train_tfidf = vectorizer.fit_transform(X_train)  # Eğitim verisini TF-IDF ile dönüştürür
    X_test_tfidf = vectorizer.transform(X_test)  # Test verisini TF-IDF ile dönüştürür

    # Logistic Regression
    #"balanced": Sınıf dengesizliği varsa ağırlıkları otomatik ayarlar
    # Varsayılan iterasyon sayısı yeterli gelmezse artırıyoruz
    clf = LogisticRegression(max_iter=200, class_weight="balanced")  
    clf.fit(X_train_tfidf, y_train)  # Modeli eğitim verisi ile eğitir
    y_pred = clf.predict(X_test_tfidf)   # Tahmin

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

    # Confusion Matrix
        #seaborn.heatmap: Matris değerlerini renkli ısı haritası olarak çizer, hücre içine sayıyı yazar.
    cm = confusion_matrix(y_test, y_pred, labels=["pozitif","notr","negatif"])
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=["pozitif","notr","negatif"], 
                yticklabels=["pozitif","notr","negatif"])
    
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix - Baseline")
    plt.savefig(f"{output_dir}/confusion_matrix.png")  # Grafiği Kaydeder
    plt.close()

if __name__ == "__main__":
    train_baseline()
