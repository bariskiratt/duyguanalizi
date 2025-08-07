import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path


output_dir = str = "artifacts/baseline"
df = pd.read_parquet("data/processed/train_data.parquet")
X_train = df['review_text']
y_train = df['label']


vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train) #train datasindaki kelimeleri vektörleştirir tf idf parametrelerini hesaplar

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)



folder = Path("")

# 2) Build full paths once
model_path      = folder / "src/saves/trained_model.pkl"
vectorizer_path = folder / "src/saves/trained_vectorizer.pkl"

# 3) Save objects
with model_path.open("wb") as f:
    pickle.dump(clf, f)

with vectorizer_path.open("wb") as f:
    pickle.dump(vectorizer, f)

print("Model ve vectorizer kaydedildi")
print("Train datası ile eğitildi")


