import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

df = pd.read_parquet('data/processed/clean.parquet')
X = df['review_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train) #train datasindaki kelimeleri vektörleştirir tf idf parametrelerini hesaplar
X_test_tfidf = vectorizer.transform(X_test)  # test datasindaki kelimeleri vektorlestirip tf idf ile skor cikarir ayrica

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)
y_pred_proba = clf.predict_proba(X_test_tfidf)
high_conf_predictions = []
for i, probs in enumerate(y_pred_proba):
    max_prob = max(probs)
    if max_prob > 0.8:  # 80% confidence
        high_conf_predictions.append(i)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("High Confidence Predictions: ", len(high_conf_predictions))
confmatrix = confusion_matrix(y_test, y_pred)

with open('confmatrix.json', 'w') as f:
    json.dump(confmatrix.tolist(), f)  # Convert numpy array to list

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('trained_vectorizer.pkl', 'wb') as f:
<<<<<<< HEAD
    pickle.dump(vectorizer, f)
=======
    pickle.dump(vectorizer, f)





>>>>>>> cb630d222949a11e9bd6056e25e58c81e74d58c7
