import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet('data/processed/clean.parquet')
X = df['review_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train) #train datasindaki kelimeleri vektörleştirir tf idf parametrelerini hesaplar
X_test_tfidf = vectorizer.transform(X_test)  # test datasindaki kelimeleri vektorlestirip tf idf ile skor cikarir ayrica

clf = LogisticRegression(max_iter=1000, multi_class="multinomial", class_weight="balanced")
clf.fit(X_train_tfidf, y_train)




