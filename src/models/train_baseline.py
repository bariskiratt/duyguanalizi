import os, json, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from typing import Dict, Any, cast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def train_baseline(
    train_parquet: str = "data/processed/test_data.parquet",
    test_parquet:  str = "data/processed/train_data.parquet",
    output_dir:    str = "artifacts/baseline",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ---------- veri ----------------------------------------------------
    df_train = pd.read_parquet(train_parquet)
    df_test  = pd.read_parquet(test_parquet)

    X_train, y_train = df_train["review_text"], df_train["label"]
    X_test,  y_test  = df_test["review_text"],  df_test["label"]

    # ---------- TF-IDF ---------------------------------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10_000,
        stop_words=None,          # Türkçe stop-words eklemek isterseniz liste verin
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # ---------- Lojistik Regresyon + grid-search ------------------------
    param_grid = {
        "C": [0.1, 1, 5],
        "max_iter": [500, 1000],
        "class_weight": ["balanced", None],
    }
    base_clf = LogisticRegression(
        solver="lbfgs",
        random_state=42,
    )
    clf = GridSearchCV(
        base_clf,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    ).fit(X_train_tfidf, y_train).best_estimator_

    # ---------- değerlendirme -------------------------------------------
    y_pred = clf.predict(X_test_tfidf)
    report = cast(Dict[str, Any],
                  classification_report(y_test, y_pred, output_dict=True))

    with open(Path(output_dir, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    print(f"Macro-F1: {report['macro avg']['f1-score']:.3f}")
    for lbl in clf.classes_:
        print(f"{lbl:<7}: {report[lbl]['f1-score']:.3f}")

    # ---------- confusion matrix ----------------------------------------
    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels_sorted, yticklabels=labels_sorted)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix – Baseline TF-IDF + LR")
    plt.tight_layout()
    plt.savefig(Path(output_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    train_baseline()
