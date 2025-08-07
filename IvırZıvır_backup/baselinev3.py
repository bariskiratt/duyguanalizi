import os, json, pickle
from pathlib import Path
from typing import Dict, Any, cast

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ----------------------------------------------------------------------
def train_baseline(
    train_parquet: str = "data/processed/train_data.parquet",
    test_parquet:  str = "data/processed/test_data.parquet",
    output_dir:    str = "artifacts/baseline",
    ngram: tuple   = (1, 2),
    max_feat: int  = 10_000,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1) veri -------------------------------------------------
    df_train = pd.read_parquet(train_parquet)
    df_test  = pd.read_parquet(test_parquet)
    X_train, y_train = df_train["review_text"], df_train["label"]
    X_test,  y_test  = df_test["review_text"],  df_test["label"]

    # ---------- 2) TF-IDF ----------------------------------------------
    vect = TfidfVectorizer(ngram_range=ngram, max_features=max_feat)
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf  = vect.transform(X_test)

    # ---------- 3) LR + opsiyonel grid-search --------------------------
    base = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
    param_grid = {"C": [0.1, 1, 3]}
    clf = GridSearchCV(base, param_grid, cv=3, scoring="f1_macro",
                       n_jobs=-1, verbose=0).fit(X_train_tfidf, y_train).best_estimator_

    # ---------- 4) tahmin & metrikler ----------------------------------
    y_pred        = clf.predict(X_test_tfidf)
    y_pred_proba  = clf.predict_proba(X_test_tfidf)
    accuracy      = accuracy_score(y_test, y_pred)
    report: Dict[str, Any] = cast(
        Dict[str, Any], classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    )

    # yüksek güvenli tahmin sayısı
    high_conf = sum(probs.max() >= 0.80 for probs in y_pred_proba)

    # ---------- 5) çıktı dosyaları -------------------------------------
    Path(output_dir, "model.pkl").write_bytes(pickle.dumps(clf))
    Path(output_dir, "vectorizer.pkl").write_bytes(pickle.dumps(vect))

    with open(Path(output_dir, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump({"accuracy": accuracy,
                   "macro_f1": report["macro avg"]["f1-score"],
                   "class_report": report,
                   "high_conf_preds": int(high_conf)}, fp, indent=2, ensure_ascii=False)

    # confusion matrix görseli
    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels_sorted, yticklabels=labels_sorted)
    plt.title("Confusion Matrix – Baseline")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(Path(output_dir, "confusion_matrix.png"))
    plt.close()

    # konsola özet
    print(f"Accuracy   : {accuracy:.3f}")
    print(f"Macro-F1   : {report['macro avg']['f1-score']:.3f}")
    for lbl in clf.classes_:
        print(f"{lbl:<7}: {report[lbl]['f1-score']:.3f}")
    print(f"High-conf (>0.8) tahmin: {high_conf}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    train_baseline()
