"""
F1 yardımcıları
• binary_compute_f1  :  ↔  eskinin compute_f1 (tek sınıf – threshold)
• macro_f1_multiclass:  çoklu sınıf makro F1
"""
from typing import List
from sklearn.metrics import f1_score, classification_report

# ------------------------------ binary ---------------------------------
def binary_compute_f1(y_true: List[int], y_probs: List[float], thr: float = 0.5):
    y_pred = [1 if p >= thr else 0 for p in y_probs]
    tp = sum(yt == yp == 1 for yt, yp in zip(y_true, y_pred))
    tn = sum(yt == yp == 0 for yt, yp in zip(y_true, y_pred))
    fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))
    fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, tn, fp, fn

def best_threshold(y_true, y_probs, step: float = 1e-3):
    best = {"thr": 0.0, "f1": -1}
    t = 0.0
    while t <= 1.0:
        _, _, f1, *_ = binary_compute_f1(y_true, y_probs, t)
        if f1 > best["f1"]:
            best = {"thr": t, "f1": f1}
        t += step
    return best

# --------------------------- multiclass --------------------------------
def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

# --------------------------- quick demo -------------------------------
if __name__ == "__main__":
    # demo – aynı y_true / y_probs listeleri kullanılabilir
    from random import random
    y_true_demo = [0, 1, 1, 0, 1]
    y_probs_demo = [random() for _ in y_true_demo]
    print("best thr demo:", best_threshold(y_true_demo, y_probs_demo))
    # multiclass demo
    y_true_m = ["negatif", "pozitif", "notr", "negatif"]
    y_pred_m = ["negatif", "pozitif", "negatif", "negatif"]
    print("macro-F1 multiclass:", macro_f1_multiclass(y_true_m, y_pred_m))
    print("\nreport\n", classification_report(y_true_m, y_pred_m, zero_division=0))
