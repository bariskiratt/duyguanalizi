import matplotlib.pyplot as plt
import random
import numpy as np


def compute_f1(y_true, y_pred, threshold):
    y_pred = [1 if p >= threshold else 0 for p in y_pred]

    # Confusion Matrix
    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    # Precision, Recall, F1
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1, TP, TN, FP, FN


def best_threshold(y_true, y_probs,accuracy = 0.0001):
    best = {"threshold" : None, "f1" : -1}
    for t in [i * accuracy for i in range(int(1/accuracy) + 1)]:
        precision, recall, f1, TP, FP, FN, TN = compute_f1(y_true, y_probs, t)
        if f1 > best["f1"]:
            best = {"threshold": t, 
                    "f1": f1, 
                    "TP": TP, 
                    "FP": FP, 
                    "FN": FN, 
                    "TN": TN, 
                    "precision": precision, 
                    "recall": recall}
    return best

y_true = [
    1,0,1,1,0,1,0,1,0,1, 0,1,0,1,1,0,1,0,1,0,
    1,1,0,0,1,0,1,0,1,0, 1,0,1,1,0,1,0,1,0,1,
    0,1,0,1,1,0,1,0,1,0, 1,1,0,0,1,0,1,0,1,0,
    1,0,1,1,0,1,0,1,0,1, 0,1,0,1,1,0,1,0,1,0,
    1,1,0,0,1,0,1,0,1,0, 1,0,1,1,0,1,0,1,0,1,
    0,1,0,1,1,0,1,0,1,0, 1,1,0,0,1,0,1,0,1,0,
    1,0,1,1,0,1,0,1,0,1, 0,1,0,1,1,0,1,0,1,0
]

y_probs = [
    0.91,0.12,0.85,0.77,0.23,0.68,0.19,0.95,0.31,0.88,
    0.27,0.99,0.14,0.81,0.73,0.22,0.65,0.18,0.93,0.36,
    0.87,0.92,0.11,0.29,0.79,0.21,0.69,0.17,0.97,0.33,
    0.83,0.16,0.89,0.74,0.25,0.62,0.13,0.91,0.34,0.85,
    0.28,0.98,0.15,0.82,0.71,0.24,0.66,0.20,0.94,0.37,
    0.86,0.93,0.10,0.30,0.78,0.22,0.67,0.19,0.96,0.32,
    0.84,0.17,0.90,0.75,0.26,0.63,0.12,0.92,0.35,0.86,
    0.29,0.97,0.13,0.83,0.72,0.23,0.64,0.18,0.95,0.38,
    0.88,0.94,0.09,0.31,0.77,0.21,0.68,0.16,0.98,0.34,
    0.82,0.15,0.91,0.76,0.27,0.61,0.11,0.93,0.36,0.87,
    0.30,0.99,0.14,0.84,0.73,0.25,0.65,0.19,0.97,0.33,
    0.85,0.18,0.89,0.74,0.28,0.62,0.10,0.94,0.35,0.88,
    0.32,0.96,0.12,0.81,0.71,0.24,0.67,0.17,0.95,0.39,
    0.86,0.92,0.08,0.29,0.79,0.20,0.69,0.15,0.99,0.31,
    0.83,0.16,0.90,0.75,0.26,0.63,0.13,0.91,0.34,0.85
]

threshold = 0.42
print("Kullanılan threshold:", threshold)
precision, recall, f1, TP, TN, FP, FN = compute_f1(y_true, y_probs, threshold)
print("precision:", precision, "recall:", recall, "f1:", f1,)

best = best_threshold(y_true, y_probs, accuracy = 0.0001)
print("Best Threshold:", best["threshold"])
print("Metrics at best:", best["precision"], best["recall"], best["f1"])


thresholds = []
f1s = []
precisions = []
recalls = []

accuracy = 0.0001  
for t in [i * accuracy for i in range(int(1/accuracy) + 1)]:
    precision, recall, f1, TP, TN, FP, FN = compute_f1(y_true, y_probs, t)
    thresholds.append(t)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

plt.figure(figsize=(10,6))
plt.plot(thresholds, f1s, label="F1 Score")
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold vs F1 / Precision / Recall")
plt.legend()
plt.grid(True)
plt.xticks([i/20 for i in range(21)])
plt.yticks([i/20 for i in range(21)])  # 0.0, 0.05, ..., 1.0
plt.show()