import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1) Sentetik veri üretimi
np.random.seed(0)
num_samples = 1000
X0 = np.random.randn(num_samples // 2, 2) + np.array([-2, -2])
X1 = np.random.randn(num_samples // 2, 2) + np.array([2, 2])
X = np.vstack((X0, X1))
y = np.hstack((np.zeros(num_samples // 2), np.ones(num_samples // 2)))

# 2) Karıştır ve eğitim/test setine böl (80/20)
perm = np.random.permutation(num_samples)
X, y = X[perm], y[perm]
split_idx = int(0.8 * num_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 3) Bias terimi ekleyerek özellik matrisine ekle
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias  = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# 4) Logistic Regression (sıfırdan)
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, num_iter=10000):
        self.lr = lr
        self.num_iter = num_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(self.num_iter):
            preds = self.sigmoid(X @ self.theta)
            grad = (1 / m) * (X.T @ (preds - y))
            self.theta -= self.lr * grad
        return self

    def predict(self, X, threshold=0.5):
        return (self.sigmoid(X @ self.theta) >= threshold).astype(int)

# 5) Modeli eğit
model = LogisticRegressionScratch(lr=0.1, num_iter=10000)
model.fit(X_train_bias, y_train)

# 6) Tahmin ve metrik hesaplama
y_pred = model.predict(X_test_bias)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 7) Sonuçları yazdır
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
