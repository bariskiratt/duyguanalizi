import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)

df = pd.read_csv("salarydata.csv")
X_raw = df[["YearsExperience"]].values
y_raw = df["Salary"].values
cost_history = []
# (isteğe bağlı) standardize et
# X_raw = (X_raw - X_raw.mean()) / X_raw.std()

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

m, n = X_train.shape
X = np.c_[np.ones(m), X_train]      # bias sütunu
theta = np.zeros(n + 1)

def predict(theta, X):
    return X @ theta

def cost(theta, X, y):
    errors = predict(theta, X) - y
    return (errors @ errors) / (2 * len(y))

def GD(X, y, theta, epochs=1000, lr=0.01):
    m = len(y)
    for epoch in range(epochs):
        errors = predict(theta, X) - y
        grad = (X.T @ errors) / m      # 1/m EKLENDİ
        theta -= lr * grad
        c = cost(theta, X, y)
        cost_history.append(c)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  Cost = {cost(theta, X, y):.4f}")
    return theta
def SGD(X,y,theta,epochs=1000, lr=0.01,batch_size = 1):
    m = len(y)
    lr0 = lr
    hist = []
    for epoch in range(epochs):
        idx = rng.permutation(m)
        X_shuf, y_shuf = X[idx],y[idx]
        for start in range(0,m,batch_size):
            end = start + batch_size
            X_mb = X_shuf[start:end]
            y_mb =y_shuf[start:end]

            errors = predict(theta,X_mb) - y_mb
            grad = (X_mb.T @ errors) / len(y_mb)
            theta -= lr * grad
            # — 3. maliyet takibi —
            c = cost(theta, X, y)        # cost() yazdığın fonksiyon
            hist.append(c)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  Cost = {c:.4f}  lr={lr:.5f}")
    return theta, hist

theta = GD(X, y_train, theta, epochs=2000, lr=0.01)
print("Learned theta for GD:", theta)

# test set hatası
X_test_b = np.c_[np.ones(len(X_test)), X_test]
print("Test MSE:", cost(theta, X_test_b, y_test)*2)   # *2 çünkü fonksiyonda ½ vardı
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Epoch")
plt.ylabel("½·MSE")          # cost fonksiyonu ½·MSE idi
plt.title("Eğitim maliyetinin değişimi")
plt.grid()
plt.show()

theta = np.zeros(n + 1)          # sıfırla veya GD’den çıkan ağırlıkları kullan
theta, sgd_cost_history = SGD(X, y_train, theta,
                              epochs=2000,
                              lr=0.01,
                              batch_size=15)

print("Learned theta for SGD:", theta)

plt.plot(sgd_cost_history)
plt.xlabel("Epoch")
plt.ylabel("½·MSE")
plt.title("SGD – eğitim maliyeti")
plt.grid()
plt.show()