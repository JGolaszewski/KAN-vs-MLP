import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def backward(self, X, y, lr):
        m = X.shape[0]
        dL_dz2 = 2 * (self.z2 - y) / m
        dW2 = self.a1.T @ dL_dz2
        db2 = np.sum(dL_dz2, axis=0)
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (self.z1 > 0)
        dW1 = X.T @ dL_dz1
        db1 = np.sum(dL_dz1, axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1