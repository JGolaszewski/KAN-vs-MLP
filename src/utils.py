import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def mse(y_true, y_pred):
    return (0.5*(y_true - y_pred)**2).mean()

def mse_prime(y_true, y_pred):
    return y_pred - y_true

def bin_cross_entropy(y_true, y_pred):
    eps = 1e-9
    return -(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

def bin_cross_entropy_prime(y_true, y_pred):
    return -(y_true/y_pred - (1-y_true)/(1-y_pred))