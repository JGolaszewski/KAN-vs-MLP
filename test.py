from impl.mlp import ActivationLayer, FullyConnectedLayer, Network
from impl.utils import (
    relu, relu_prime,
    bin_cross_entropy, bin_cross_entropy_prime,
    sigmoid, sigmoid_prime
)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

def simulation(n, split):
    X = pd.DataFrame(
        {
            'Grade': np.random.normal(3.5, 1, n).round(2).clip(2, 5),
            'StudyingTime': np.random.normal(20, 10, n).round(2).clip(0, None)
        }
    )
    
    y = X.apply(
        lambda x: np.random.rand() < (
            0.1 
            + 0.4 * (x['Grade'] - 2)/3
            + 0.4 * 1/(1+np.exp((x['StudyingTime'] - 20) / -20))
        ),
        axis=1
    )
    
    return train_test_split(X, y, test_size=split, stratify=y)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = simulation(100000, 0.25)
    
    cols = ['Grade', 'StudyingTime']

    for col_name in cols:
        mean = X_train[[col_name]].mean()
        std = X_train[[col_name]].std()
        X_train[[col_name]] = (X_train[[col_name]] - mean)/std
        X_test[[col_name]] = (X_test[[col_name]] - mean)/std

    mlp_network = Network(bin_cross_entropy, bin_cross_entropy_prime)
    
    mlp_network.add(FullyConnectedLayer(2, 32))
    mlp_network.add(ActivationLayer(relu, relu_prime))
    mlp_network.add(FullyConnectedLayer(32,16))
    mlp_network.add(ActivationLayer(relu, relu_prime))
    mlp_network.add(FullyConnectedLayer(16,8))
    mlp_network.add(ActivationLayer(relu, relu_prime))
    mlp_network.add(FullyConnectedLayer(8,4))
    mlp_network.add(ActivationLayer(relu, relu_prime))
    mlp_network.add(FullyConnectedLayer(4,1))
    mlp_network.add(ActivationLayer(sigmoid, sigmoid_prime))
    
    mlp_network.fit(
        X_train.to_numpy(), y_train.to_numpy(),
        epoches=300, learning_rate=0.01, batch_size=64
    )
    
    pred = mlp_network.predict(X_test.to_numpy())
    pred = list(map(lambda x: x[0,0] > 0.5, pred))
    
    print(f"Accuracy: {accuracy_score(y_test, pred)}")
    print(f"F1: {f1_score(y_test, pred)}")
    print(pd.DataFrame(confusion_matrix(y_test, pred)))
    
    pass