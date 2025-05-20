import numpy as np
import utils

class ActivationLayer:
    def forward(self, input_data):
        self.input = input_data
        return utils.sigmoid(input_data)

    def backward(self, output_error):
        return utils.sigmoid_prime(self.input) * output_error
    
    def step(self, eta):
        return
    
class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.delta_w = np.zeros((input_size, output_size))
        self.delta_b = np.zeros((1,output_size))
        self.passes = 0

        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.delta_w += weights_error
        self.delta_b += output_error
        self.passes += 1
        return input_error

    def step(self, eta):
        self.weights -= eta * self.delta_w / self.passes
        self.bias -= eta * self.delta_b / self.passes

        self.delta_w = np.zeros(self.weights.shape)
        self.delta_b = np.zeros(self.bias.shape)
        self.passes = 0

class Network:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        result = []
        for i in range(input_data.shape[0]):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epoches, learning_rate, batch_size=64):
        for i in range(epoches):
            err = 0

            idx = np.argsort(np.random.random(x_train.shape[0]))[:batch_size]
            x_batch = x_train[idx]
            y_batch = y_train[idx]

            for j in range(batch_size):
                output = x_batch[j]
                for layer in self.layers:
                    output = layer.forward(output)

                err += utils.mse(y_batch[j], output)

                error = utils.mse_prime(y_batch[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error)
            
            for layer in self.layers:
                layer.step(learning_rate)

            if (self.verbose) and ((i%10) == 0):
                err /= batch_size
                print('epoch: %5d/%d   error=%0.9f' % (i, epoches, err))