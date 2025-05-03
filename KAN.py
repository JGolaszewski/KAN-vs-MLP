import numpy as np

class KANLayer:
    def __init__(self, input_size, output_size, num_grid=5):
        self.input_size = input_size
        self.output_size = output_size
        self.num_grid = num_grid
        self.grid = np.linspace(-2, 2, num_grid)
        self.coeff = np.random.randn(input_size, output_size, num_grid) * 0.1
        self.bin_indices = None
        self.alpha = None
        self.df_dx = None

    def forward(self, X):
        X_clipped = np.clip(X, self.grid[0], self.grid[-1])
        bin_indices = np.digitize(X_clipped, self.grid) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_grid - 2)
        
        left = self.grid[bin_indices]
        right = self.grid[bin_indices + 1]
        alpha = (X_clipped - left) / (right - left + 1e-8)
        
        # Advanced indexing preparation
        i = np.arange(self.input_size)[None, :, None]    # (1, input_size, 1)
        j = np.arange(self.output_size)[None, None, :]   # (1, 1, output_size)
        k = bin_indices[:, :, None]                      # (batch, input_size, 1)
        
        # Broadcast indices to (batch, input_size, output_size)
        coeff_left = self.coeff[i, j, k]
        coeff_right = self.coeff[i, j, k + 1]
        
        # Expand alpha for broadcasting
        alpha_exp = alpha[:, :, None]  # (batch, input_size, 1)
        
        # Linear interpolation
        interpolated = (1 - alpha_exp) * coeff_left + alpha_exp * coeff_right
        
        # Save for backward pass
        self.bin_indices = bin_indices
        self.alpha = alpha
        self.df_dx = (coeff_right - coeff_left) / (right - left + 1e-8)[:, :, None]
        
        return np.sum(interpolated, axis=1)

    def backward(self, dL_doutput, lr):
        batch_size = dL_doutput.shape[0]
        dcoeff = np.zeros_like(self.coeff)
        
        # Gradient accumulation
        for b in range(batch_size):
            for i in range(self.input_size):
                bin_idx = self.bin_indices[b, i]
                alpha = self.alpha[b, i]
                
                for j in range(self.output_size):
                    grad = dL_doutput[b, j]
                    
                    # Update coefficients
                    dcoeff[i, j, bin_idx] += (1 - alpha) * grad
                    dcoeff[i, j, bin_idx + 1] += alpha * grad
        
        # Update coefficients with learning rate
        self.coeff -= lr * (dcoeff / batch_size)
        
        # Gradient propagation to input
        return np.einsum('bj,bij->bi', dL_doutput, self.df_dx)

class KAN:
    def __init__(self, layers_config):
        self.layers = []
        for config in layers_config:
            input_size, output_size = config
            self.layers.append(KANLayer(input_size, output_size))
        self.forward_result = None

    def forward(self, X):
        self.forward_result = X
        for layer in self.layers:
            self.forward_result = layer.forward(self.forward_result)
        return self.forward_result

    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def backward(self, y, lr):
        y_pred = self.forward_result
        batch_size = y_pred.shape[0]
        dL_doutput = 2 * (y_pred - y) / batch_size
        
        for layer in reversed(self.layers):
            dL_doutput = layer.backward(dL_doutput, lr)