import numpy as np
from typing import List
from scipy.interpolate import BSpline

class KANLayer:
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 3):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        interior_knots = np.linspace(-1, 1, grid_size + 1)
        self.knots = np.concatenate([
            np.repeat(interior_knots[0], spline_order),
            interior_knots,
            np.repeat(interior_knots[-1], spline_order)
        ])
        
        self.num_bases = len(self.knots) - spline_order - 1

        self.weights = np.random.randn(out_dim, in_dim, self.num_bases) * 0.1
        
    def basis_functions(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        bases = np.zeros((batch_size, self.in_dim, self.num_bases))
        
        for i in range(self.in_dim):
            for j in range(self.num_bases):
                # f. basis
                c = np.zeros(self.num_bases)
                c[j] = 1.0
                
                spline = BSpline(self.knots, c, self.spline_order)
                
                x_clipped = np.clip(x[:, i], self.knots[0], self.knots[-1])
                bases[:, i, j] = spline(x_clipped)
        
        return bases
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        bases = self.basis_functions(x)  # (batch, in_dim, num_bases)
        
        # apply new weight
        output = np.zeros((batch_size, self.out_dim))
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                output[:, i] += np.sum(bases[:, j, :] * self.weights[i, j, :], axis=1)
        
        return output

class NewKAN:
    def __init__(self, layer_sizes: List[int], grid_size: int = 5, spline_order: int = 3):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                KANLayer(layer_sizes[i], layer_sizes[i+1], grid_size, spline_order)
            )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)