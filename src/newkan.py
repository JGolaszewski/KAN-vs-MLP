import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=5):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        
        self.spline_coeffs = nn.Parameter(torch.randn(output_dim, input_dim, grid_size + 3) * 0.1)
        self.base_weight = nn.Parameter(torch.ones(output_dim, input_dim))
        self.spline_weight = nn.Parameter(torch.ones(output_dim, input_dim))
        
        self.grid = nn.Parameter(torch.linspace(-2, 2, grid_size + 1), requires_grad=False)
        
        self.silu = nn.SiLU()
        
    def bspline_basis(self, x, i, k, grid):
        if k == 0:
            return ((grid[i] <= x) & (x < grid[i+1])).float()
        else:
            left = torch.zeros_like(x)
            right = torch.zeros_like(x)
            
            if grid[i+k] != grid[i]:
                left = (x - grid[i]) / (grid[i+k] - grid[i]) * self.bspline_basis(x, i, k-1, grid)
            if grid[i+k+1] != grid[i+1]:
                right = (grid[i+k+1] - x) / (grid[i+k+1] - grid[i+1]) * self.bspline_basis(x, i+1, k-1, grid)
            
            return left + right
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        extended_grid = torch.cat([
            self.grid[0] - (self.grid[1] - self.grid[0]) * torch.arange(3, 0, -1),
            self.grid,
            self.grid[-1] + (self.grid[1] - self.grid[0]) * torch.arange(1, 4)
        ])
        
        basis_matrix = torch.zeros(batch_size, self.input_dim, self.grid_size + 3, device=x.device)
        
        for i in range(self.grid_size + 3):
            for j in range(self.input_dim):
                basis_matrix[:, j, i] = self.bspline_basis(x[:, j], i, 3, extended_grid)
        
        spline_output = torch.einsum('bik,oik->boi', basis_matrix, self.spline_coeffs)
        
        base_output = self.silu(x.unsqueeze(1).expand(-1, self.output_dim, -1))
        
        weighted_output = (self.base_weight.unsqueeze(0) * base_output + 
                          self.spline_weight.unsqueeze(0) * spline_output)
        
        return weighted_output.sum(dim=2)

class NewKAN(nn.Module):
    def __init__(self, layers, grid_size=5):
        super(NewKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(KANLayer(layers[i], layers[i+1], grid_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())