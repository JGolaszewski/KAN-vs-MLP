import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SimpleKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.activation_params = nn.Parameter(torch.randn(output_dim, input_dim, 10) * 0.1)
        self.base_weight = nn.Parameter(torch.ones(output_dim, input_dim))
        self.activation_weight = nn.Parameter(torch.ones(output_dim, input_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # base activation (tanh)
        base_act = torch.tanh(x.unsqueeze(1).expand(-1, self.output_dim, -1))
        
        # learnable activation using Gaussian basis
        x_expanded = x.unsqueeze(1).unsqueeze(-1)  # [batch, 1, input_dim, 1]
        params = self.activation_params.unsqueeze(0)  # [1, output_dim, input_dim, 10]
        positions = torch.linspace(-2, 2, 10, device=x.device).view(1, 1, 1, -1)
        gaussian_responses = torch.exp(-((x_expanded - positions) ** 2))
        weighted_responses = torch.sum(params * gaussian_responses, dim=-1)
        
        combined = (self.base_weight.unsqueeze(0) * base_act + 
                   self.activation_weight.unsqueeze(0) * weighted_responses)
        output = combined.sum(dim=2)
        return output

class SimpleKAN(nn.Module):
    def __init__(self, layers):
        super(SimpleKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(SimpleKANLayer(layers[i], layers[i+1]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())