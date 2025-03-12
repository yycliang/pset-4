"""
Implement a neural field using a simple MLP.

The MLP maps from `in_features`-dimensional points (e.g., 2D xy positions) 
to `out_features`-dimensional points (e.g., 1D color values). To make your
implementation more general, also let the user specify any activation function.
"""

import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features
        out_features: int,  # Number of output features
        hidden_features: int,  # Number of features in the hidden layers
        hidden_layers: int,  # Number of hidden layers in the MLP
        bias: bool = True,  # Whether to include a bias term in the linear layers
        activation: str = "ReLU",  # E.g., "ReLU", "Tanh", "GELU", etc.
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

        self.net = self.initialize_net()

    def initialize_net(self):
        """Build the network according to the provided hyperparameters."""
        raise NotImplementedError("Not implemented!")
    
    def forward(self, x):
        """
        Implement a forward pass where the output AND the input require
        gradients so as to be differentiable.
        """
        raise NotImplementedError("Not implemented!")
