"""
Implement a neural field using a SIREN.

The SIREN maps from `in_features`-dimensional points (e.g., 2D xy positions) 
to `out_features`-dimensional points (e.g., 1D color values). Pay special
attention to the paper linked in the README when implementing this model.
"""
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        is_first_layer: bool,
        omega_0: float,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(is_first_layer, in_features)

    @torch.no_grad()
    def init_weights(self, is_first_layer, in_features):
        """
        Initialize the weights of the layer according to the scheme
        described in the SIREN paper.
        """
        raise NotImplementedError("Not implemented!")

    def forward(self, x):
        raise NotImplementedError("Not implemented!")


class SIREN(nn.Module):
    def __init__(
        self,
        in_features: int,  # Number of input features
        out_features: int,  # Number of output features
        hidden_features: int,  # Number of features in the hidden layers
        hidden_layers: int,  # Number of hidden layers
        bias: bool = True,  # Whether to include a bias term in the linear layers
        last_layer_linear: bool = False,  # Whether to use a linear layer for the last layer
        first_omega_0: float = 20.0,  # omega_0 for the first layer
        hidden_omega_0: float = 20.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.bias = bias
        self.last_layer_linear = last_layer_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.net = self.initialize_net()

    def initialize_net(self):
        """
        Build the network according to the provided hyperparameters.
        """
        raise NotImplementedError("Not implemented!")

    def forward(self, x):
        """
        Implement a forward pass where the output AND the input require
        gradients so as to be differentiable.
        """
        raise NotImplementedError("Not implemented!")
