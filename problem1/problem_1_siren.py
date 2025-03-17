"""
Implement a neural field using a SIREN.

The SIREN maps from `in_features`-dimensional points (e.g., 2D xy positions)
to `out_features`-dimensional points (e.g., 1D color values). Pay special
attention to the paper linked in the README when implementing this model.

Some conventions we use that you will need to follow IF you want to use the unit tests:
1. If the last layer is linear, then it always has a bias (other terms have a bias IFF self.bias is True)
2. There is one layer from `in_features` to `hidden_features` and then `hidden_layers` layers from 
    `hidden_features` to `hidden_features`; the last layer (which could be linear or not) is from
    `hidden_features` to `out_features`.
"""

import torch
import torch.nn as nn
import jaxtyping
from typing import Tuple
import numpy as np

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
    def init_weights(self, is_first_layer: bool, in_features: int):
        """
        Initialize the weights of the layer according to the scheme
        described in the SIREN paper.
        """
        # raise NotImplementedError("Not implemented!")
        ## creating th bound 
        if is_first_layer:
            bound = (1/ in_features) 
        else: 
            ## initalize th othr hidden layers
            bound = (np.sqrt(6/in_features)) /self.omega_0
        ## set the bound 
        self.linear.weight.uniform_(-bound, bound)
        ## getting tolrance issue errors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # raise NotImplementedError("Not implemented!")
        ## the forward pass wth omega0 and linear 
        return torch.sin(self.omega_0 * self.linear(x))


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
        # raise NotImplementedError("Not implemented!")
        final_layers = []
        ## diff freq factor
        final_layers.append(
            SineLayer(
                self.in_features, 
                self.hidden_features, 
                bias=self.bias, 
                is_first_layer=True, 
                omega_0=self.first_omega_0
            )
        )
        ## the hidden layrs
        for i in range(self.hidden_layers):
            final_layers.append(
                SineLayer(
                    self.hidden_features, 
                    self.hidden_features, 
                    bias=self.bias, 
                    is_first_layer=False, 
                    omega_0=self.hidden_omega_0
                )
            )
        #3 for th last layer
        if self.last_layer_linear: 
            final_layers.append(nn.Linear(
                self.hidden_features, 
                self.out_features, 
                bias=True
            ))
        else: 
            final_layers.append(
                SineLayer(
                    self.hidden_features,
                    self.out_features, 
                    bias=True, 
                    is_first_layer=False, 
                    omega_0=self.hidden_omega_0,
                )
            )
        return nn.Sequential(*final_layers)

    def forward(self, coords: jaxtyping.Float[torch.Tensor, "N D"]) -> Tuple[
        jaxtyping.Float[torch.Tensor, "N out_features"],
        jaxtyping.Float[torch.Tensor, "N D"],
    ]:
        """
        Implement a forward pass where the output AND the input require gradients so as to be differentiable.

        Return: tuple of (outputs, gradient-enabled coords). Shape of outputs should be (N, out_features).

        Hint: coords should be (N, d) where N is the number of points (batch size) and d is the dimensionality
        if your input/field. Copy them and enable gradients on the copy. Then, pass them into your network
        recalling that that in `utils.py` we use the convention that the input values are in [-1, 1] where
        -1 means "furthest left" or "furthest bottom" (depending on the dimension) and 1 means "furthest right"
        or "furthest top".
        """
        # raise NotImplementedError("Not implemented!")
        #the input coords have the grads
        coordinates = coords.clone().detach().requires_grad_(True)
        #  adjust input shape to match infeatures
        if coordinates.shape[1] < self.in_features:
            # Expand th edimess if input has fewer features than expected
            coordinates = coordinates.expand(-1,self.in_features)
        elif coordinates.shape[1] > self.in_features:
            # Reduce dimes if input has more features than expected
            projection_layer = torch.nn.Linear(
                coordinates.shape[1], 
                self.in_features, 
                bias=False
            ).to(coordinates.device)
            coordinates = projection_layer(coordinates)
        # Forwardpass through the model
        outputs = self.net(coordinates)
        return outputs, coordinates
    
    
