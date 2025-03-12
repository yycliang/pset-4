"""
Implement a training loop for the MLP and SIREN models.
"""

import torch
from torch.utils.data import DataLoader

from problem_1_gradients import gradient, laplace
from problem_1_mlp import MLP
from problem_1_siren import SIREN
from utils import ImageDataset, plot, psnr


def train(
    model,  # "MLP" or "SIREN"
    dataset: ImageDataset,  # Dataset of coordinates and pixels for an image
    lr: float,  # Learning rate
    total_steps: int,  # Number of gradient descent step
    steps_til_summary: int,  # Number of steps between summaries
    device: torch.device,  # "cuda" or "cpu"
    **kwargs,  # Model-specific arguments
):
    """
    Train the model on the provided dataset.
    
    Given the **kwargs, initialize a neural field model and an optimizer.
    Then, train the model and log the loss and PSNR for each step. Examples
    in the notebook use MSE loss, but feel free to experiment with other
    objective functions. Additionally, in the notebook, we plot the reconstruction
    and various gradients every `steps_til_summary` steps using `utils.plot()`.
    """
    raise NotImplementedError("Not implemented!")
