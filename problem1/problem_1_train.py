"""
Implement a training loop for the MLP and SIREN models.
"""

import torch
from torch.utils.data import DataLoader

from problem_1_gradients import gradient, laplace
from problem_1_mlp import MLP
from problem_1_siren import SIREN
from utils import ImageDataset, plot, psnr
from typing import Dict, Any


def train(
    model,  # "MLP" or "SIREN"
    dataset: ImageDataset,  # Dataset of coordinates and pixels for an image
    lr: float,  # Learning rate
    total_steps: int,  # Number of gradient descent step
    steps_til_summary: int,  # Number of steps between summaries (i.e. print/plot)
    device: torch.device,  # "cuda" or "cpu"
    **kwargs: Dict[str, Any],  # Model-specific arguments
):
    """
    Train the model on the provided dataset.
    
    Given the **kwargs, initialize a neural field model and an optimizer.
    Then, train the model and log the loss and PSNR for each step. Examples
    in the notebook use MSE loss, but feel free to experiment with other
    objective functions. Additionally, in the notebook, we plot the reconstruction
    and various gradients every `steps_til_summary` steps using `utils.plot()`.

    You re allowed to change the arguments as you see fit so long as you can plot
    images of the reconstruction and the gradients/laplacian every `steps_til_summary` steps.
    Look at `should_look_like` for examples of what we would like to see. Make sure to
    also plot (MSE) loss and PSNR every `steps_til_summary` steps.

    You should train for `total_steps` gradient steps on the whole image (look at `ImageDataset` in `utils.py`)
    and visualize the results every `steps_til_summary` steps. The visualization must at least include:
    1. The MSE and PSNR
    2. The reconstructed image
    (Optionally you can also include the laplace or gradient of the image).

    PSNR is defined here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    # raise NotImplementedError("Not implemented!")
    ## dataset and dataloader start
    dataloader = DataLoader(
        dataset, 
        batch_size=len(dataset), 
        shuffle=True
    )
    ## initalize
    if model == "MLP":
        # making sure its cuda
        model_net = MLP(**kwargs).to(device)
    elif model == "SIREN": 
        model_net = SIREN(**kwargs).to(device)
    ## optimzer and loss function mse
    optimizer = torch.optim.Adam(model_net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    ## now the training
    for step in range(1, total_steps + 1): 
        ## go over dataloader coords
        for coords, target in dataloader: 
            ## to deivice
            coords, target = coords.to(device), target.to(device)
            ## enable th gradients in input
            coords.requires_grad = True
            ## forward passing
            output = model_net(coords)[0]
            ## find th loss now with th mse
            mse_loss = loss_func(output,target)
            ## now do th backprop
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
        ## finding th psnr 
        psnr_value = psnr(output,target)
        ## logging at each stp of the way
        if step % steps_til_summary == 0 or step == 1:
            ## plot rconstruction and gradients 
            plot(coords, output, gradient(output, coords), laplace(output, coords))
            print(mse_loss.item())
            print(psnr_value)
    return model_net