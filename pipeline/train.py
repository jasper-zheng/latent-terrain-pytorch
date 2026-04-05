from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from fourier_cppn import FourierCPPN


def train_fourier_cppn_for_trajectory(
    trajectory: torch.Tensor,
    latent_dim: int,
    c_max: int,
    gauss_scale: float,
    mapping_size: int,
    device: str,
    epochs: int,
    lr: float,
    loader: DataLoader,
    in_dim: int = 1,
    print_every: int = 50,
) -> FourierCPPN:
    model = FourierCPPN(
        in_dim=in_dim,
        out_dim=latent_dim,
        c_max=c_max,
        gauss_scale=gauss_scale,
        mapping_size=mapping_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count = 0
        for coords, targets in loader:
            coords = coords.to(device)
            targets = targets.to(device)
            pred = model(coords)
            loss = loss_fn(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)
        if print_every and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch}/{epochs} | loss={total_loss / max(1,count):.6f}")
    return model
