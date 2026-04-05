import math
import torch
from torch import nn


class FourierCPPN(nn.Module):
    """
    TorchScript-friendly PyTorch implementation of the Fourier Compositional Pattern Producing Network (Fourier-CPPN)
    mirroring the architecture from the provided C++ version.

    Architecture (matches cppn.cpp):
      - If gauss_scale > 0: apply Fourier feature mapping with fixed random matrix B_T and two_pi scaling,
        then Linear(mapping_size*2 -> c_max) + PReLU.
      - Else: Linear(in_dim -> c_max) + PReLU.
      - Then three hidden blocks: Linear(c_max -> c_max) + PReLU (repeated 3 times).
      - Output: Linear(c_max -> out_dim).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        c_max: int,
        gauss_scale: float,
        mapping_size: int,
    ) -> None:
        super().__init__()

        # Store configuration (annotated for TorchScript)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.c_max = int(c_max)
        self.mapping_size = int(mapping_size)
        self.gauss = float(gauss_scale)
        self.use_fourier = bool(self.gauss > 0.0)

        # Layers (both input layers are constructed to keep TorchScript simple)
        self.fc_in_linear = nn.Linear(self.in_dim, self.c_max)
        self.fc_in_gau = nn.Linear(self.mapping_size * 2, self.c_max)
        self.fc2 = nn.Linear(self.c_max, self.c_max)
        self.fc3 = nn.Linear(self.c_max, self.c_max)
        self.fc4 = nn.Linear(self.c_max, self.c_max)
        self.fc_out = nn.Linear(self.c_max, self.out_dim)

        # Activations
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

        # Buffers for Fourier features path
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi))
        # B_T matches C++: normal(0, gauss_scale, {mapping_size, in_dim}).transpose(0, 1)
        # If gauss == 0, this becomes zeros which is fine (unused if use_fourier is False)
        B = torch.normal(mean=0.0, std=self.gauss, size=(self.mapping_size, self.in_dim))
        self.register_buffer("B_T", B.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            # x: [N, in_dim], B_T: [in_dim, mapping_size]
            x = torch.matmul(self.two_pi * x, self.B_T)
            x = torch.cat((x.sin(), x.cos()), dim=1)
            x = self.prelu1(self.fc_in_gau(x))
        else:
            x = self.prelu1(self.fc_in_linear(x))

        x = self.prelu2(self.fc2(x))
        x = self.prelu3(self.fc3(x))
        x = self.prelu4(self.fc4(x))
        x = self.fc_out(x)
        return x


__all__ = ["FourierCPPN"]
