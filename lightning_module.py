from typing import Any, Optional

import torch
import pytorch_lightning as pl

from fourier_cppn import FourierCPPN


class LitFourierCPPN(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the FourierCPPN network.

    training_step and validation_step are intentionally left blank for now.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        c_max: int,
        gauss_scale: float,
        mapping_size: int,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = FourierCPPN(
            in_dim=in_dim,
            out_dim=out_dim,
            c_max=c_max,
            gauss_scale=gauss_scale,
            mapping_size=mapping_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:  # noqa: D401
        # Intentionally left blank as requested
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:  # noqa: D401
        # Intentionally left blank as requested
        pass

    def configure_optimizers(self):
        # Provide a default optimizer to keep Lightning happy even if steps are blank
        lr = getattr(self.hparams, "learning_rate", 1e-3)
        return torch.optim.Adam(self.parameters(), lr=lr)


__all__ = ["LitFourierCPPN"]
