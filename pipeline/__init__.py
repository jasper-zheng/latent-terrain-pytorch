"""
Reusable utilities for the Fourier-CPPN latent trajectory experiments.
Modules:
- audio: audio loading, segmentation, dataset resolution
- datasets: trajectory datasets and dataloaders
- reconstruct: latent->audio reconstruction helpers (codec-based)
- train: training loop for FourierCPPN
- eval: PSNR and FAD computation (VGGish-based)
- hpo: Optuna hyperparameter tuning for FourierCPPN
"""

from . import audio as audio
from . import datasets as datasets
from . import reconstruct as reconstruct
from . import train as train
from . import eval as eval
from . import hpo as hpo

__all__ = ["audio", "datasets", "reconstruct", "train", "eval", "hpo"]
