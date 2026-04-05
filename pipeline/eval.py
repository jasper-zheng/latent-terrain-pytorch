from pathlib import Path
from typing import List, Iterable, Tuple

import torch
import torchaudio

from .datasets import build_noisy_test_dataloader
from .reconstruct import reconstruct_waveform_from_latents

from fourier_cppn import FourierCPPN
from scripts.factory import Codec

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

@torch.no_grad()
def compute_psnr_from_mse(mse: float, data_range: float) -> float:
    import math as _math
    eps = 1e-12
    if data_range <= 0:
        data_range = 1.0
    return 10.0 * _math.log10((data_range * data_range) / (mse + eps))


@torch.no_grad()
def evaluate_model_psnr(model, trajectory: torch.Tensor, device: str, batch_size: int) -> Tuple[float, float]:
    loader = build_noisy_test_dataloader(trajectory, batch_size=batch_size, noise_std=0.001, max_clip=0.005)
    model = model.to(device).eval()

    sum_sq = 0.0
    count = 0
    with torch.no_grad():
        for coords, targets in loader:
            coords = coords.to(device)
            targets = targets.to(device)
            pred = model(coords)
            diff = pred - targets
            sum_sq += float((diff * diff).sum().item())
            count += int(targets.numel())

    mse = sum_sq / max(1, count)
    tmin = float(trajectory.min().item())
    tmax = float(trajectory.max().item())
    data_range = tmax - tmin
    if data_range <= 0:
        data_range = float(trajectory.abs().max().item())
    if data_range <= 0:
        data_range = 1.0
    psnr = compute_psnr_from_mse(mse, data_range)
    return psnr, mse



@torch.no_grad()
def evaluate_model(model: FourierCPPN, trajectory: torch.Tensor, codec: Codec, batch_size: int = 128, reconstruct_audio: bool = False, device: str = "cpu") -> Tuple[float, float, torch.Tensor | None, torch.Tensor | None]:
    """Evaluate a trained model against the same latent vectors using noisy time coordinates.
    Returns (psnr_db, mse).
    """
    test_loader = build_noisy_test_dataloader(trajectory, batch_size=batch_size, noise_std=0.001, max_clip=0.005)
    model = model.to(device).eval()

    sum_sq = 0.0
    count = 0
    with torch.no_grad():
        recon_wavs = []
        ref_wavs = []
        for coords, targets in test_loader:
            coords = coords.to(device)
            targets = targets.to(device)
            pred = model(coords)
            diff = pred - targets
            sum_sq += float((diff * diff).sum().item())
            count += int(targets.numel())
            # print(f"pred: {pred.shape}, targets: {targets.shape}") => [108, 64]
            if reconstruct_audio:
                # Optional: reconstruct audio from predicted latent vectors and compare (not implemented here)
                recon_wav = reconstruct_waveform_from_latents(codec, pred.T)
                recon_wavs.append(recon_wav)
                ref_wav = reconstruct_waveform_from_latents(codec, targets.T)
                ref_wavs.append(ref_wav)
        if reconstruct_audio:
            recon = torch.cat(recon_wavs, dim=-1)  # concatenate along time axis
            ref = torch.cat(ref_wavs, dim=-1)
        else:
            recon = None
            ref = None

    mse = sum_sq / max(1, count)
    # Use per-trajectory peak-to-peak as data range (fallback to max|x|, then 1.0)
    tmin = float(trajectory.min().item())
    tmax = float(trajectory.max().item())
    data_range = tmax - tmin
    if data_range <= 0:
        data_range = float(trajectory.abs().max().item())
    if data_range <= 0:
        data_range = 1.0
    psnr = compute_psnr_from_mse(mse, data_range)
    
    return psnr, mse, recon, ref