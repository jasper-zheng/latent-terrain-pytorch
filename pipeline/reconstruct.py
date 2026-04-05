from typing import List

import torch

from scripts.factory import Codec


@torch.no_grad()
def reconstruct_waveform_from_latents(codec: Codec, latents: torch.Tensor, use_stereo: bool = False) -> torch.Tensor:
    if latents.dim() != 2:
        raise ValueError("Expected latents shape [D, T_latent] (channel-first for codec)")
    wav = codec.decode(latents.unsqueeze(0))[0]  # [C, T]
    # Normalize to [-1, 1]
    max_abs = float(wav.abs().max().item())
    if max_abs > 0:
        wav = wav / max_abs
    # Channels policy
    if use_stereo and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav.unsqueeze(0)  # [B=1, C, T]


@torch.no_grad()
def reconstruct_all_segments(codec: Codec, models: List, trajectories: List[torch.Tensor],
                             segment_samples: int, hop: int, device: str, use_stereo: bool) -> List[torch.Tensor]:
    wavs: List[torch.Tensor] = []
    for model, traj in zip(models, trajectories):
        # traj is expected [B, D, T]
        T_latent = int(traj.shape[2])
        coords = torch.linspace(0.0, 1.0, T_latent, device=device).unsqueeze(1)
        with torch.no_grad():
            lat_pred = model.to(device).eval()(coords).t()  # [D, T_latent]
        wav = reconstruct_waveform_from_latents(codec, lat_pred, use_stereo=use_stereo)
        wavs.append(wav)
    return wavs
