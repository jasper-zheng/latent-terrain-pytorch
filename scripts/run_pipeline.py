#!/usr/bin/env python3
"""
End-to-end runner for the Fourier-CPPN latent trajectory pipeline.

Workflow:
- Load audio segments from a dataset directory (10s segments by default @ 44.1kHz)
- Build a codec (FlowDec, Stable Audio Open, or Music2Latent)
- Encode each segment to a latent trajectory [B, D, T_lat]
- Train a Fourier-CPPN per trajectory to map t∈[0,1] -> latent[D]
- Evaluate PSNR per segment
- Optionally reconstruct audio from predicted latents and save WAV files
- Optionally compute FAD between original vs reconstructed sets (VGGish-based)

"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import torch
import torchaudio

from pipeline import audio as audio_utils
from pipeline import datasets as ds_utils
from pipeline import train as train_utils
from pipeline import reconstruct as recon_utils
from pipeline import eval as eval_utils
from scripts.factory import FlowDecWrapper, StableAudioOpenWrapper, Music2LatentWrapper, Codec


# ---------------- Utils ---------------- #

def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_codec(name: str, device: str) -> Codec:
    n = name.lower()
    if n in {"flowdec"}:
        return FlowDecWrapper(device=device)
    if n in {"stable-audio-open", "stableaudio", "sao"}:
        return StableAudioOpenWrapper(device=device)
    if n in {"music2latent", "m2l",}:
        return Music2LatentWrapper(device=device)
    if name == 'rave-speech':
        vctk = torch.jit.load('models/VCTK.ts').to(device).eval()
        return vctk
    if name == 'rave-string':
        string = torch.jit.load('models/aam_string_16.ts').to(device).eval()
        return string
    if name == 'rave-drum':
        drum = torch.jit.load('models/aam_drum_16.ts').to(device).eval()
        return drum
    raise ValueError(f"Unknown codec '{name}'. Choose from: flowdec, stable-audio-open, music2latent, rave-speech, rave-string, rave-drum")

def segments_to_latent_trajectories(segments: List[torch.Tensor], codec: Codec, device: str) -> List[torch.Tensor]:
    trajectories: List[torch.Tensor] = []
    with torch.no_grad():
        for seg in segments:
            # print(f"Encoding segment of shape {seg.shape}")
            lat = codec.encode(seg.to(device))  # [T_latent, latent_dim]
            trajectories.append(lat)
    print(f"Built {len(trajectories)} latent trajectories")
    return trajectories

@dataclass
class CPPNHparams:
    c_max: int = 128
    mapping_size: int = 64
    gauss_scale: float = 10.0
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 256


@dataclass
class RunConfig:
    dataset_dir: str
    dataset_name: str = ""
    codec_name: str = "music2latent"
    out_dir: str = "outputs"
    segment_seconds: float = 10.0
    target_sr: int = 44100
    stereo: bool = False
    save_models: bool = False
    save_audio: bool = False
    compute_fad: bool = False
    device: str | None = None
    hpo_trials: int = 30
    hpo_epochs: int = 200


# ---------------- Main Pipeline ---------------- #

def run(config: RunConfig, hparams: CPPNHparams) -> Dict[str, Any]:
    device = pick_device(config.device)

    # Resolve dataset dir (defaults)
    base = Path(__file__).resolve().parent.parent
    default_map = {
        "aam_test": base / "data" / "aam_test",
    }
    audio_dir = Path(config.dataset_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    segment_samples = int(round(config.segment_seconds * config.target_sr))

    # Build segments [B=1, C, T]
    segments: List[torch.Tensor] = audio_utils.build_segment_dataset(audio_dir, target_sr=config.target_sr, segment_samples=segment_samples, use_stereo=config.stereo)
    if len(segments) == 0:
        raise RuntimeError(f"No valid segments found in {audio_dir}")

    # Codec
    codec = build_codec(config.codec_name, device=device)

    # Encode segments to trajectories [B, D, T_lat]
    trajectories: List[torch.Tensor] = []
    for seg in segments:
        with torch.no_grad():
            z = codec.encode(seg.to(device))
        # ensure CPU and contiguous for dataloaders
        trajectories.append(z.detach().cpu().contiguous())

    # Train a model per trajectory
    models = []
    psnrs: List[float] = []
    mses: List[float] = []

    for i, traj in enumerate(trajectories):
        latent_dim = int(traj.shape[1]) if traj.dim() == 3 else int(traj.shape[-1])
        loader = ds_utils.build_dataloader_for_trajectory(traj, batch_size=hparams.batch_size, shuffle=False)
        model = train_utils.train_fourier_cppn_for_trajectory(
            trajectory=traj,
            latent_dim=latent_dim,
            c_max=hparams.c_max,
            gauss_scale=hparams.gauss_scale,
            mapping_size=hparams.mapping_size,
            device=device,
            epochs=hparams.epochs,
            lr=hparams.lr,
            loader=loader,
            print_every=max(1, hparams.epochs // 3),
        )
        models.append(model)
        # PSNR
        psnr, mse = eval_utils.evaluate_model_psnr(model, trajectory=traj, device=device, batch_size=hparams.batch_size)
        psnrs.append(float(psnr))
        mses.append(float(mse))
        print(f"[Segment {i:04d}] PSNR={psnr:.2f} dB | MSE={mse:.6e}")

    # Aggregate PSNR
    import numpy as _np
    psnr_mean = float(_np.mean(psnrs))
    psnr_p90 = float(_np.percentile(psnrs, 90))
    print(f"PSNR (mean)={psnr_mean:.2f} dB | (p90)={psnr_p90:.2f} dB over {len(psnrs)} segments")

    # Outputs
    out_root = Path(config.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Save models (optional, TorchScript)
    model_paths: List[str] = []
    if config.save_models:
        model_dir = out_root / "fourier_cppn_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(models):
            scripted = torch.jit.script(model.cpu().eval())
            mp = model_dir / f"fourier_cppn_traj_{i:04d}.pt"
            scripted.save(str(mp))
            model_paths.append(str(mp))
        print(f"Saved {len(model_paths)} scripted models → {model_dir}")

    # Reconstruct audio (optional)
    recon_paths: List[str] = []
    recon_wavs: List[torch.Tensor] = []
    if config.save_audio or config.compute_fad:
        recon_wavs = recon_utils.reconstruct_all_segments(
            codec=codec,
            models=models,
            trajectories=trajectories,
            segment_samples=segment_samples,
            hop=1,
            device=device,
            use_stereo=config.stereo,
        )
        if config.save_audio:
            audio_dir_out = out_root / "recon"
            audio_dir_out.mkdir(parents=True, exist_ok=True)
            for i, wav in enumerate(recon_wavs):
                wav_path = audio_dir_out / f"recon_{i:04d}.wav"
                torchaudio.save(str(wav_path), wav.squeeze(0).cpu(), sample_rate=config.target_sr)
                recon_paths.append(str(wav_path))
            print(f"Saved {len(recon_paths)} recon WAVs → {audio_dir_out}")

    summary = {
        "dataset": str(audio_dir),
        "codec": config.codec_name,
        "segments": len(segments),
        "psnr_per_segment": psnrs,
        "mse_per_segment": mses,
        "psnr_mean": psnr_mean,
        "psnr_p90": psnr_p90,
        "saved_models": model_paths,
        "saved_audio": recon_paths,
        "device": device,
        "hparams": asdict(hparams),
    }

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------- CLI ---------------- #

def main():
    p = argparse.ArgumentParser(description="Run Fourier-CPPN trajectory fitting pipeline")
    p.add_argument("--dataset-name", type=str, default="aam_test", help="Named dataset mapping (e.g., aam_test) or ignored if --dataset-dir set")
    p.add_argument("--dataset-dir", type=str, default=None, help="Path to a directory of audio files (overrides --dataset-name)")
    p.add_argument("--codec", type=str, default="flowdec", help="Codec to use: flowdec | stable-open | music2latent")
    p.add_argument("--out-dir", type=str, default="outputs", help="Output directory for models/audio/summary.json")
    p.add_argument("--segment-seconds", type=float, default=10.0, help="Segment length in seconds")
    p.add_argument("--sr", type=int, default=44100, help="Target sample rate for audio I/O")
    p.add_argument("--stereo", action="store_true", help="Use stereo segments; otherwise mix to mono")

    # Hparams
    p.add_argument("--c-max", type=int, default=128)
    p.add_argument("--mapping-size", type=int, default=64)
    p.add_argument("--gauss-scale", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)

    # Options
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--save-audio", action="store_true")
    p.add_argument("--compute-fad", action="store_true")
    p.add_argument("--device", type=str, default=None, help="Device override: mps|cuda|cpu (default: auto)")

    args = p.parse_args()

    cfg = RunConfig(
        dataset_dir=args.dataset_dir,
        codec_name=args.codec,
        out_dir=args.out_dir,
        segment_seconds=args.segment_seconds,
        target_sr=args.sr,
        stereo=bool(args.stereo),
        save_models=bool(args.save_models),
        save_audio=bool(args.save_audio),
        compute_fad=bool(args.compute_fad),
        device=args.device,
    )
    hp = CPPNHparams(
        c_max=int(args.c_max),
        mapping_size=int(args.mapping_size),
        gauss_scale=float(args.gauss_scale),
        lr=float(args.lr),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )

    run(cfg, hp)


if __name__ == "__main__":
    main()
