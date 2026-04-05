from pathlib import Path
from typing import List, Optional, Dict

import torch
import torchaudio


def resolve_audio_dir(name: str, mapping: Dict[str, Path], custom: Optional[Path] = None) -> Path:
    if custom is not None:
        return Path(custom)
    if name in mapping:
        return mapping[name]
    return Path(name)


def load_audio(path: Path, target_sr: int, use_stereo: bool = False) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if use_stereo:
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
    else:
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
    return wav


def segment_or_discard(wav: torch.Tensor, segment_samples: int) -> List[torch.Tensor]:
    T = wav.shape[1]
    segments: List[torch.Tensor] = []
    if T < segment_samples:
        return segments
    num_full = T // segment_samples
    for i in range(num_full):
        start = i * segment_samples
        end = start + segment_samples
        segments.append(wav[:, start:end].unsqueeze(0))  # [B=1, C, T]
    return segments


def build_segment_dataset(audio_dir: Path, target_sr: int, segment_samples: int, use_stereo: bool = False, max_segments: int = 725) -> List[torch.Tensor]:
    exts = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")
    files: List[str] = []
    for e in exts:
        files.extend([str(p) for p in Path(audio_dir).glob(e)])
    files = sorted(files)
    all_segments: List[torch.Tensor] = []
    for fp in files:
        try:
            wav = load_audio(Path(fp), target_sr=target_sr, use_stereo=use_stereo)
            segs = segment_or_discard(wav, segment_samples)
            all_segments.extend(segs)
            if len(all_segments) >= max_segments:
                break
        except Exception as e:
            print(f"Failed {fp}: {e}")
    return all_segments
