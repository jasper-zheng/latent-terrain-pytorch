from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory: torch.Tensor, t_min: float = 0.0, t_max: float = 1.0):
        """
        Accepts trajectory shaped either [B, D, T] or [T, D]; internally stores [T, D].
        """
        if trajectory.dim() == 3:
            traj_td = trajectory.squeeze(0).transpose(0, 1)  # [T, D]
        elif trajectory.dim() == 2:
            traj_td = trajectory
        else:
            raise ValueError("trajectory must be [B,D,T] or [T,D]")
        self.traj = traj_td
        self.T = traj_td.shape[0]
        self.coords = torch.linspace(t_min, t_max, self.T).unsqueeze(1).to(trajectory.device)

    def __len__(self):
        return self.T

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.traj[idx]


class NoisyTrajectoryTestDataset(Dataset):
    def __init__(self, trajectory: torch.Tensor, noise_std: float, max_clip: float, t_min: float = 0.0, t_max: float = 1.0):
        if trajectory.dim() == 3:
            traj_td = trajectory.squeeze(0).transpose(0, 1)  # [T, D]
        elif trajectory.dim() == 2:
            traj_td = trajectory
        else:
            raise ValueError("trajectory must be [B,D,T] or [T,D]")
        self.traj = traj_td
        self.T = traj_td.shape[0]
        base = torch.linspace(t_min, t_max, self.T).unsqueeze(1)
        noise = torch.randn_like(base) * noise_std
        noise = noise.clamp(min=-max_clip, max=max_clip)
        perturbed = (base + noise).clamp_(t_min, t_max)
        self.coords = perturbed

    def __len__(self):
        return self.T

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.traj[idx]


def build_dataloader_for_trajectory(trajectory: torch.Tensor, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    ds = TrajectoryDataset(trajectory)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle, drop_last=False, num_workers=num_workers)


def build_noisy_test_dataloader(trajectory: torch.Tensor, batch_size: int, noise_std: float, max_clip: float) -> DataLoader:
    ds = NoisyTrajectoryTestDataset(trajectory, noise_std=noise_std, max_clip=max_clip)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False, drop_last=False)
