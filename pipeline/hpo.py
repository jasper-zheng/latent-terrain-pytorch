from typing import Dict, Any, Optional

import torch

from .datasets import build_dataloader_for_trajectory
from .eval import evaluate_model_psnr
from .train import train_fourier_cppn_for_trajectory


def _latent_dim_from_trajectory(trajectory: torch.Tensor) -> int:
    if trajectory.dim() == 3:  # [B, D, T]
        return int(trajectory.shape[1])
    if trajectory.dim() == 2:  # [T, D]
        return int(trajectory.shape[1])
    raise ValueError("trajectory must be [B,D,T] or [T,D]")


def tune_cppn_hparams_for_trajectory(
    trajectory: torch.Tensor,
    device: str = "cpu",
    max_trials: int = 15,
    epochs: int = 300,
    batch_size: int = 256,
    fixed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs Optuna to tune Fourier-CPPN hyperparameters on a single trajectory using PSNR.

    Returns a dict with best params and score: {"params": {...}, "psnr": float}
    """
    try:
        import optuna  # type: ignore
    except Exception as e:
        raise RuntimeError("Optuna not installed. Install with `pip install optuna`." ) from e

    latent_dim = _latent_dim_from_trajectory(trajectory)

    if fixed is None:
        fixed = {}

    def objective(trial: "optuna.trial.Trial") -> float:
        c_max = fixed.get("c_max") or trial.suggest_categorical("c_max", [64, 128, 256, 512])
        mapping_size = fixed.get("mapping_size") or trial.suggest_categorical("mapping_size", [32, 64, 128, 256, 512])
        gauss_scale = fixed.get("gauss_scale", 2.0)
        # print(gauss_scale)
        lr = fixed.get("lr") or trial.suggest_float("lr", 1e-4, 3e-3, log=True)

        loader = build_dataloader_for_trajectory(trajectory, batch_size=batch_size, shuffle=False)
        model = train_fourier_cppn_for_trajectory(
            trajectory=trajectory,
            latent_dim=latent_dim,
            c_max=int(c_max),
            gauss_scale=float(gauss_scale),
            mapping_size=int(mapping_size),
            device=device,
            epochs=int(epochs),
            lr=float(lr),
            loader=loader,
            print_every=0,
        )
        psnr, _ = evaluate_model_psnr(model, trajectory=trajectory, device=device, batch_size=batch_size)
        # We maximize PSNR
        trial.set_user_attr("psnr", float(psnr))
        return float(psnr)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=max_trials)

    best_params = study.best_trial.params
    # Inject fixed params (if any) to make the full set explicit
    for k, v in (fixed or {}).items():
        best_params.setdefault(k, v)

    return {"params": best_params, "psnr": float(study.best_value)}
