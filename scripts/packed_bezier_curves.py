"""
Packed Bézier curves generator and demo.

This script provides utilities to:
- Generate evenly spaced points along a cubic Bézier curve (fixed spacing, truncates count if short)
- Place multiple non-overlapping Bézier curves inside [0,1]^2 using Poisson-disk anchors
- Visualize the packed curves with optional anchors and tile bboxes

Usage examples:
  # Show a demo of packed curves
  python scripts/packed_bezier_curves.py --mode packed --max-curves 32 --spacing 0.02 --seed 2026

  # Save a figure instead of showing it
  python scripts/packed_bezier_curves.py --mode packed --save outputs/packed_beziers.png

  # Single curve demo
  python scripts/packed_bezier_curves.py --mode single --n-points 200 --spacing 0.02
"""
from __future__ import annotations

import argparse
import math
from typing import Optional, Tuple, Dict, Union, List, cast

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ------------------------------
# Random helpers
# ------------------------------
def _make_generator(seed: Optional[int] = None) -> Optional[torch.Generator]:
    if seed is None:
        return None
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g


# ------------------------------
# Single cubic Bézier + arc-length resampling
# ------------------------------
def _bezier_cubic(P: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bézier at parameter t.

    Args:
        P: Control points, shape [4, 2]
        t: Parameter(s) in [0,1], shape [M] or []
    Returns:
        Points shape [M, 2] (or [2] if scalar t)
    """
    assert P.shape == (4, 2), f"P must be [4,2], got {tuple(P.shape)}"
    one_minus_t = 1.0 - t
    t2 = t * t
    t3 = t2 * t
    omt2 = one_minus_t * one_minus_t
    omt3 = omt2 * one_minus_t

    b0 = omt3
    b1 = 3.0 * omt2 * t
    b2 = 3.0 * one_minus_t * t2
    b3 = t3

    def bcast(x: torch.Tensor) -> torch.Tensor:
        return x if t.ndim == 0 else x.unsqueeze(0).expand(t.shape[0], -1)

    return (
        b0.unsqueeze(-1) * bcast(P[0]) +
        b1.unsqueeze(-1) * bcast(P[1]) +
        b2.unsqueeze(-1) * bcast(P[2]) +
        b3.unsqueeze(-1) * bcast(P[3])
    )


@torch.no_grad()
def random_bezier_points(
    n_points: int,
    spacing: float,
    ctrl_mode: str = 'box',
    margin: float = 0.25,
    center_sigma: float = 0.25,
    oversample: int = 512,
    max_attempts: int = 8,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
    return_meta: bool = False,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    reject_short: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Generate a cubic Bézier curve with fixed arc-length spacing between points.

    If the curve is too short for (n_points-1)*spacing, the function resamples control
    points and retries up to `max_attempts`. If `reject_short=True` and after
    `max_attempts` the curve is still too short, a RuntimeError is raised. Otherwise,
    it returns as many points as fit (spacing preserved).

    Control points are sampled near [0,1]^2, or inside the provided bbox if given.
    """
    if not isinstance(n_points, int) or n_points < 1:
        raise ValueError(f"n_points must be an int >= 1, got {n_points}")
    if not (isinstance(spacing, (int, float)) and spacing > 0):
        raise ValueError(f"spacing must be a positive number, got {spacing}")
    if ctrl_mode not in {'box', 'gaussian'}:
        raise ValueError(f"ctrl_mode must be 'box' or 'gaussian', got {ctrl_mode}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if center_sigma <= 0:
        raise ValueError(f"center_sigma must be > 0, got {center_sigma}")
    if oversample < 64:
        raise ValueError("oversample should be >= 64 for reasonable accuracy")

    gen = _make_generator(seed)

    def sample_ctrl() -> torch.Tensor:
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            low = torch.tensor([xmin, ymin], dtype=dtype)
            high = torch.tensor([xmax, ymax], dtype=dtype)
            P = low + (high - low) * torch.rand((4, 2), generator=gen, dtype=dtype)
            return P
        if ctrl_mode == 'box':
            low = torch.tensor([-margin, -margin], dtype=dtype)
            high = torch.tensor([1.0 + margin, 1.0 + margin], dtype=dtype)
            P = low + (high - low) * torch.rand((4, 2), generator=gen, dtype=dtype)
            # Re-center around [0.5, 0.5]
            P = P - P.mean(dim=0, keepdim=True) + torch.tensor([0.5, 0.5], dtype=dtype)
        else:
            mu = torch.tensor([0.5, 0.5], dtype=dtype)
            std = torch.tensor([center_sigma, center_sigma], dtype=dtype)
            P = mu + std * torch.randn((4, 2), generator=gen, dtype=dtype)
        return P

    required_len = (n_points - 1) * spacing

    # Try up to max_attempts to find a curve with sufficient length
    total_len = 0.0
    for attempt in range(max_attempts):
        P = sample_ctrl()
        t_vals = torch.linspace(0.0, 1.0, oversample, dtype=dtype)
        curve_pts = _bezier_cubic(P, t_vals)  # [M,2]
        if curve_pts.shape[0] < 2:
            S = torch.tensor([0.0], dtype=dtype)
            total_len = 0.0
        else:
            deltas = curve_pts[1:] - curve_pts[:-1]
            seg_lens = torch.linalg.norm(deltas, dim=-1)
            S = torch.cat([torch.zeros(1, dtype=dtype), torch.cumsum(seg_lens, dim=0)], dim=0)
            total_len = S[-1].item()
        if total_len >= required_len or n_points == 1:
            break

    # If still too short after retries, either raise (reject) or proceed with truncation
    if n_points > 1 and total_len < required_len and reject_short:
        raise RuntimeError(
            f"Bézier curve too short after {max_attempts} attempts: length={total_len:.6f} < required={required_len:.6f}"
        )

    # Targets along arc length (fixed spacing, truncate if needed)
    if n_points == 1:
        s_targets = torch.tensor([0.5 * total_len], dtype=dtype)
    else:
        s_full = torch.arange(n_points, dtype=dtype) * spacing
        s_targets = s_full[s_full <= (S[-1] if S.ndim > 0 else torch.tensor(0.0, dtype=dtype))]
        if s_targets.numel() == 0:
            s_targets = torch.tensor([0.0], dtype=dtype)

    # Inverse arc-length mapping by linear interpolation
    idx_right = torch.searchsorted(S, s_targets, right=False)
    idx_right = torch.clamp(idx_right, 1, max(1, S.shape[0] - 1))
    idx_left = idx_right - 1

    S_left = S[idx_left]
    S_right = S[idx_right]
    denom = (S_right - S_left)
    denom = torch.where(denom <= 1e-12, torch.ones_like(denom), denom)
    w = (s_targets - S_left) / denom

    P_left = curve_pts[idx_left]
    P_right = curve_pts[idx_right]
    out_pts = P_left + w.unsqueeze(-1) * (P_right - P_left)

    out_pts = out_pts.to(device=device, dtype=dtype)

    meta: Dict[str, torch.Tensor] = {
        'control_points': P.to(device=device, dtype=dtype),
        'total_length': torch.tensor([total_len], dtype=dtype, device=device),
        'cumlen': S.to(device=device, dtype=dtype),
    }

    return (out_pts, meta) if return_meta else out_pts

# ------------------------------
# Poisson-disk anchors + packing
# ------------------------------
def _poisson_disk_2d(radius: float, k: int = 30, seed: int = 0,
                      width: float = 1.0, height: float = 1.0) -> np.ndarray:
    """Bridson's Poisson-disk sampling in 2D."""
    rng = np.random.default_rng(seed)
    cell_size = radius / np.sqrt(2.0)
    grid_shape = (int(np.ceil(height / cell_size)), int(np.ceil(width / cell_size)))
    grid = -np.ones(grid_shape, dtype=int)  # store indices of samples

    def grid_coords(p):
        return int(p[1] / cell_size), int(p[0] / cell_size)

    samples = []
    active = []

    # Initial point
    p0 = np.array([rng.uniform(0, width), rng.uniform(0, height)], dtype=np.float32)
    samples.append(p0)
    gi, gj = grid_coords(p0)
    grid[gi, gj] = 0
    active.append(0)

    def in_domain(p):
        return (0 <= p[0] < width) and (0 <= p[1] < height)

    def far_enough(p):
        gi, gj = grid_coords(p)
        i0 = max(gi - 2, 0)
        i1 = min(gi + 3, grid.shape[0])
        j0 = max(gj - 2, 0)
        j1 = min(gj + 3, grid.shape[1])
        for ii in range(i0, i1):
            for jj in range(j0, j1):
                idx = grid[ii, jj]
                if idx >= 0:
                    q = samples[idx]
                    if np.linalg.norm(p - q) < radius:
                        return False
        return True

    while active:
        idx = rng.integers(0, len(active))
        s_idx = active[idx]
        s = samples[s_idx]
        placed = False
        for _ in range(k):
            r = rng.uniform(radius, 2 * radius)
            theta = rng.uniform(0, 2 * np.pi)
            p = s + r * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            if in_domain(p) and far_enough(p):
                samples.append(p)
                gi, gj = grid_coords(p)
                grid[gi, gj] = len(samples) - 1
                active.append(len(samples) - 1)
                placed = True
                break
        if not placed:
            active.pop(idx)

    return np.array(samples, dtype=np.float32)


def _clamp_bbox(xc: float, yc: float, half: float,
                xmin: float = 0.0, ymin: float = 0.0, xmax: float = 1.0, ymax: float = 1.0) -> Tuple[float, float, float, float]:
    x0 = max(xmin, xc - half)
    y0 = max(ymin, yc - half)
    x1 = min(xmax, xc + half)
    y1 = min(ymax, yc + half)
    return (x0, y0, x1, y1)


def _min_distance_to_set(A: torch.Tensor, B: torch.Tensor) -> float:
    """Compute min pairwise distance between points in A and B."""
    if A.numel() == 0 or B.numel() == 0:
        return float('inf')
    d = torch.cdist(A, B, p=2)
    return float(d.min().item())


def generate_packed_bezier_curves(
    max_curves: int = 20,
    n_points_per_curve: int = 200,
    spacing: float = 0.02,
    anchor_radius: float = 0.12,
    tile_half: float = 0.12,
    clearance: float = 0.02,
    oversample: int = 512,
    seed: int = 123,
    device: Union[str, torch.device] = 'cpu',
    check_stride: int = 1,
    max_attempts: int = 5,
    reject_short: bool = False,
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]], np.ndarray]:
    """Place multiple non-overlapping Bézier curves using Poisson-disk anchors."""
    rng = np.random.default_rng(seed)
    anchors = _poisson_disk_2d(radius=anchor_radius, k=30, seed=seed, width=1.0, height=1.0)
    # Sort anchors by proximity to center for nicer coverage order
    ctr = np.array([0.5, 0.5], dtype=np.float32)
    order = np.argsort(np.linalg.norm(anchors - ctr, axis=1))
    anchors = anchors[order]

    curves: List[torch.Tensor] = []
    metas: List[Dict[str, torch.Tensor]] = []

    # Aggregated points for fast distance checks
    all_pts = torch.empty((0, 2), dtype=torch.float32, device=device)

    for (ax, ay) in anchors:
        if len(curves) >= max_curves:
            break
        # Build bbox around anchor, clamp to unit square
        bbox = _clamp_bbox(float(ax), float(ay), tile_half)

        for _ in range(max(1, max_attempts)):
            if len(curves) >= max_curves:
                break
            curve_seed = int(rng.integers(0, 2**31 - 1))
            pts, meta = cast(Tuple[torch.Tensor, Dict[str, torch.Tensor]] ,random_bezier_points(
                n_points=n_points_per_curve,
                spacing=spacing,
                ctrl_mode='box',
                margin=0.0,
                center_sigma=0.25,
                oversample=oversample,
                max_attempts=max_attempts,
                seed=curve_seed,
                device=device,
                return_meta=True,
                bbox=bbox,
                reject_short=reject_short,
            ))
            # Subsample for checks to reduce cost, but keep endpoints
            if check_stride > 1 and pts.shape[0] > 2:
                mask = torch.zeros(pts.shape[0], dtype=torch.bool)
                mask[0] = True
                mask[-1] = True
                mask[1:-1:check_stride] = True
                pts_check = pts[mask]
            else:
                pts_check = pts

            # Check clearance to all previous curves
            mind = _min_distance_to_set(pts_check, all_pts) if all_pts.numel() > 0 else float('inf')
            if mind >= clearance:
                curves.append(pts)
                metas.append(meta)
                all_pts = torch.cat([all_pts, pts_check.to(device=device, dtype=torch.float32)], dim=0)
                break  # accepted for this anchor

    return curves, metas, anchors


# ------------------------------
# Visualization
# ------------------------------
def plot_random_bezier(
    n_points: int = 101,
    spacing: float = 0.02,
    ctrl_mode: str = 'box',
    margin: float = 0.25,
    center_sigma: float = 0.25,
    oversample: int = 512,
    seed: int = 123,
    device: str = 'cpu',
    show_ctrl: bool = True,
):
    pts, meta = cast(Tuple[torch.Tensor, Dict[str, torch.Tensor]], random_bezier_points(
        n_points=n_points,
        spacing=spacing,
        ctrl_mode=ctrl_mode,
        margin=margin,
        center_sigma=center_sigma,
        oversample=oversample,
        seed=seed,
        device=device,
        return_meta=True,
    ))

    pts_np = pts.detach().cpu().numpy()
    C_np = meta['control_points'].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot(pts_np[:, 0], pts_np[:, 1], '-o', ms=2.5, lw=1.0, color='#9467bd', label='bezier points')
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, linestyle='--', edgecolor='gray', linewidth=1.0, label='[0,1]^2'))
    if show_ctrl:
        ax.plot(C_np[:, 0], C_np[:, 1], 's--', color='orange', lw=0.8, ms=4, label='control polygon')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Cubic Bézier (<= {n_points} pts, spacing={spacing})')
    ax.legend(loc='best')
    return fig, ax


def plot_packed_curves(
    max_curves: int = 20,
    n_points_per_curve: int = 200,
    spacing: float = 0.02,
    anchor_radius: float = 0.12,
    tile_half: float = 0.12,
    clearance: float = 0.02,
    oversample: int = 512,
    seed: int = 123,
    device: str = 'cpu',
    show_anchors: bool = True,
    show_bboxes: bool = True,
    check_stride: int = 2,
    per_anchor_attempts: int = 5,
):
    curves, metas, anchors = generate_packed_bezier_curves(
        max_curves=max_curves,
        n_points_per_curve=n_points_per_curve,
        spacing=spacing,
        anchor_radius=anchor_radius,
        tile_half=tile_half,
        clearance=clearance,
        oversample=oversample,
        seed=seed,
        device=device,
        check_stride=check_stride,
        per_anchor_attempts=per_anchor_attempts,
    )

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linestyle='--', edgecolor='gray', linewidth=1.0))

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(curves))))
    for i, pts in enumerate(curves):
        P = pts.detach().cpu().numpy()
        ax.plot(P[:, 0], P[:, 1], '-', lw=1.2, color=colors[i % len(colors)])
        ax.scatter(P[::max(1, pts.shape[0] // 50), 0], P[::max(1, pts.shape[0] // 50), 1], s=6, color=colors[i % len(colors)], alpha=0.7)

    if show_anchors and anchors.size > 0:
        ax.scatter(anchors[:, 0], anchors[:, 1], s=10, c='black', alpha=0.3, label='anchors')

    if show_bboxes and anchors.size > 0:
        first = True
        for (axc, ayc) in anchors:
            x0, y0, x1, y1 = _clamp_bbox(float(axc), float(ayc), tile_half)
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                             fill=False, linestyle=':', edgecolor='black', linewidth=0.8, alpha=0.4,
                             label='bboxes' if first else None)
            ax.add_patch(rect)
            first = False

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best')
    return fig, ax


# ------------------------------
# CLI
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Packed Bézier curves generator and demo")
    p.add_argument('--mode', choices=['single', 'packed'], default='packed', help='Demo mode: single curve or packed curves')
    p.add_argument('--seed', type=int, default=123, help='RNG seed')
    p.add_argument('--device', type=str, default='cpu', help="Torch device for outputs ('cpu'|'cuda'|'mps')")
    p.add_argument('--save', type=str, default=None, help='Path to save the figure (PNG). If omitted, shows the plot')

    # Single
    p.add_argument('--n-points', type=int, default=200, help='Number of points for single curve (upper bound)')
    p.add_argument('--spacing', type=float, default=0.02, help='Fixed spacing between points')

    # Packed
    p.add_argument('--max-curves', type=int, default=24, help='Max number of curves to place')
    p.add_argument('--n-points-per-curve', type=int, default=160, help='Requested points per curve (actual may be <=)')
    p.add_argument('--anchor-radius', type=float, default=0.14, help='Poisson-disk min distance between anchors')
    p.add_argument('--tile-half', type=float, default=0.14, help='Half-size of bbox around each anchor')
    p.add_argument('--clearance', type=float, default=0.03, help='Minimum allowed distance between curves')
    p.add_argument('--oversample', type=int, default=512, help='Oversampling for arc-length computation')
    p.add_argument('--check-stride', type=int, default=2, help='Subsample stride for clearance checks')
    p.add_argument('--per-anchor-attempts', type=int, default=5, help='Retries per anchor if clearance fails')
    p.add_argument('--no-anchors', action='store_true', help='Hide anchor points in plot')
    p.add_argument('--no-bboxes', action='store_true', help='Hide tile bboxes in plot')
    return p


def main():
    args = build_argparser().parse_args()

    if args.mode == 'single':
        fig, ax = plot_random_bezier(
            n_points=args.n_points,
            spacing=args.spacing,
            seed=args.seed,
            device=args.device,
            show_ctrl=True,
        )
    else:
        fig, ax = plot_packed_curves(
            max_curves=args.max_curves,
            n_points_per_curve=args.n_points_per_curve,
            spacing=args.spacing,
            anchor_radius=args.anchor_radius,
            tile_half=args.tile_half,
            clearance=args.clearance,
            oversample=args.oversample,
            seed=args.seed,
            device=args.device,
            show_anchors=(not args.no_anchors),
            show_bboxes=(not args.no_bboxes),
            check_stride=args.check_stride,
            per_anchor_attempts=args.per_anchor_attempts,
        )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
