"""Offline SVD-direction ablation vs high-frequency jitter benchmark (#14501).

Diagnostic: evaluates rank-r singular direction ablations on power >15 Hz.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.research.neural_policy.edits import (
    LowRankEditRecipe,
    apply_low_rank_weight_edit,
)
from src.shared.python.core.contracts import require


def compute_high_frequency_power(
    signal: NDArray[np.floating],
    sample_rate_hz: float = 100.0,
    cutoff_hz: float = 15.0,
) -> float:
    """Compute spectral power above cutoff_hz using real Fourier transform."""
    require(signal.ndim in (1, 2) and sample_rate_hz > 0.0, "valid signal and rate")
    sig = np.asarray(signal, dtype=float)
    n = sig.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    mask = freqs > cutoff_hz
    if not np.any(mask):
        return 0.0
    power = (np.abs(np.fft.rfft(sig, axis=0)) ** 2) / (n * n)
    return float(np.sum(power[mask]))


def low_pass_filter_signal(
    signal: NDArray[np.floating],
    sample_rate_hz: float = 100.0,
    cutoff_hz: float = 15.0,
) -> NDArray[np.floating]:
    """Zero-out frequency components above cutoff_hz via rfft."""
    require(signal.ndim in (1, 2), "signal must be 1D or 2D")
    sig = np.asarray(signal, dtype=float)
    n = sig.shape[0]
    fft_vals = np.fft.rfft(sig, axis=0)
    fft_vals[np.fft.rfftfreq(n, d=1.0 / sample_rate_hz) > cutoff_hz] = 0.0
    return np.fft.irfft(fft_vals, n=n, axis=0)


def ablate_top_r_directions(
    weight: NDArray[np.floating],
    r: int,
) -> tuple[NDArray[np.floating], list[LowRankEditRecipe]]:
    """Ablate top-r singular directions using rank-1 edit operator (#14478)."""
    require(weight.ndim == 2 and r >= 0, "weight must be 2D and r >= 0")
    w_curr = np.asarray(weight, dtype=float).copy()
    recipes: list[LowRankEditRecipe] = []
    if r == 0:
        return w_curr, recipes
    u_mat, _, _ = np.linalg.svd(w_curr, full_matrices=False)
    for i in range(min(r, u_mat.shape[1])):
        w_curr, recipe = apply_low_rank_weight_edit(w_curr, u_mat[:, i], alpha=1.0)
        recipes.append(recipe)
    return w_curr, recipes


def run_jitter_ablation_benchmark(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    k_max_rank: int = 2,
    sample_rate_hz: float = 100.0,
    cutoff_hz: float = 15.0,
) -> dict[str, Any]:
    """Benchmark top-r singular ablation vs low-pass baseline across seeds."""
    require(len(seeds) >= 5, "require >= 5 seeds for reporting", len(seeds))
    n_steps = 200
    t = np.linspace(0.0, (n_steps - 1) / sample_rate_hz, n_steps)
    rows: list[dict[str, Any]] = []

    for s in seeds:
        rng = np.random.default_rng(s)
        x_clean = np.column_stack([np.sin(4.0 * np.pi * t), np.cos(6.0 * np.pi * t)])
        x_demo = x_clean + 0.4 * np.sin(56.0 * np.pi * t)[:, None] * np.array(
            [[1.0, 0.5]]
        )
        w_base = rng.standard_normal((3, 2))
        u_mat, _, _ = np.linalg.svd(w_base, full_matrices=False)
        w_base = w_base + 3.0 * np.outer(u_mat[:, 0], np.array([1.0, 0.5]))
        tau_target = x_clean @ w_base.T
        tau_base = x_demo @ w_base.T

        row: dict[str, Any] = {
            "seed": s,
            "singular_spectrum": [
                float(v) for v in np.linalg.svd(w_base, compute_uv=False)
            ],
            "baseline": {
                "power_gt_15hz": compute_high_frequency_power(
                    tau_base, sample_rate_hz, cutoff_hz
                ),
                "tracking_rmse": float(np.sqrt(np.mean((tau_base - tau_target) ** 2))),
            },
            "ablations": {},
        }
        for r in range(1, k_max_rank + 1):
            w_abl, _ = ablate_top_r_directions(w_base, r=r)
            t_abl = x_demo @ w_abl.T
            row["ablations"][f"r_{r}"] = {
                "power_gt_15hz": compute_high_frequency_power(
                    t_abl, sample_rate_hz, cutoff_hz
                ),
                "tracking_rmse": float(np.sqrt(np.mean((t_abl - tau_target) ** 2))),
            }
        t_lp = low_pass_filter_signal(x_demo, sample_rate_hz, cutoff_hz) @ w_base.T
        row["low_pass_baseline"] = {
            "power_gt_15hz": compute_high_frequency_power(
                t_lp, sample_rate_hz, cutoff_hz
            ),
            "tracking_rmse": float(np.sqrt(np.mean((t_lp - tau_target) ** 2))),
        }
        rows.append(row)

    mean_r1 = float(np.mean([r["ablations"]["r_1"]["power_gt_15hz"] for r in rows]))
    mean_lp = float(np.mean([r["low_pass_baseline"]["power_gt_15hz"] for r in rows]))
    return {
        "provenance": {
            "issue": 14501,
            "parent_epic": 14475,
            "provenance_spec": "#14436/#14476",
            "seeds": list(seeds),
            "cutoff_hz": cutoff_hz,
        },
        "rows": rows,
        "ablation_beat_low_pass": bool(mean_r1 < mean_lp),
        "disclaimer": (
            "Ablation is a diagnostic, not a repair claim. "
            "No claim is made about real markerless mocap until tested on real captures."
        ),
    }
