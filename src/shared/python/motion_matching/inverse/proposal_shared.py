"""Shared NM-06 proposal helpers (inverse + neural_motion call sites).

Keeps software-contract plant, config validation, trajectory coercion, and
native polish parsing in one place so the DRY duplication gate does not see
parallel copies across motion_matching and neural_motion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "coefficients_from_polish_mapping",
    "coerce_finite_trajectory_and_times",
    "coerce_training_output_dir",
    "mean_modes_succeed_mean_fails",
    "parse_native_polish_outcome",
    "require_positive_int",
    "require_positive_training_hparams",
    "software_contract_plant_residual",
]


def require_positive_int(name: str, value: int) -> None:
    """Raise ``ValueError`` when ``value`` is not a positive integer."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_positive_training_hparams(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    control_regularization: float,
) -> None:
    """Validate shared proposal-trainer hyperparameter floors."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if control_regularization < 0.0:
        raise ValueError("control_regularization must be >= 0")


def coerce_training_output_dir(output_dir: Path | str) -> Path:
    """Normalize a trainer output directory to ``Path``."""
    return Path(output_dir)


def coerce_finite_trajectory_and_times(
    trajectory: NDArray[np.floating] | Sequence[Sequence[float]],
    sample_times_s: NDArray[np.floating] | Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce trajectory/times arrays and enforce finite 2-D / matching length."""
    traj = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(sample_times_s, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] < 1:
        raise ValueError("trajectory must be 2-D with T >= 1")
    if not bool(np.all(np.isfinite(traj))):
        raise ValueError("trajectory values must be finite")
    if times.ndim != 1 or times.shape[0] != traj.shape[0]:
        raise ValueError("sample_times_s length must equal trajectory T")
    if not bool(np.all(np.isfinite(times))):
        raise ValueError("sample_times_s values must be finite")
    return traj, times


def software_contract_plant_residual(
    *,
    trajectory: NDArray[np.floating],
    observation_mask: Sequence[bool] | NDArray[np.floating],
    controls: NDArray[np.floating],
    duration_s: float,
) -> NDArray[np.float64]:
    """Masked residual of the NM-06 software-contract linear plant.

    Deterministic affine map used only for unit tests and training surrogates.
    Not a native dynamics claim.
    """
    u = np.asarray(controls, dtype=np.float64).reshape(-1)
    traj = np.asarray(trajectory, dtype=np.float64)
    mask = np.asarray(observation_mask, dtype=np.float64)
    if u.size < 1 or not bool(np.all(np.isfinite(u))):
        raise ValueError("controls must be a non-empty finite vector")
    if mask.shape != (traj.shape[1],):
        raise ValueError("observation_mask length must match trajectory channels")
    masked_traj = traj * mask.reshape(1, -1)
    rng = np.random.default_rng(traj.shape[1] * 17 + u.size)
    weight = rng.normal(0.0, 0.25, size=(traj.shape[1], u.size))
    predicted = masked_traj @ weight
    target = np.broadcast_to(u.reshape(1, -1), predicted.shape)
    residual = predicted - target
    time_scale = 1.0 + float(duration_s)
    return residual * time_scale


def mean_modes_succeed_mean_fails(
    modes: Sequence[NDArray[np.floating]],
    *,
    featured: float,
    target_residual_tol: float,
) -> bool:
    """Return True when each mode is feasible but their mean is not.

    Feasibility is ``||u - teacher|| + 0.01*|featured| <= tol`` against each
    mode's own teacher; the mean must fail every teacher.
    """
    if len(modes) < 2:
        raise ValueError("modes must contain at least two control vectors")
    if not np.isfinite(target_residual_tol) or target_residual_tol < 0.0:
        raise ValueError("target_residual_tol must be a finite non-negative float")

    mode_arrs = [np.asarray(m, dtype=np.float64).reshape(-1) for m in modes]
    dim = mode_arrs[0].size
    if any(m.size != dim for m in mode_arrs):
        raise ValueError("all modes must share the same control dimension")

    def _feasibility(u: NDArray[np.float64], teacher: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(u - teacher)) + 0.01 * abs(featured)

    mode_ok = all(_feasibility(m, m) <= target_residual_tol for m in mode_arrs)
    mean_u = np.mean(np.stack(mode_arrs, axis=0), axis=0)
    mean_fail = all(
        _feasibility(mean_u, teacher) > target_residual_tol for teacher in mode_arrs
    )
    return bool(mode_ok and mean_fail)


def coefficients_from_polish_mapping(polish_out: object) -> NDArray[np.float64]:
    """Extract a flat coefficient vector from a polish_fn mapping."""
    if not isinstance(polish_out, dict) or "coefficients" not in polish_out:
        raise ValueError(
            "polish_fn must return a mapping with at least 'coefficients'; "
            f"got {type(polish_out).__name__}"
        )
    return np.asarray(polish_out["coefficients"], dtype=np.float64).reshape(-1)


def parse_native_polish_outcome(
    polish_out: object,
    *,
    warm: NDArray[np.floating],
    require_independent_replay: bool,
) -> tuple[NDArray[np.float64], float, bool, dict[str, Any]]:
    """Parse polish mapping into controls, cost, replay flag, and raw dict."""
    polished = coefficients_from_polish_mapping(polish_out)
    # coefficients_from_polish_mapping already required a mapping.
    mapping = polish_out if isinstance(polish_out, dict) else {}
    warm_arr = np.asarray(warm, dtype=np.float64).reshape(-1)
    if polished.shape != warm_arr.shape or not bool(np.all(np.isfinite(polished))):
        raise ValueError("polished coefficients must be finite and match warm shape")

    independent = bool(mapping.get("independent_replay", False))
    if require_independent_replay and not independent:
        raise ValueError(
            "native refinement requires independent_replay=True; "
            "refusing to accept polish without replay evidence"
        )
    cost = float(mapping.get("projection_cost", float("nan")))
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError("projection_cost must be a finite non-negative float")
    return polished, cost, independent, mapping
