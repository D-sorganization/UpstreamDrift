"""Inference surface for :class:`TimestepInverseDynamics`.

Loads a checkpoint payload, de-standardises the predicted torques using
the stats persisted at training time, and returns physical-unit (Nm)
torques.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from src.shared.python.motion_matching._checkpoint_artifacts import load_checkpoint_dict

from .model import TimestepInverseDynamics

logger = logging.getLogger(__name__)


def predict_torques(
    model: TimestepInverseDynamics,
    q: ArrayLike,
    qd: ArrayLike,
    qdd: ArrayLike,
    *,
    state_stats: dict | None = None,
    tau_stats: dict | None = None,
) -> NDArray[np.float32]:
    """Predict per-timestep torques in physical units (Nm).

    The standardisation stats may be supplied directly via the keyword
    arguments or attached to the model as ``model._state_stats`` /
    ``model._tau_stats`` (the loaders below set those attributes).

    Parameters
    ----------
    model
        Trained :class:`TimestepInverseDynamics`.
    q, qd, qdd
        ``(B, 27)`` arrays of generalised coordinates/velocities/accelerations,
        in physical units. NaN entries (unmapped DOFs) are tolerated.
    state_stats
        ``{"mean": (81,), "std": (81,)}`` for input standardisation.
    tau_stats
        ``{"mean": (27,), "std": (27,)}`` for output de-standardisation.

    Returns
    -------
    np.ndarray
        ``(B, 27)`` float32 torques in Newton-metres.

    Raises
    ------
    TypeError, ValueError
        For shape/dtype contract violations or missing standardisation stats.
    """
    if not isinstance(model, TimestepInverseDynamics):
        raise TypeError(
            f"model must be TimestepInverseDynamics, got {type(model).__name__}"
        )
    state_stats = (
        state_stats
        if state_stats is not None
        else _attached_stats(model, "_state_stats")
    )
    tau_stats = (
        tau_stats if tau_stats is not None else _attached_stats(model, "_tau_stats")
    )
    if state_stats is None or tau_stats is None:
        raise ValueError(
            "predict_torques requires standardisation stats (state_stats and "
            "tau_stats); attach them to the model or pass them explicitly"
        )

    q_arr = _as_2d(q, name="q")
    qd_arr = _as_2d(qd, name="qd")
    qdd_arr = _as_2d(qdd, name="qdd")
    n_joints = model.cfg.output_dim
    for arr, name in ((q_arr, "q"), (qd_arr, "qd"), (qdd_arr, "qdd")):
        if arr.shape[-1] != n_joints:
            raise ValueError(
                f"{name}.shape[-1] must equal n_joints={n_joints}; got {arr.shape[-1]}"
            )
        if arr.shape[0] != q_arr.shape[0]:
            raise ValueError(
                f"batch sizes must match; got {arr.shape[0]} vs {q_arr.shape[0]}"
            )

    raw_state = np.concatenate([q_arr, qd_arr, qdd_arr], axis=-1).astype(
        np.float32, copy=False
    )
    state_mean = np.asarray(state_stats["mean"], dtype=np.float32)
    state_std = np.asarray(state_stats["std"], dtype=np.float32)
    tau_mean = np.asarray(tau_stats["mean"], dtype=np.float32)
    tau_std = np.asarray(tau_stats["std"], dtype=np.float32)
    standardised = _standardise(raw_state, state_mean, state_std)
    state_t = torch.from_numpy(standardised)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    state_t = state_t.to(device)

    model.eval()
    with torch.no_grad():
        pred_std = model(state_t)
    pred_std_np = pred_std.detach().cpu().numpy().astype(np.float32)
    return (pred_std_np * tau_std + tau_mean).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attached_stats(model: TimestepInverseDynamics, attr: str) -> dict | None:
    return getattr(model, attr, None)


def _as_2d(arr: ArrayLike, *, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(a.shape)}")
    return a


def _standardise(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(raw - mean) / std`` with NaN preserved and divide-by-zero guarded.

    Standard deviations <= 1e-8 are floored to 1.0 so a constant column
    (e.g. an unmapped DOF with all-NaN entries) does not explode.
    """
    safe_std = np.where(std > 1e-8, std, np.float32(1.0)).astype(np.float32)
    return ((raw - mean) / safe_std).astype(np.float32)


def load_with_stats(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> TimestepInverseDynamics:
    """Load a checkpoint and attach the standardisation stats to the model.

    Convenience wrapper that mirrors the cVAE / regressor loaders. The
    returned model has ``_state_stats`` and ``_tau_stats`` attributes
    suitable for :func:`predict_torques`.
    """
    ckpt_path = Path(checkpoint_path)
    payload = load_checkpoint_dict(
        ckpt_path,
        map_location=map_location,
        required_keys=("state_dict", "config", "schema_version"),
        artifact_name="TimestepInverseDynamics checkpoint",
    )
    model = TimestepInverseDynamics.from_checkpoint(
        ckpt_path, map_location=map_location
    )
    state_stats = payload.get("state_stats")
    tau_stats = payload.get("tau_stats")
    if state_stats is not None:
        model._state_stats = state_stats  # noqa: SLF001 - intentional sidecar
    if tau_stats is not None:
        model._tau_stats = tau_stats  # noqa: SLF001 - intentional sidecar
    return model
