"""MuJoCo cross-version compatibility helpers.

MuJoCo 3.13 removed ``MjData.qM`` and changed the signature of ``mj_fullM``
from ``mj_fullM(model, dst, data.qM)`` to ``mj_fullM(model, data, dst)``.
This module provides a unified helper that detects the available signature
at runtime while avoiding top-level imports of ``mujoco`` so that AST-based
import guards remain satisfied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["copy_mjdata_state", "full_mass_matrix"]


def copy_mjdata_state(dst: Any, src: Any) -> None:
    """Copy dynamic simulation state slices from src MjData into dst MjData.

    Copies qpos, qvel, qacc, ctrl, act, qfrc_applied, xfrc_applied, and time
    without requiring full MjData cloning.
    """
    dst.time = float(src.time)
    np.copyto(dst.qpos, src.qpos)
    np.copyto(dst.qvel, src.qvel)
    np.copyto(dst.qacc, src.qacc)
    np.copyto(dst.ctrl, src.ctrl)
    np.copyto(dst.act, src.act)
    np.copyto(dst.qfrc_applied, src.qfrc_applied)
    np.copyto(dst.xfrc_applied, src.xfrc_applied)


def full_mass_matrix(
    mj: Any,
    model: Any,
    data: Any,
    dst: np.ndarray | None = None,
) -> np.ndarray:
    """Return the dense joint-space inertia matrix ``M(q)`` across MuJoCo versions.

    Handles MuJoCo < 3.13 (where ``mj_fullM(model, dst, data.qM)`` requires
    ``data.qM``) and MuJoCo >= 3.13 (where ``data.qM`` was removed and the
    signature changed to ``mj_fullM(model, data, dst)``).

    Preconditions:
        - ``model`` must have an ``nv`` attribute with integer value >= 0.
        - If ``dst`` is provided, ``dst.shape`` must equal ``(nv, nv)``.

    Postconditions:
        - Returns a 2-D float array of shape ``(nv, nv)``.

    Args:
        mj: The MuJoCo module or wrapper providing ``mj_fullM``.
        model: The MuJoCo model instance.
        data: The MuJoCo data instance.
        dst: Optional preallocated destination array of shape ``(nv, nv)``.

    Returns:
        The dense ``(nv, nv)`` inertia matrix (populating ``dst`` if provided).

    Raises:
        ValueError: If ``model.nv`` is missing or negative, or if ``dst`` shape
            does not match ``(nv, nv)``, or if the resulting array shape is invalid.
    """
    if not hasattr(model, "nv") or model.nv is None:
        raise ValueError(
            f"model must have 'nv' attribute >= 0, got {getattr(model, 'nv', None)}"
        )
    try:
        nv = int(model.nv)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"model.nv must be an integer >= 0, got {getattr(model, 'nv', None)}"
        ) from err
    if nv < 0:
        raise ValueError(f"model.nv must be >= 0, got {nv}")

    if dst is None:
        out = np.zeros((nv, nv), dtype=np.float64)
    else:
        if dst.shape != (nv, nv):
            raise ValueError(f"dst must have shape ({nv}, {nv}), got {dst.shape}")
        out = dst

    if hasattr(data, "qM"):
        mj.mj_fullM(model, out, data.qM)
    else:
        mj.mj_fullM(model, data, out)

    if out.shape != (nv, nv):
        raise ValueError(f"Result shape must be ({nv}, {nv}), got {out.shape}")

    return out
