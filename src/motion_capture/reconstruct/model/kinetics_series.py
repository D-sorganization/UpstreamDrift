"""Capture-rig inverse dynamics kinetics converted to ForceTorqueSeries (FTO-25, #11310).

Authorities and conventions:
- Frame: ADR-0041 camera world (gravity (0, -9.81, 0), Y-up). The capture-rig
  model is already constructed in ADR-0041 world coordinates, so registration
  and canonical Z-up conversions are bypassed; the frame is labeled 'adr0041_world'
  directly.
- Provenance: "capture-model inverse dynamics (point-mass model)". This is a
  simplified model; it is not engine-qualified.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    joint_torque_wrench,
)

from .kinematics import ArticulatedModel

__all__ = ["kinetics_to_force_series"]

_AXIS_UNIT: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


def _extract_inputs(
    kinetics: Mapping[str, Any] | np.ndarray | list[Any],
    model: ArticulatedModel,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Parse and validate tau, q, and fps from input kinetics."""
    if isinstance(kinetics, Mapping):
        tau = np.asarray(kinetics["tau"], dtype=np.float64)
        raw_q = kinetics.get("q")
        q = np.asarray(raw_q, dtype=np.float64) if raw_q is not None else None
        fps = float(kinetics.get("fps", 30.0))
    else:
        tau = np.asarray(kinetics, dtype=np.float64)
        q = None
        fps = 30.0

    if tau.ndim == 1:
        tau = tau[np.newaxis, :]
    require(tau.ndim == 2, "tau must be 1D or 2D array")
    require(fps > 0, "fps must be positive")

    t_frames = tau.shape[0]
    if q is None:
        q = np.zeros((t_frames, model.n_dof), dtype=np.float64)
    else:
        if q.ndim == 1:
            q = q[np.newaxis, :]
        require(q.shape == (t_frames, model.n_dof), "q must match (T, n_dof)")

    rot_dofs = sum(len(j.axes) for j in model.joints)
    if tau.shape[1] == model.n_dof:
        use_slices = True
    elif tau.shape[1] == rot_dofs:
        use_slices = False
    else:
        raise ValueError(
            f"tau columns ({tau.shape[1]}) must match model.n_dof ({model.n_dof}) "
            f"or rotational DOFs ({rot_dofs})"
        )

    return tau, q, fps, use_slices


def _build_frame_wrenches(
    model: ArticulatedModel,
    pos_t: np.ndarray,
    frames_t: np.ndarray,
    tau_t: np.ndarray,
    use_slices: bool,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Build actuator wrenches and collect unavailable labels for one frame."""
    wrenches = []
    unavailable: list[str] = []
    rot_col = 0

    for i, j in enumerate(model.joints):
        anchor = (float(pos_t[i, 0]), float(pos_t[i, 1]), float(pos_t[i, 2]))
        if not j.axes:
            unavailable.append(f"actuator:{j.name}")
            continue

        n_ax = len(j.axes)
        if use_slices:
            a, b = model._slices[i]
            tau_j = tau_t[a:b]
        else:
            tau_j = tau_t[rot_col : rot_col + n_ax]
            rot_col += n_ax

        r_matrix = frames_t[i]
        if n_ax == 1:
            ax = r_matrix @ _AXIS_UNIT[j.axes[0]]
            norm = float(np.linalg.norm(ax))
            if norm > 1e-12:
                ax = ax / norm
            w = joint_torque_wrench(
                f"actuator:{j.name}",
                j.name,
                float(tau_j[0]),
                ax,
                anchor,
                "capture-model",
            )
            wrenches.append(w)
        else:
            axes_list = []
            for ch in j.axes:
                ax = r_matrix @ _AXIS_UNIT[ch]
                norm = float(np.linalg.norm(ax))
                if norm > 1e-12:
                    ax = ax / norm
                axes_list.append(ax)
            w = joint_torque_wrench(
                f"actuator:{j.name}",
                j.name,
                [float(x) for x in tau_j],
                np.asarray(axes_list, dtype=np.float64),
                anchor,
                "capture-model",
            )
            wrenches.append(w)

    return tuple(wrenches), tuple(unavailable)


def kinetics_to_force_series(
    kinetics: Mapping[str, Any] | np.ndarray | list[Any],
    model: ArticulatedModel,
) -> ForceTorqueSeries:
    """Convert inverse dynamics kinetics into an ADR-0041 world ForceTorqueSeries.

    Bypasses canonical Z-up registration because capture-rig models are
    evaluated directly in the camera world frame.
    """
    tau, q, fps, use_slices = _extract_inputs(kinetics, model)
    positions, frames = model.forward_frames(q)
    t_frames = tau.shape[0]
    out_frames = []

    for t in range(t_frames):
        time_s = float(t / fps)
        wrenches, unavailable = _build_frame_wrenches(
            model, positions[t], frames[t], tau[t], use_slices
        )
        metadata = {
            "unavailable_labels": unavailable,
            "provenance": "capture-model inverse dynamics (point-mass model)",
            "convention": "adr0041_world y-up (direct capture-rig camera frame, no Z-up conversion)",
        }
        out_frames.append(
            ForceTorqueFrame(
                time_s=time_s,
                engine="capture_model",
                world_frame="adr0041_world",
                wrenches=wrenches,
                metadata=metadata,
            )
        )

    return ForceTorqueSeries(engine="capture_model", frames=tuple(out_frames))
