"""Shared force conversions for engine providers and axial loads (ADR-0052, #11287).

Declares:
- joint_torque_wrench: builds OverlayWrench for 1D revolute or multi-axis gimbal joints.
- world_wrench_from_local: rotates local force/torque halves into world frame via R.
- move_wrench_point: moves a wrench to a new application point with moment arm.
- SegmentAxis: defines segment proximal/distal endpoints for axial load computation.
- axial_loads_from_reactions: extracts axial loads (tension/compression) from reaction wrenches.
- frame_with_axial_loads: attaches computed axial loads to a ForceTorqueFrame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.body_part_viz.axial_loads import (
    AxialLoadFrame,
    axial_force_from_proximal_reaction,
)
from src.shared.python.motion_matching.force_torque import (
    SpatialWrench,
    transform_wrench,
    validate_vec3,
)

from .contracts import ForceTorqueFrame, OverlayWrench, WrenchKind


def joint_torque_wrench(
    label: str,
    body: str,
    tau_nm: float | Sequence[float],
    axis_world: ArrayLike,
    anchor_world: Sequence[float],
    source: str,
) -> OverlayWrench:
    """Build a JOINT_ACTUATOR OverlayWrench from joint torque scalar or multi-axis moments.

    Parameters
    ----------
    label : str
        Canonical wrench label (e.g. 'actuator:knee_z').
    body : str
        Name of the body the joint acts upon.
    tau_nm : float | Sequence[float]
        Torque magnitude(s) in N*m. If scalar, axis_world must be shape (3,).
        If sequence of length k, axis_world must be shape (k, 3).
    axis_world : ArrayLike
        Joint rotation axis or axes in world frame. Each axis must have unit norm
        within 1e-6 (never silently normalized).
    anchor_world : Sequence[float]
        Joint anchor location in world coordinates [m].
    source : str
        Producing engine or model identifier.

    Returns
    -------
    OverlayWrench
        Actuator wrench with kind JOINT_ACTUATOR, force_n=None, and torque_nm.
    """
    p_anchor = validate_vec3(anchor_world, "anchor_world")

    axis_arr = np.asarray(axis_world, dtype=np.float64)
    if not np.all(np.isfinite(axis_arr)):
        raise ValueError("axis_world must be finite")

    if isinstance(tau_nm, (int, float)):
        if not math.isfinite(tau_nm):
            raise ValueError("tau_nm must be finite")
        if axis_arr.shape != (3,):
            raise ValueError(
                f"For scalar tau_nm, axis_world must have shape (3,), got {axis_arr.shape}"
            )
        norm = float(np.linalg.norm(axis_arr))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(
                f"axis_world must have unit norm within 1e-6, got norm={norm}"
            )
        torque_vec = float(tau_nm) * axis_arr
    elif isinstance(tau_nm, Sequence) and not isinstance(tau_nm, (str, bytes)):
        tau_arr = np.asarray(tau_nm, dtype=np.float64)
        if tau_arr.ndim != 1:
            raise ValueError("tau_nm sequence must be 1-dimensional")
        if not np.all(np.isfinite(tau_arr)):
            raise ValueError("tau_nm must be finite")
        if (
            axis_arr.ndim != 2
            or axis_arr.shape[1] != 3
            or axis_arr.shape[0] != tau_arr.shape[0]
        ):
            raise ValueError(
                f"axis_world shape {axis_arr.shape} must match tau_nm length {tau_arr.shape[0]}"
            )
        for ax in axis_arr:
            norm = float(np.linalg.norm(ax))
            if abs(norm - 1.0) > 1e-6:
                raise ValueError(
                    f"axis_world axes must have unit norm within 1e-6, got norm={norm}"
                )
        torque_vec = np.sum(tau_arr[:, np.newaxis] * axis_arr, axis=0)
    else:
        raise TypeError(f"tau_nm must be float or Sequence[float], got {type(tau_nm)}")

    torque_tuple = (float(torque_vec[0]), float(torque_vec[1]), float(torque_vec[2]))
    return OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label=label,
        body=body,
        point_m=p_anchor,
        force_n=None,
        torque_nm=torque_tuple,
        source=source,
    )


def _validate_rotation_matrix(rotation_world_from_local: ArrayLike) -> np.ndarray:
    """Validate that the input is an orthonormal 3x3 rotation matrix with det +1."""
    R = np.asarray(rotation_world_from_local, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(
            f"rotation_world_from_local must have shape (3, 3), got {R.shape}"
        )
    if not np.all(np.isfinite(R)):
        raise ValueError("rotation_world_from_local must be finite")

    ortho_diff = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if ortho_diff > 1e-9:
        raise ValueError(
            f"rotation_world_from_local must be orthonormal within 1e-9 (max dev={ortho_diff})"
        )

    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-9:
        raise ValueError(
            f"rotation_world_from_local must have determinant +1 within 1e-9, got {det}"
        )
    return R


def world_wrench_from_local(
    label: str,
    body: str,
    kind: WrenchKind,
    force_local: Sequence[float] | None,
    torque_local: Sequence[float] | None,
    rotation_world_from_local: ArrayLike,
    point_world: Sequence[float],
    source: str,
) -> OverlayWrench:
    """Transform local-frame force and torque halves into world coordinates.

    Parameters
    ----------
    label : str
        Canonical wrench label.
    body : str
        Target body name.
    kind : WrenchKind
        Categorical wrench origin.
    force_local : Sequence[float] | None
        Local force vector [N] or None if unavailable.
    torque_local : Sequence[float] | None
        Local torque vector [N*m] or None if unavailable.
    rotation_world_from_local : ArrayLike
        3x3 rotation matrix mapping local coordinates to world coordinates (world = R @ local).
        Must be orthonormal with determinant +1 within 1e-9.
    point_world : Sequence[float]
        Point of application in world coordinates [m].
    source : str
        Source identifier.

    Returns
    -------
    OverlayWrench
        World-aligned wrench.
    """
    R = _validate_rotation_matrix(rotation_world_from_local)
    p_world = validate_vec3(point_world, "point_world")

    if force_local is None and torque_local is None:
        raise ValueError("At least one of force_local or torque_local must be provided")

    fl = (
        validate_vec3(force_local, "force_local")
        if force_local is not None
        else (0.0, 0.0, 0.0)
    )
    tl = (
        validate_vec3(torque_local, "torque_local")
        if torque_local is not None
        else (0.0, 0.0, 0.0)
    )

    dummy = SpatialWrench(
        application_frame="local",
        point_m=(0.0, 0.0, 0.0),
        force_n=fl,
        torque_nm=tl,
    )
    sw = transform_wrench(
        dummy,
        target_frame="world",
        new_point_m=(0.0, 0.0, 0.0),
        rotation_matrix=R,
    )

    return OverlayWrench(
        kind=kind,
        label=label,
        body=body,
        point_m=p_world,
        force_n=sw.force_n if force_local is not None else None,
        torque_nm=sw.torque_nm if torque_local is not None else None,
        source=source,
    )


def move_wrench_point(
    wrench: OverlayWrench, new_point_m: Sequence[float]
) -> OverlayWrench:
    """Move a wrench to a new application point with moment-arm adjustment.

    Computes:
        tau_B = tau_A + (p_A - p_B) x F

    If force_n is None, the torque about a new point cannot be known without
    the force, so torque_nm would be None. Since OverlayWrench requires at least
    one half to be present, attempting to move a wrench with force_n=None to a
    different point raises ValueError.

    If torque_nm is None and force_n is known, the moved torque remains None
    (never assumed zero).

    Parameters
    ----------
    wrench : OverlayWrench
        Input wrench.
    new_point_m : Sequence[float]
        New application point coordinates [m].

    Returns
    -------
    OverlayWrench
        Equivalent wrench applied at new_point_m.

    Raises
    ------
    ValueError
        If force_n is None and new_point_m differs from wrench.point_m.
    """
    p_B = validate_vec3(new_point_m, "new_point_m")
    if wrench.point_m == p_B:
        return wrench

    if wrench.force_n is None:
        raise ValueError(
            "Cannot move wrench to a new application point when force_n is None: "
            "the resulting torque is unknown (torque_nm is None), leaving neither force nor torque available."
        )

    if wrench.torque_nm is None:
        new_torque = None
    else:
        p_A = np.asarray(wrench.point_m, dtype=np.float64)
        p_B_arr = np.asarray(p_B, dtype=np.float64)
        r = p_A - p_B_arr
        F = np.asarray(wrench.force_n, dtype=np.float64)
        tau_A = np.asarray(wrench.torque_nm, dtype=np.float64)
        tau_B = tau_A + np.cross(r, F)
        new_torque = (float(tau_B[0]), float(tau_B[1]), float(tau_B[2]))

    return OverlayWrench(
        kind=wrench.kind,
        label=wrench.label,
        body=wrench.body,
        point_m=p_B,
        force_n=wrench.force_n,
        torque_nm=new_torque,
        source=wrench.source,
    )


@dataclass(frozen=True)
class SegmentAxis:
    """Segment endpoint geometry used for axial load projection."""

    segment: str
    joint_label: str
    proximal_m: tuple[float, float, float]
    distal_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.segment, str) or not self.segment.strip():
            raise ValueError("segment must be a non-empty string")
        if not isinstance(self.joint_label, str) or not self.joint_label.strip():
            raise ValueError("joint_label must be a non-empty string")

        p = validate_vec3(self.proximal_m, "proximal_m")
        d = validate_vec3(self.distal_m, "distal_m")
        object.__setattr__(self, "proximal_m", p)
        object.__setattr__(self, "distal_m", d)

        axis = (d[0] - p[0], d[1] - p[1], d[2] - p[2])
        length_sq = axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2
        if math.isclose(length_sq, 0.0, abs_tol=1e-12):
            raise ValueError("proximal_m and distal_m cannot be coincident")


def axial_loads_from_reactions(
    frame: ForceTorqueFrame,
    axes: Sequence[SegmentAxis],
    source: str,
) -> AxialLoadFrame:
    """Compute segment axial loads (tension positive) from JOINT_REACTION wrenches.

    Parameters
    ----------
    frame : ForceTorqueFrame
        Instantaneous force/torque frame containing JOINT_REACTION wrenches.
    axes : Sequence[SegmentAxis]
        Segment endpoint configurations mapping each segment to its proximal reaction joint.
    source : str
        Source identifier.

    Returns
    -------
    AxialLoadFrame
        Frame containing per-segment axial loads in Newtons (tension > 0, compression < 0).
    """
    reactions: dict[str, OverlayWrench] = {}
    for w in frame.wrenches:
        if w.kind == WrenchKind.JOINT_REACTION:
            reactions[w.label] = w

    values_n: dict[str, float | None] = {}
    for axis in axes:
        w = reactions.get(axis.joint_label)
        if w is not None and w.force_n is not None:
            val = axial_force_from_proximal_reaction(
                w.force_n, axis.proximal_m, axis.distal_m
            )
            values_n[axis.segment] = val
        else:
            values_n[axis.segment] = None

    return AxialLoadFrame(time_s=frame.time_s, values_n=values_n, source=source)


def frame_with_axial_loads(
    frame: ForceTorqueFrame,
    axes: Sequence[SegmentAxis],
    source: str,
) -> ForceTorqueFrame:
    """Return a copy of frame with its axial_loads field populated from reaction wrenches."""
    axial = axial_loads_from_reactions(frame, axes, source)
    return ForceTorqueFrame(
        time_s=frame.time_s,
        engine=frame.engine,
        wrenches=frame.wrenches,
        axial_loads=axial,
        world_frame=frame.world_frame,
        units=frame.units,
    )
