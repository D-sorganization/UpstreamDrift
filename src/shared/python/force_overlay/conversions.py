"""Shared force and torque conversions across physics engines (ADR-0052, #11287).

Single home for:
- Converting 1-DOF or multi-axis joint torques to 3D world moment vectors.
- Rotating local-frame reactions or wrenches to world frame.
- Moving point of application with moment-arm cross product.
- Projecting joint reaction forces to tension/compression axial loads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    axis_world: Sequence[float] | Sequence[Sequence[float]],
    anchor_world: Sequence[float],
    source: str,
) -> OverlayWrench:
    """Construct a torque-only joint actuator wrench expressed in world coordinates.

    Parameters:
        label: Standard wrench label (e.g. 'joint:knee_z').
        body: Actuated body name.
        tau_nm: Scalar torque or sequence of torques for multi-axis joints.
        axis_world: Corresponding unit joint axis/axes in world coordinates.
        anchor_world: World position of joint anchor.
        source: Producing engine or telemetry source.

    Returns:
        OverlayWrench with kind=JOINT_ACTUATOR, force_n=None, and torque_nm=sum(tau_i * axis_i).
    """
    anchor = validate_vec3(anchor_world, "anchor_world")

    # Normalize scalar vs vector inputs
    if isinstance(tau_nm, (int, float)):
        taus = [float(tau_nm)]
        axes = [validate_vec3(cast(Sequence[float], axis_world), "axis_world")]
    else:
        taus = [float(t) for t in tau_nm]
        axes = [
            validate_vec3(cast(Sequence[float], ax), f"axis_world[{i}]")
            for i, ax in enumerate(axis_world)
        ]

    if len(taus) != len(axes):
        raise ValueError(
            f"tau_nm and axis_world dimension mismatch: {len(taus)} torques vs {len(axes)} axes"
        )

    torque_accum = np.zeros(3, dtype=np.float64)
    for tau, ax in zip(taus, axes, strict=True):
        ax_arr = np.asarray(ax, dtype=np.float64)
        norm = float(np.linalg.norm(ax_arr))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError(f"axis_world must have unit norm within 1e-6, got {norm}")
        torque_accum += tau * ax_arr

    torque_tuple = (
        float(torque_accum[0]),
        float(torque_accum[1]),
        float(torque_accum[2]),
    )
    return OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label=label,
        body=body,
        point_m=anchor,
        force_n=None,
        torque_nm=torque_tuple,
        source=source,
    )


def world_wrench_from_local(
    label: str,
    body: str,
    kind: WrenchKind | str,
    force_local: Sequence[float] | None,
    torque_local: Sequence[float] | None,
    rotation_world_from_local: ArrayLike,
    point_world: Sequence[float],
    source: str,
) -> OverlayWrench:
    """Transform a local-frame force and/or torque to world coordinates.

    Parameters:
        label: Wrench label.
        body: Target body.
        kind: Wrench category.
        force_local: Force in local frame, or None if unavailable.
        torque_local: Torque in local frame, or None if unavailable.
        rotation_world_from_local: 3x3 orthonormal rotation matrix R (world = R @ local).
        point_world: Point of application in world coordinates.
        source: Origin engine or telemetry source.
    """
    if force_local is None and torque_local is None:
        raise ValueError("At least one of force_local or torque_local must be provided")

    R = np.asarray(rotation_world_from_local, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(
            f"rotation_world_from_local must have shape (3, 3), got {R.shape}"
        )

    # Verify orthonormality: R @ R.T ≈ I and det(R) ≈ 1
    ortho_error = float(np.max(np.abs(R @ R.T - np.eye(3))))
    det_val = float(np.linalg.det(R))
    if ortho_error > 1e-9 or not math.isclose(det_val, 1.0, abs_tol=1e-9):
        raise ValueError(
            f"rotation_world_from_local must be orthonormal with determinant +1 within 1e-9 "
            f"(ortho_error={ortho_error}, det={det_val})"
        )

    pt_w = validate_vec3(point_world, "point_world")

    if force_local is not None and torque_local is not None:
        # Route both halves through canonical transform_wrench
        sw_local = SpatialWrench(
            application_frame="local",
            direction_convention="applied_to_body",
            point_m=(0.0, 0.0, 0.0),
            force_n=validate_vec3(force_local, "force_local"),
            torque_nm=validate_vec3(torque_local, "torque_local"),
        )
        sw_world = transform_wrench(
            wrench=sw_local,
            target_frame="world",
            new_point_m=(0.0, 0.0, 0.0),
            rotation_matrix=R,
        )
        force_w: tuple[float, float, float] | None = sw_world.force_n
        torque_w: tuple[float, float, float] | None = sw_world.torque_nm
    elif force_local is not None:
        f_vec = np.asarray(validate_vec3(force_local, "force_local"), dtype=np.float64)
        f_rot = R @ f_vec
        force_w = (float(f_rot[0]), float(f_rot[1]), float(f_rot[2]))
        torque_w = None
    else:
        assert torque_local is not None
        t_vec = np.asarray(
            validate_vec3(torque_local, "torque_local"), dtype=np.float64
        )
        t_rot = R @ t_vec
        force_w = None
        torque_w = (float(t_rot[0]), float(t_rot[1]), float(t_rot[2]))

    return OverlayWrench(
        kind=kind if isinstance(kind, WrenchKind) else WrenchKind(kind),
        label=label,
        body=body,
        point_m=pt_w,
        force_n=force_w,
        torque_nm=torque_w,
        source=source,
    )


def move_wrench_point(
    wrench: OverlayWrench,
    new_point_m: Sequence[float],
) -> OverlayWrench:
    """Move the point of application of a wrench, updating the moment arm.

    Formula:
        τ_B = τ_A + (p_A - p_B) × F

    Semantics:
        - If force_n is None, torque about a new point cannot be known without the force.
          Raises ValueError.
        - If torque_nm is None but force_n is present, torque remains unknown (None)
          because original torque was unknown. Never assumes zero.
        - If both halves are present, routes through canonical transform_wrench.
    """
    p_B = validate_vec3(new_point_m, "new_point_m")
    if wrench.point_m == p_B:
        return wrench

    if wrench.force_n is None:
        raise ValueError(
            "Cannot move torque-only wrench to a new point: moving requires both halves "
            "(force is needed for the moment arm)"
        )

    if wrench.torque_nm is None:
        return OverlayWrench(
            kind=wrench.kind,
            label=wrench.label,
            body=wrench.body,
            point_m=p_B,
            force_n=wrench.force_n,
            torque_nm=None,
            source=wrench.source,
        )

    sw = wrench.to_spatial_wrench()
    transformed = transform_wrench(
        wrench=sw,
        target_frame=wrench.APPLICATION_FRAME,
        new_point_m=p_B,
    )
    return OverlayWrench(
        kind=wrench.kind,
        label=wrench.label,
        body=wrench.body,
        point_m=p_B,
        force_n=transformed.force_n,
        torque_nm=transformed.torque_nm,
        source=wrench.source,
    )


@dataclass(frozen=True)
class SegmentAxis:
    """Geometric proximal-to-distal section axis of a anatomical segment."""

    segment: str
    joint_label: str
    proximal_m: tuple[float, float, float]
    distal_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.segment or not isinstance(self.segment, str):
            raise ValueError("segment must be a non-empty string")
        if not self.joint_label or not isinstance(self.joint_label, str):
            raise ValueError("joint_label must be a non-empty string")

        p = validate_vec3(self.proximal_m, "proximal_m")
        d = validate_vec3(self.distal_m, "distal_m")
        object.__setattr__(self, "proximal_m", p)
        object.__setattr__(self, "distal_m", d)

        if (
            math.isclose(p[0], d[0], abs_tol=1e-9)
            and math.isclose(p[1], d[1], abs_tol=1e-9)
            and math.isclose(p[2], d[2], abs_tol=1e-9)
        ):
            raise ValueError(
                f"SegmentAxis {self.segment}: proximal and distal endpoints cannot be coincident"
            )


def axial_loads_from_reactions(
    frame: ForceTorqueFrame,
    axes: Sequence[SegmentAxis],
    source: str = "",
) -> AxialLoadFrame:
    """Project joint reaction forces in frame onto anatomical segment axes.

    Computes:
        N = -F_parent_on_segment · unit(proximal → distal)  (tension positive).
    """
    reaction_map: dict[str, tuple[float, float, float]] = {}
    for w in frame.wrenches:
        if w.kind == WrenchKind.JOINT_REACTION and w.force_n is not None:
            reaction_map[w.label] = w.force_n

    values_n: dict[str, float | None] = {}
    for ax in axes:
        force = reaction_map.get(ax.joint_label)
        if force is not None:
            values_n[ax.segment] = axial_force_from_proximal_reaction(
                force=force,
                proximal=ax.proximal_m,
                distal=ax.distal_m,
            )
        else:
            values_n[ax.segment] = None

    return AxialLoadFrame(
        time_s=frame.time_s,
        values_n=values_n,
        source=source or frame.engine,
    )


def frame_with_axial_loads(
    frame: ForceTorqueFrame,
    axes: Sequence[SegmentAxis],
    source: str = "",
) -> ForceTorqueFrame:
    """Return a copy of frame with axial_loads computed from its joint reactions."""
    axial = axial_loads_from_reactions(frame, axes, source)
    return ForceTorqueFrame(
        time_s=frame.time_s,
        engine=frame.engine,
        wrenches=frame.wrenches,
        axial_loads=axial,
        world_frame=frame.world_frame,
        units=frame.units,
    )
