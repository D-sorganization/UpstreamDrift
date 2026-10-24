"""Grip attachment helpers for the generated Drake golf humanoid.

Split out of :mod:`.humanoid_urdf` to keep that module inside the 1200-line
file budget (#13466). Both helpers target the generator's humanoid
(:func:`.humanoid_urdf.render_urdf_string`): ``right_hand`` welded to
``club_shaft`` and a free ``left_hand``. ``pydrake`` is imported lazily.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src.shared.python.biomechanics.grip_wrench import GripAnalysis

if TYPE_CHECKING:  # pragma: no cover - import-time hint only
    from pydrake.multibody.plant import MultibodyPlant

__all__ = [
    "LEFT_HAND_UNAVAILABLE_REASON",
    "add_bilateral_left_hand_bushing",
    "right_hand_grip_analysis",
]


def add_bilateral_left_hand_bushing(
    plant: MultibodyPlant,
    *,
    topology: Any | None = None,
    bushing: Any | None = None,
) -> Any:
    """Attach ``left_hand`` to ``club_shaft`` via ``LinearBushingRollPitchYaw``."""
    from importlib import import_module  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from src.shared.python.grip_contact.contact_topology import (  # noqa: PLC0415
        contact_topology,
    )
    from src.shared.python.grip_contact.parameters import (  # noqa: PLC0415
        BushingParameters,
        default_bushing,
    )
    from src.shared.python.grip_contact.topology_contact import (  # noqa: PLC0415
        hand_x_origin,
    )

    math_mod: Any = import_module("pydrake.math")
    tree_mod: Any = import_module("pydrake.multibody.tree")

    topo = topology if topology is not None else contact_topology()
    b_params = bushing if bushing is not None else default_bushing()
    if not isinstance(b_params, BushingParameters):
        raise TypeError(
            f"bushing must be BushingParameters, got {type(b_params).__name__}"
        )

    dx = hand_x_origin(topo, "lead") - hand_x_origin(topo, "trail")
    club_body = plant.GetBodyByName("club_shaft")
    left_hand_body = plant.GetBodyByName("left_hand")
    right_hand_body = plant.GetBodyByName("right_hand")
    club_lead_tf = math_mod.RigidTransform(np.array([dx, 0.0, 0.10], dtype=float))
    club_trail_tf = math_mod.RigidTransform(np.array([0.0, 0.0, 0.10], dtype=float))
    ident_tf = math_mod.RigidTransform()

    frame_club_lead = plant.AddFrame(
        tree_mod.FixedOffsetFrame(
            "club_lead_grip", club_body.body_frame(), club_lead_tf
        )
    )
    plant.AddFrame(
        tree_mod.FixedOffsetFrame(
            "club_trail_grip", club_body.body_frame(), club_trail_tf
        )
    )
    frame_left_hand = plant.AddFrame(
        tree_mod.FixedOffsetFrame(
            "left_hand_grip", left_hand_body.body_frame(), ident_tf
        )
    )
    plant.AddFrame(
        tree_mod.FixedOffsetFrame(
            "right_hand_grip", right_hand_body.body_frame(), ident_tf
        )
    )
    return plant.AddForceElement(
        tree_mod.LinearBushingRollPitchYaw(
            frame_left_hand,
            frame_club_lead,
            np.asarray(b_params.rotational_stiffness_nm_rad, dtype=float),
            np.asarray(b_params.rotational_damping_nms_rad, dtype=float),
            np.asarray(b_params.translational_stiffness_n_m, dtype=float),
            np.asarray(b_params.translational_damping_ns_m, dtype=float),
        )
    )


LEFT_HAND_UNAVAILABLE_REASON = (
    "left hand unavailable: the URDF welds the club to the right hand only "
    "(DRAKE_PARITY_SPEC section 3.4), so there is no left-hand constraint"
)


def right_hand_grip_analysis(
    plant: MultibodyPlant,
    context: Any,
    vdot: Any,
    *,
    club_body: str = "club_shaft",
) -> GripAnalysis:
    """Right-hand grip wrench ON THE CLUB for the right-hand-weld URDF (GCV-8).

    The only load path into the club is the right-hand weld, so Newton-Euler
    of the club gives that wrench directly (the holding-hand formula with no
    other hand).  It acts at the club body origin, the weld anchor.  The left
    hand is ``None`` with a reason, never zero.

    Args:
        plant: finalized plant holding the humanoid URDF.
        context: plant context at the state of interest.
        vdot: generalised acceleration, shape ``(plant.num_velocities(),)``.
        club_body: name of the club body.

    Postconditions: ``right`` is set, ``left`` is ``None`` and
    ``unavailable_reason`` states why; no net or couple is reported.

    Raises:
        ValueError: on a wrongly shaped ``vdot`` or an unknown club body.
    """
    import numpy as np  # noqa: PLC0415

    from src.engines.physics_engines.drake.python.full_body_model import (  # noqa: PLC0415,E501
        club_newton_euler,
    )
    from src.shared.python.biomechanics.grip_extraction import (  # noqa: PLC0415
        hand_from_arrays,
        holding_hand_wrench,
    )
    from src.shared.python.biomechanics.grip_wrench import (  # noqa: PLC0415
        analyze_grip,
    )

    acceleration = np.asarray(vdot, dtype=float)
    if acceleration.shape != (plant.num_velocities(),):
        raise ValueError(
            f"vdot must have shape ({plant.num_velocities()},), "
            f"got {acceleration.shape}"
        )
    if not plant.HasBodyNamed(club_body):
        raise ValueError(f"club body {club_body!r} not found in the plant")
    body = plant.GetBodyByName(club_body)
    gravity = np.asarray(plant.gravity_field().gravity_vector(), dtype=float)
    club = club_newton_euler(
        plant, context, body, acceleration, gravity, welded=[body.index()]
    )
    point = np.asarray(plant.EvalBodyPoseInWorld(context, body).translation())
    force, torque = holding_hand_wrench(
        closing_force_n=np.zeros(3),
        closing_torque_nm=np.zeros(3),
        closing_point_m=point,
        holding_point_m=point,
        **club,
    )
    right = hand_from_arrays("R", point, force, torque)
    analysis = analyze_grip(
        None,
        right,
        split_method="unavailable",
        metadata={"engine": "drake", "path": "urdf_right_hand_weld"},
    )
    return replace(analysis, unavailable_reason=LEFT_HAND_UNAVAILABLE_REASON)
