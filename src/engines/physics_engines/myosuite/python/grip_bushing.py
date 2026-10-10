"""MyoSuite bushing grip on the same input as OpenSim (issue #11739, OSV-7).

MyoSuite runs MuJoCo.  This module therefore evaluates exactly the MuJoCo
bushing grip (:mod:`src.engines.physics_engines.mujoco.python.grip_bushing`:
the shared OpenSim ``BushingForce`` law through ``mjcb_passive``, RK4) but
with both MuJoCo models, the spec full-body weld model that supplies the hand
frames and the one-body free club, loaded through MyoSuite's own simulation
layer (``MujocoEnv``: the ``mj_model`` / ``mj_data`` pair MyoSuite holds), the
same route as the same-input full-body parity (``full_body_parity``).  The
MyoHub musculoskeletal body is not used: its topology is unrelated to the
44-coordinate specification (11 of 44 coordinates map), so a torque- or
motion-driven grip comparison is only meaningful on the specification model.

Raises ``ImportError`` when ``myosuite`` is not importable (the grip tests skip).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python import grip_bushing as mujoco_grip
from src.engines.physics_engines.myosuite.python.full_body_parity import (
    load_myosuite_runtime,
)
from src.shared.python.grip_contact import CoordinateSwing, GripInterface
from src.shared.python.grip_contact.parity import BushingProbe, GripKineticsSeries

ENGINE = "myosuite"
DEFAULT_TIMESTEP_S = mujoco_grip.DEFAULT_TIMESTEP_S


def myosuite_loader(xml: str) -> tuple[Any, Any]:
    """Compile ``xml`` through MyoSuite's ``MujocoEnv`` and return its model, data."""
    env = load_myosuite_runtime(xml)
    return env.mj_model, env.mj_data


def simulate_grip_bushing(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    interface: GripInterface | None = None,
    timestep_s: float = DEFAULT_TIMESTEP_S,
    t_end_s: float | None = None,
) -> GripKineticsSeries:
    """The MuJoCo bushing run on MyoSuite-held models (see module docstring)."""
    series = mujoco_grip.simulate_grip_bushing(
        spec_bytes,
        swing,
        interface,
        timestep_s,
        t_end_s,
        loader=myosuite_loader,
        engine=ENGINE,
    )
    meta = dict(series.metadata)
    meta["runtime"] = "MyoSuite MujocoEnv (mj_model / mj_data held by MyoSuite)"
    return GripKineticsSeries(
        engine=series.engine,
        time_s=series.time_s,
        force_on_club_n=series.force_on_club_n,
        torque_on_club_nm=series.torque_on_club_nm,
        grip_point_m=series.grip_point_m,
        deflection_m=series.deflection_m,
        rotation_deflection_rad=series.rotation_deflection_rad,
        club_rotation=series.club_rotation,
        metadata=meta,
    )


def probe_bushing_forces(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    translation_hand_m: np.ndarray,
    interface: GripInterface | None = None,
) -> BushingProbe:
    """``F = K delta`` probe on MyoSuite-held models."""
    return mujoco_grip.probe_bushing_forces(
        spec_bytes, swing, translation_hand_m, interface, loader=myosuite_loader
    )
