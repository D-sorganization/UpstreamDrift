"""MuJoCo swing source — narrow facade extracting clubhead kinematics.

Implements the engine side of issue #8975 (EPIC #8965 / WS2): a small,
Law-of-Demeter-friendly API that runs a scripted golf swing under **full
MuJoCo forward dynamics** (``mj_step`` on the in-repo upper-body golf-swing
MJCF, driven by a smooth half-sine torque pulse on its joint motors) and
reads real clubhead kinematics out of the simulation state.

Nothing here knows about the swing→flight pipeline; the
``MuJoCoSwingStateProvider`` in :mod:`swing_state_providers` adapts the
:class:`ClubheadKinematics` produced here into a ``SwingState``.

Sourcing method (honest labeling)
---------------------------------
The swing is *scripted* (an open-loop torque profile, not a biomechanical
controller), but the resulting motion is genuine MuJoCo forward dynamics:
joint torques → mj_step → clubhead velocity/angular velocity/orientation
read from ``mjData``.  Metadata labels this ``mujoco_forward_dynamics``.

Design-by-Contract
------------------
* ``extract_clubhead_state`` requires a valid body name and returns finite,
  correctly-shaped vectors with a unit face normal.
* ``run_reference_swing`` requires a positive target speed and reports the
  achieved speed + residual honestly in its result — it never fabricates a
  match to the request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.contracts import ensure, require

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mujoco

#: Body name of the clubhead in the in-repo golf-swing MJCF models.
DEFAULT_CLUBHEAD_BODY = "clubhead"

#: Local-frame outward clubface normal of the ``clubhead`` body: the
#: ``clubface`` geom sits at +y in the body frame (see
#: ``_golf_swing_upper_body_xml.py``).
_LOCAL_FACE_NORMAL = np.array([0.0, 1.0, 0.0])

#: Duration of the scripted torque pulse [s].
_SWING_DURATION_S = 0.5

#: Torque-scale search bounds (fraction of each actuator's ctrlrange max).
_MIN_TORQUE_SCALE = 0.02
_MAX_TORQUE_SCALE = 1.0

#: Reference torque scale used for the golden-fixture recording.
REFERENCE_TORQUE_SCALE = 0.15

#: Relative tolerance at which the speed calibration stops early.
_CALIBRATION_RTOL = 0.02

_CALIBRATION_MAX_ITER = 12

MODEL_NAME = "upper_body_golf_swing"

#: Import path of the MJCF asset (module attribute holding the XML string).
MODEL_ASSET_MODULE = "src.engines.physics_engines.mujoco._golf_swing_upper_body_xml"
MODEL_ASSET_ATTRIBUTE = "UPPER_BODY_GOLF_SWING_XML"


@dataclass(frozen=True)
class ClubheadKinematics:
    """World-frame clubhead state read from a MuJoCo simulation.

    Attributes:
        velocity:          Linear velocity [m/s], shape (3,).
        angular_velocity:  Angular velocity [rad/s], shape (3,).
        orientation_quat:  Body orientation quaternion (w, x, y, z), shape (4,).
        face_normal:       Outward clubface unit normal, shape (3,).
        mass:              Clubhead body mass [kg].
        inertia_diagonal:  Principal body inertia [kg·m²], shape (3,).
        sim_time:          Simulation time of the sample [s].
    """

    velocity: np.ndarray
    angular_velocity: np.ndarray
    orientation_quat: np.ndarray
    face_normal: np.ndarray
    mass: float
    inertia_diagonal: np.ndarray
    sim_time: float

    @property
    def speed(self) -> float:
        """Clubhead speed [m/s]."""
        return float(np.linalg.norm(self.velocity))


def load_golf_swing_model() -> mujoco.MjModel:
    """Load the in-repo upper-body golf-swing MJCF as an ``MjModel``.

    Postcondition: the model contains a body named
    :data:`DEFAULT_CLUBHEAD_BODY`.
    """
    import importlib

    import mujoco

    xml = getattr(importlib.import_module(MODEL_ASSET_MODULE), MODEL_ASSET_ATTRIBUTE)
    model = mujoco.MjModel.from_xml_string(xml)
    ensure(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, DEFAULT_CLUBHEAD_BODY) >= 0,
        f"golf-swing model must contain a '{DEFAULT_CLUBHEAD_BODY}' body",
    )
    return model


def extract_clubhead_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str = DEFAULT_CLUBHEAD_BODY,
) -> ClubheadKinematics:
    """Read world-frame clubhead kinematics from a MuJoCo state.

    Args:
        model:     The MuJoCo model.
        data:      Simulation state (must be internally consistent, i.e. after
                   ``mj_step`` or ``mj_forward``).
        body_name: Name of the clubhead body.

    Returns:
        A fully-populated :class:`ClubheadKinematics`.

    Preconditions:  ``body_name`` exists in ``model``.
    Postconditions: all vectors finite; ``face_normal`` is a unit vector.
    """
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    require(body_id >= 0, f"body '{body_name}' not found in MuJoCo model", body_name)

    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0
    )  # flg_local=0 → world frame; [0:3] angular, [3:6] linear
    quat = np.array(data.xquat[body_id], dtype=float)
    face_normal = np.zeros(3)
    mujoco.mju_rotVecQuat(face_normal, _LOCAL_FACE_NORMAL, quat)

    state = ClubheadKinematics(
        velocity=vel6[3:6].copy(),
        angular_velocity=vel6[0:3].copy(),
        orientation_quat=quat,
        face_normal=face_normal,
        mass=float(model.body_mass[body_id]),
        inertia_diagonal=np.array(model.body_inertia[body_id], dtype=float),
        sim_time=float(data.time),
    )
    ensure(
        bool(
            np.isfinite(state.velocity).all()
            and np.isfinite(state.angular_velocity).all()
            and np.isfinite(state.orientation_quat).all()
        ),
        "extracted clubhead kinematics must be finite",
    )
    ensure(
        math.isfinite(float(np.linalg.norm(face_normal)))
        and abs(float(np.linalg.norm(face_normal)) - 1.0) < 1e-6,
        "clubface normal must be a unit vector",
        face_normal,
    )
    return state


def run_scripted_swing(
    torque_scale: float,
    duration_s: float = _SWING_DURATION_S,
    model: mujoco.MjModel | None = None,
) -> ClubheadKinematics:
    """Run one scripted swing under full forward dynamics; return peak state.

    Every actuator is driven with a half-sine torque pulse
    ``ctrl = ctrlrange_max * torque_scale * sin(pi * t / duration)`` and the
    clubhead state at the timestep of **peak clubhead speed** is returned.

    Args:
        torque_scale: Fraction of each actuator's upper ctrlrange in (0, 1].
        duration_s:   Length of the torque pulse / simulation [s].
        model:        Optional pre-loaded model (reused across calibration
                      runs); loaded fresh when None.

    Preconditions:  ``0 < torque_scale <= 1``; ``duration_s > 0``.
    Postcondition:  returned kinematics are finite (contract in
                    :func:`extract_clubhead_state`).
    """
    import mujoco

    require(
        0.0 < torque_scale <= _MAX_TORQUE_SCALE,
        "torque_scale must be in (0, 1]",
        torque_scale,
    )
    require(duration_s > 0.0, "duration_s must be > 0", duration_s)

    if model is None:
        model = load_golf_swing_model()
    data = mujoco.MjData(model)
    ctrl_max = model.actuator_ctrlrange[:, 1]
    n_steps = int(duration_s / model.opt.timestep)

    best: ClubheadKinematics | None = None
    for step in range(n_steps):
        t = step * model.opt.timestep
        data.ctrl[:] = ctrl_max * torque_scale * math.sin(math.pi * t / duration_s)
        mujoco.mj_step(model, data)
        state = extract_clubhead_state(model, data)
        if best is None or state.speed > best.speed:
            best = state
    ensure(best is not None, "swing simulation produced no steps", n_steps)
    assert best is not None  # narrow type after ensure
    return best


def run_reference_swing(
    target_speed_ms: float,
) -> tuple[ClubheadKinematics, dict[str, Any]]:
    """Run swings, calibrating torque scale toward ``target_speed_ms``.

    Peak clubhead speed grows monotonically with torque scale over the
    supported range, so a bisection over ``torque_scale`` converges on the
    requested speed.  The achieved speed and residual are reported honestly
    in the metadata — never fabricated to match the request.

    Args:
        target_speed_ms: Desired clubhead speed at impact [m/s], > 0.

    Returns:
        ``(kinematics, metadata)`` where metadata records the model name,
        sourcing method, timestep, torque scale, target/achieved speed and
        relative residual.
    """
    require(
        math.isfinite(target_speed_ms) and target_speed_ms > 0.0,
        "target_speed_ms must be finite and > 0",
        target_speed_ms,
    )
    model = load_golf_swing_model()
    lo, hi = _MIN_TORQUE_SCALE, _MAX_TORQUE_SCALE
    scale = REFERENCE_TORQUE_SCALE
    best = run_scripted_swing(scale, model=model)
    for _ in range(_CALIBRATION_MAX_ITER):
        residual = (best.speed - target_speed_ms) / target_speed_ms
        if abs(residual) <= _CALIBRATION_RTOL:
            break
        if best.speed > target_speed_ms:
            hi = scale
        else:
            lo = scale
        scale = 0.5 * (lo + hi)
        best = run_scripted_swing(scale, model=model)

    metadata: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_asset": f"{MODEL_ASSET_MODULE}.{MODEL_ASSET_ATTRIBUTE}",
        "method": "mujoco_forward_dynamics",
        "control": "scripted half-sine torque pulse (open loop)",
        "timestep_s": float(model.opt.timestep),
        "swing_duration_s": _SWING_DURATION_S,
        "torque_scale": scale,
        "target_speed_ms": target_speed_ms,
        "achieved_speed_ms": best.speed,
        "speed_residual_rel": (best.speed - target_speed_ms) / target_speed_ms,
        "peak_time_s": best.sim_time,
    }
    return best, metadata
