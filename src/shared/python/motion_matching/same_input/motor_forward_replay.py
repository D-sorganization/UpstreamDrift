"""Restricted hash-bound motor forward replay; not full runtime or physical proof.

Trusted hashes must be stored outside the untrusted archive. This independent
numerical rerun supports only stateless fixed-gain control, rigid scalar
coordinates, optional closure projection, and a prescribed latched external
force. Contact is not verified. The measured runtime covers a declared subset;
it is not an environment lock.
"""

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.impact_optimization.option_b_prototype import (
    FixedGainController,
    SplineResidual,
)
from src.shared.python.motion_matching.club_shaft_chain import passive_coordinates
from . import motor_stage_verification, mujoco_actuator_probe, stage_numeric_archive
from .integrator import StepPolicy, integrate
from src.shared.python.motion_matching.impact_force import ImpactForce
from .mujoco_actuator_probe import ControlledMujocoDrive, MotorRolloutPlant
from .solution_bundle_mujoco_replay import measured_runtime_fingerprint
from .solution_bundle_prototype import _unique_object
from .stage_numeric_archive import MOTOR_SEMANTICS, decode_numeric, encode_numeric
from .stage_trace import StageRecorder


@dataclass(frozen=True)
class MotorForwardCheck:
    stages: int
    forward_replay_verified: bool = True
    independent_runtime_verified: bool = False
    physical_feasibility_verified: bool = False
    contact_verified: bool = False
    event: tuple[float, np.ndarray, np.ndarray] | None = None


def measured_motor_runtime() -> str:
    """Hash relevant local source and base fingerprint, not a full runtime lock."""
    parts = [measured_runtime_fingerprint()]
    for module in (
        motor_stage_verification,
        mujoco_actuator_probe,
        stage_numeric_archive,
    ):
        if module.__file__ is None:
            raise ValueError("runtime source file is unavailable")
        parts.append(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    parts.append(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def verify_motor_forward(
    archive: bytes,
    spec_bytes: bytes,
    motor_xml: bytes,
    controller_json: bytes,
    *,
    expected_archive_sha256: str,
    expected_model_sha256: str,
    expected_motor_sha256: str,
    expected_controller_sha256: str,
    expected_runtime_sha256: str,
) -> MotorForwardCheck:
    """Freshly integrate fixed controller and compare every saved numerical stage exactly."""
    for name, value, expected in (
        ("archive", archive, expected_archive_sha256),
        ("model", spec_bytes, expected_model_sha256),
        ("motor", motor_xml, expected_motor_sha256),
        ("controller", controller_json, expected_controller_sha256),
    ):
        if (
            not isinstance(value, bytes)
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
            or hashlib.sha256(value).hexdigest() != expected
        ):
            raise ValueError(f"{name} hash mismatch")
    if measured_motor_runtime() != expected_runtime_sha256:
        raise ValueError("runtime fingerprint mismatch")
    trace = decode_numeric(archive, expected_semantics=MOTOR_SEMANTICS)
    plan = trace.plan
    if plan is None:
        raise ValueError("stage plan is missing")
    try:
        c = json.loads(controller_json, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid controller JSON") from exc
    keys = {
        "q0",
        "v0",
        "times_s",
        "q_ref",
        "qd_ref",
        "tau_ff_ref",
        "kp",
        "kd",
        "lower",
        "upper",
        "breakpoints_s",
        "coefficients",
        "scale",
        "trust_radius",
        "min_spacing_s",
        "kkt_regularization",
    }
    if not isinstance(c, dict) or set(c) not in (keys, keys | {"impact"}):
        raise ValueError("invalid controller contract")
    regularization = c["kkt_regularization"]
    if (
        isinstance(regularization, bool)
        or not isinstance(regularization, (int, float))
        or not np.isfinite(regularization)
        or regularization < 0
    ):
        raise ValueError("invalid KKT regularization")
    spline = SplineResidual(
        breakpoints_s=c["breakpoints_s"],
        coefficients_nm=c["coefficients"],
        scale_nm=c["scale"],
        trust_radius=c["trust_radius"],
        min_spacing_s=c["min_spacing_s"],
    )
    controller = FixedGainController(
        times_s=c["times_s"],
        q_ref=c["q_ref"],
        qd_ref=c["qd_ref"],
        tau_ff_ref=c["tau_ff_ref"],
        kp=c["kp"],
        kd=c["kd"],
        lower_nm=c["lower"],
        upper_nm=c["upper"],
        residual=spline,
    )
    model = NativeMujocoFullBodyModel(spec_bytes)
    model.kkt_regularization = float(regularization)
    if tuple(model.coordinate_order) != plan.coordinate_order:
        raise ValueError("model coordinate order mismatch")
    if passive_coordinates(json.loads(spec_bytes)):
        raise ValueError("passive shaft coordinates are unsupported")
    try:
        drive = ControlledMujocoDrive(model, motor_xml.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("motor XML must be UTF-8") from exc
    dofs = drive.model.jnt_dofadr[drive.model.actuator_trnid[:, 0]]
    indices = []
    for dof in dofs:
        matches = np.flatnonzero(model._order_dofs == dof)
        if len(matches) != 1:
            raise ValueError("motor joint missing from coordinate order")
        indices.append(int(matches[0]))
    plant = MotorRolloutPlant(drive, np.asarray(indices, dtype=np.int64))
    impact = None
    if "impact" in c:
        event = c["impact"]
        if (
            not isinstance(event, dict)
            or set(event)
            != {
                "status",
                "frame",
                "point_in_frame_m",
                "axis_in_frame",
                "time_s",
                "duration_s",
                "force_world_n",
                "contact_verified",
            }
            or event["status"] != "recorded_latched_force"
            or event["contact_verified"] is not False
        ):
            raise ValueError("invalid prescribed-force contract")
        if event["frame"] not in plant.kinematic_frames(
            np.asarray(c["q0"], dtype=float)
        ):
            raise ValueError("impact frame is unavailable")
        impact = ImpactForce(
            frame=event["frame"],
            point_in_frame=np.asarray(event["point_in_frame_m"], dtype=float),
            axis_in_frame=np.asarray(event["axis_in_frame"], dtype=float),
            t_start_s=event["time_s"],
            duration_s=event["duration_s"],
            swing_span_s=(0.0, plan.requested_steps * plan.dt_s),
            force_world=np.asarray(event["force_world_n"], dtype=float),
        )
    if any(s.force_active for step in trace.steps for s in step.segments) != (
        impact is not None
    ):
        raise ValueError("impact and recorded force segments mismatch")
    events: list[tuple[float, np.ndarray, np.ndarray]] = []

    def observe_event(t: float, q: np.ndarray, v: np.ndarray) -> None:
        events.append((float(t), q.copy(), v.copy()))

    recorder = StageRecorder(capture_motor_drive=True)
    result = integrate(
        plant,
        np.asarray(c["q0"], dtype=float),
        np.asarray(c["v0"], dtype=float),
        lambda _i, t, q, v: controller.evaluate(t, q, v).commanded_nm,
        steps=plan.requested_steps,
        dt_s=plan.dt_s,
        policy=StepPolicy(
            substeps=plan.substeps, project=plan.project, effort_holds=plan.effort_holds
        ),
        recorder=recorder,
        impact=impact,
        impact_event_observer=observe_event if impact is not None else None,
    )
    if result.failure is not None:
        raise ValueError("motor forward integration failed")
    if len(events) != (0 if impact is None else 1):
        raise ValueError("prescribed event not observed exactly once")
    with (
        np.load(BytesIO(archive), allow_pickle=False) as recorded,
        np.load(BytesIO(encode_numeric(recorder)), allow_pickle=False) as actual,
    ):
        if recorded.files != actual.files or not all(
            np.array_equal(recorded[k], actual[k]) for k in recorded.files
        ):
            raise ValueError("motor forward stages differ")
    return MotorForwardCheck(
        sum(len(s.stages) for s in recorder.steps), event=events[0] if events else None
    )
