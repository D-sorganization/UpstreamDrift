"""Verify recorded motor-stage forces against freshly built MuJoCo components.

This is a force re-evaluation at recorded numerical stage states, not a forward
re-simulation, a physiological actuator model, or an independent runtime audit.
Trusted hashes must be retained separately from the potentially untrusted data.
"""

from dataclasses import dataclass
import hashlib

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from .mujoco_actuator_probe import ControlledMujocoDrive, MotorRolloutPlant
from .stage_numeric_archive import MOTOR_SEMANTICS, decode_numeric


@dataclass(frozen=True)
class MotorStageForceCheck:
    """Narrow force-only result; no claim about trajectory or runtime independence."""

    stages: int
    max_abs_error: float
    independent_runtime_verified: bool = False
    forward_replay_verified: bool = False
    physical_feasibility_verified: bool = False


def verify_motor_stage_forces(
    archive: bytes,
    spec_bytes: bytes,
    motor_xml: bytes,
    *,
    expected_archive_sha256: str,
    expected_model_sha256: str,
    expected_motor_sha256: str,
    atol: float,
) -> MotorStageForceCheck:
    """Re-evaluate motor inputs at each recorded numerical state, failing closed."""
    for value, expected in (
        (archive, expected_archive_sha256),
        (spec_bytes, expected_model_sha256),
        (motor_xml, expected_motor_sha256),
    ):
        if (
            not isinstance(value, bytes)
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
            or hashlib.sha256(value).hexdigest() != expected
        ):
            raise ValueError("trusted hash mismatch")
    if (
        isinstance(atol, bool)
        or not isinstance(atol, (int, float, np.floating))
        or not np.isfinite(atol)
        or atol < 0
    ):
        raise ValueError("atol must be finite and nonnegative")
    trace = decode_numeric(archive, expected_semantics=MOTOR_SEMANTICS)
    if trace.plan is None or trace.plan.project:
        raise ValueError("only nonprojected motor recordings are supported")
    model = NativeMujocoFullBodyModel(spec_bytes)
    if tuple(model.coordinate_order) != trace.plan.coordinate_order:
        raise ValueError("model coordinate order mismatch")
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
    max_error = 0.0
    count = 0
    for step in trace.steps:
        for segment in step.segments:
            if segment.force_active:
                raise ValueError("external force or impact is unsupported")
        for stage in step.stages:
            _, observed, external = plant.acceleration_with_drive(
                stage.q, stage.v, stage.tau
            )
            if stage.applied_drive is None or stage.applied_external is None:
                raise ValueError("recorded force is unavailable")
            if not np.array_equal(external, stage.applied_external):
                raise ValueError("recorded external force differs")
            error = float(np.max(np.abs(observed - stage.applied_drive)))
            if not np.isfinite(error) or error > atol:
                raise ValueError("recorded motor force differs")
            max_error = max(max_error, error)
            count += 1
    return MotorStageForceCheck(count, max_error)
