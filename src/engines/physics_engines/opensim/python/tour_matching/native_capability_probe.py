"""Bounded native OpenSim capability probe and failure receipts."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from .native_prepared_state import DeclaredColdStart

NATIVE_CAPABILITY_PROTOCOL = "native-dynamics-closed-source-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def _native_runtime_identity(opensim: Any) -> str:
    """Hash the pinned Python runtime and the OpenSim native binaries it loads."""
    package = Path(opensim.__file__).resolve().parent
    binaries = []
    for path in sorted(package.rglob("*")):
        name = path.name.lower()
        suffix = path.suffix.lower()
        is_native_library = (
            suffix in {".dll", ".pyd", ".so", ".dylib"} or ".so." in name
        )
        if path.is_file() and is_native_library:
            binaries.append(
                {
                    "path": path.relative_to(package).as_posix(),
                    "sha256": _sha(path),
                }
            )
    payload = {
        "opensim_version": str(opensim.__version__),
        "python_version": sys.version,
        "platform": platform.platform(),
        "binding_sha256": _sha(Path(opensim.__file__)),
        "native_binaries": binaries,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class NativeProbeFailure(RuntimeError):
    """Bounded, stage-specific native probe failure for a private receipt."""

    def __init__(
        self,
        stage: str,
        exception_type: str,
        message: str,
        window_seconds: tuple[float, float] | None,
        *,
        attempted_window_seconds: tuple[float, float] | None = None,
        last_observed_time_seconds: float | None = None,
        cause_chain: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.exception_type = exception_type
        self.message = message[:240]
        self.window_seconds = window_seconds
        self.attempted_window_seconds = attempted_window_seconds
        self.last_observed_time_seconds = last_observed_time_seconds
        self.cause_chain = tuple(
            (kind, detail[:240]) for kind, detail in cause_chain[:3]
        )

    def private_receipt(
        self, request_sha256: str, source_sha256: str
    ) -> dict[str, Any]:
        return {
            "schema_version": "native-probe-failure-v1",
            "request_sha256": request_sha256,
            "source_sha256": source_sha256,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "message": self.message,
            "observed_window_seconds": self.window_seconds,
            "attempted_window_seconds": self.attempted_window_seconds,
            "last_observed_time_seconds": self.last_observed_time_seconds,
            "reached_time_seconds": (
                None if self.window_seconds is None else self.window_seconds[1]
            ),
            "cause_chain": [
                {"exception_type": kind, "message": detail}
                for kind, detail in self.cause_chain
            ],
        }


def _native_short_capability_probe(
    model: Any,
    state: Any,
    horizon_s: float,
    source_sha256: str,
    runtime_sha256: str,
    declared: DeclaredColdStart | None,
) -> tuple[str, float, float]:
    """Exercise OpenSim's native paths, joints and integrator without feedback."""
    initial_time = float(state.getTime())
    stage: dict[str, Any] = {
        "name": "native-model-initialization",
        "start": initial_time,
        "end": initial_time,
        "end_observed": False,
        "attempted_end": None,
        "last_observed_time": initial_time,
    }
    try:
        return _run_native_short_capability_probe(
            model, state, horizon_s, source_sha256, runtime_sha256, declared, stage
        )
    except (OSError, ValueError, RuntimeError, TypeError, AttributeError) as exc:
        if isinstance(exc, NativeProbeFailure):
            raise
        causes: list[tuple[str, str]] = []
        cause = exc.__cause__ or exc.__context__
        while cause is not None and len(causes) < 3:
            causes.append((type(cause).__name__, str(cause) or type(cause).__name__))
            cause = cause.__cause__ or cause.__context__
        raise NativeProbeFailure(
            str(stage["name"]),
            type(exc).__name__,
            str(exc) or type(exc).__name__,
            (
                (float(stage["start"]), float(stage["end"]))
                if stage["end_observed"]
                else None
            ),
            attempted_window_seconds=(
                None
                if stage["attempted_end"] is None
                else (float(stage["start"]), float(stage["attempted_end"]))
            ),
            last_observed_time_seconds=float(stage["last_observed_time"]),
            cause_chain=tuple(causes),
        ) from exc


def _observe_joint_functions(
    model: Any,
    state: Any,
    osim: Any,
    observations: list[tuple[str, float]],
    joint_evaluations: list[tuple[str, str]],
) -> None:
    """Observe finite transforms from each native joint implementation."""
    joints = model.getJointSet()
    for index in range(joints.getSize()):
        joint = joints.get(index)
        joint_path = str(joint.getAbsolutePathString())
        joint_class = str(joint.getConcreteClassName())
        joint_evaluations.append((joint_path, joint_class))
        parent = joint.getParentFrame().getTransformInGround(state)
        child = joint.getChildFrame().getTransformInGround(state)
        for name, transform in (("parent", parent), ("child", child)):
            translation = transform.p()
            quaternion = osim.Rotation.convertRotationToQuaternion(transform.R())
            values = np.asarray(
                [translation.get(i) for i in range(3)]
                + [quaternion.get(i) for i in range(4)],
                dtype=float,
            )
            if not np.isfinite(values).all():
                raise ValueError("Native joint transform is nonfinite")
            observations.append(
                (
                    f"joint:{joint_path}:{joint_class}:{index}:{name}",
                    float(np.linalg.norm(values)),
                )
            )


def _observe_initial_dynamics(
    model: Any, state: Any, observations: list[tuple[str, float]]
) -> tuple[tuple[str, float], ...]:
    """Realize native derivatives and finite muscle path/force outputs."""
    model.realizeDynamics(state)
    model.realizeAcceleration(state)
    state_names = model.getStateVariableNames()
    initial_derivatives = tuple(
        (
            str(state_names.get(i)),
            float(
                model.getStateVariableDerivativeValue(state, str(state_names.get(i)))
            ),
        )
        for i in range(state_names.getSize())
    )
    if not initial_derivatives or not all(
        math.isfinite(value) for _, value in initial_derivatives
    ):
        raise ValueError("Native initial state derivatives are nonfinite")
    muscles = model.getMuscles()
    for muscle_index in range(muscles.getSize()):
        muscle = muscles.get(muscle_index)
        length = float(muscle.getLength(state))
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Native muscle path length is invalid")
        force = float(muscle.getActuation(state))
        if not math.isfinite(force):
            raise ValueError("Native muscle actuation is nonfinite")
        observations.extend(
            (
                (f"muscle:{muscle_index}:length", length),
                (f"muscle:{muscle_index}:native-actuation", force),
            )
        )
    return initial_derivatives


def _audit_initial_inputs(
    model: Any,
    state: Any,
    declared: DeclaredColdStart | None,
    stage: dict[str, Any],
) -> tuple[
    float,
    tuple[tuple[str, float], ...],
    tuple[float, ...],
    tuple[tuple[str, float], ...],
    dict[str, Any],
    Any,
]:
    """Hash the exact native state, controls, discrete state and constraints."""
    start = float(state.getTime())
    stage["start"] = start
    stage["end"] = start
    stage["end_observed"] = False
    stage["attempted_end"] = None
    stage["last_observed_time"] = start
    stage["name"] = "native-state-control-constraint-audit"
    state_names = model.getStateVariableNames()
    initial_state = tuple(
        (str(state_names.get(i)), float(model.getStateVariableValues(state).get(i)))
        for i in range(state_names.getSize())
    )
    if not initial_state or not all(math.isfinite(value) for _, value in initial_state):
        raise ValueError("Native capability probe initial named state is invalid")
    actuators = model.getActuators()
    initial_controls = model.getControls(state)
    control_values = tuple(
        float(initial_controls.get(i)) for i in range(model.getNumControls())
    )
    if any(not math.isfinite(value) for value in control_values):
        raise ValueError("Native capability probe default controls are nonfinite")
    discrete_names = model.getDiscreteVariableNames()
    initial_discrete = tuple(
        (
            str(discrete_names.get(i)),
            float(model.getDiscreteVariableValue(state, str(discrete_names.get(i)))),
        )
        for i in range(discrete_names.getSize())
    )
    if any(not math.isfinite(value) for _, value in initial_discrete):
        raise ValueError("Native capability probe initial discrete state is nonfinite")
    constraint_payload: dict[str, Any] = {"declared": declared is not None}
    if declared is not None:
        from .native_prepared_state import observe_declared_native_sample

        audit = observe_declared_native_sample(model, state, declared)
        constraint_payload = {
            "declared": True,
            "constraint_enforcement": dict(declared.constraint_enforcement),
            "position_errors": audit.position_errors,
            "velocity_errors": audit.velocity_errors,
            "residual_tolerance": declared.residual_tolerance,
        }
    return (
        start,
        initial_state,
        control_values,
        initial_discrete,
        constraint_payload,
        actuators,
    )


def _integrate_native_window(
    model: Any,
    state: Any,
    horizon_s: float,
    start: float,
    stage: dict[str, Any],
    osim: Any,
) -> tuple[float, Any, Any]:
    """Advance one bounded native interval and retain observed integration time."""
    duration = float(horizon_s)
    step = min(0.001, duration / 10.0)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Native capability probe requires a positive horizon")
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-6)
    manager.initialize(state)
    stage["name"] = "native-short-integration"
    stage["attempted_end"] = start + step
    final_state = manager.integrate(start + step)
    stage["end"] = float(final_state.getTime())
    stage["end_observed"] = True
    stage["last_observed_time"] = float(final_state.getTime())
    return step, final_state, manager


def _observe_final_state(
    model: Any, final_state: Any, muscles: Any, stage: dict[str, Any]
) -> tuple[
    np.ndarray,
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
]:
    """Validate final continuous state, derivatives, forces and discrete values."""
    stage["name"] = "native-final-state-and-discrete-audit"
    model.realizeDynamics(final_state)
    model.realizeAcceleration(final_state)
    state_values = model.getStateVariableValues(final_state)
    state_names = model.getStateVariableNames()
    values = np.asarray(
        [state_values.get(index) for index in range(state_names.getSize())],
        dtype=float,
    )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Native capability probe produced nonfinite state")
    final_derivatives = tuple(
        (
            str(state_names.get(i)),
            float(
                model.getStateVariableDerivativeValue(
                    final_state, str(state_names.get(i))
                )
            ),
        )
        for i in range(state_names.getSize())
    )
    if not all(math.isfinite(value) for _, value in final_derivatives):
        raise ValueError("Native final state derivatives are nonfinite")
    final_forces = tuple(
        (
            str(muscles.get(i).getAbsolutePathString()),
            float(muscles.get(i).getActuation(final_state)),
        )
        for i in range(muscles.getSize())
    )
    if any(not math.isfinite(value) for _, value in final_forces):
        raise ValueError("Native final muscle actuation is nonfinite")
    discrete_names = model.getDiscreteVariableNames()
    discrete_values = []
    for index in range(discrete_names.getSize()):
        name = str(discrete_names.get(index))
        value = float(model.getDiscreteVariableValue(final_state, name))
        if not math.isfinite(value):
            raise ValueError("Native probe discrete state is nonfinite")
        discrete_values.append((name, value))
    return values, final_derivatives, final_forces, tuple(discrete_values)


def _build_probe_payload(
    source_sha256: str,
    runtime_sha256: str,
    start: float,
    step: float,
    reached_time: float,
    initial_state: tuple[tuple[str, float], ...],
    control_values: tuple[float, ...],
    actuators: Any,
    initial_discrete: tuple[tuple[str, float], ...],
    constraint_payload: dict[str, Any],
    initial_derivatives: tuple[tuple[str, float], ...],
    values: np.ndarray,
    final_derivatives: tuple[tuple[str, float], ...],
    final_forces: tuple[tuple[str, float], ...],
    observations: list[tuple[str, float]],
    native_joint_evaluations: list[tuple[str, str]],
    discrete_values: tuple[tuple[str, float], ...],
) -> tuple[str, float]:
    """Bind probe evidence to exact model, runtime, inputs and observed outputs."""
    payload = {
        "probe_protocol": NATIVE_CAPABILITY_PROTOCOL,
        "probe_implementation_sha256": _sha(Path(__file__)),
        "source_sha256": source_sha256,
        "runtime_sha256": runtime_sha256,
        "start_seconds": start,
        "end_seconds": reached_time,
        "step_seconds": step,
        "initial_named_state_sha256": hashlib.sha256(
            json.dumps(initial_state, allow_nan=False).encode()
        ).hexdigest(),
        "initial_native_controls_sha256": hashlib.sha256(
            json.dumps(
                {
                    "control_values": control_values,
                    "actuator_paths": tuple(
                        str(actuators.get(i).getAbsolutePathString())
                        for i in range(actuators.getSize())
                    ),
                },
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "control_protocol": "source-controller-free native zero/default actuator controls",
        "initial_discrete_state_sha256": hashlib.sha256(
            json.dumps(initial_discrete, allow_nan=False).encode()
        ).hexdigest(),
        "declared_constraint_audit_sha256": hashlib.sha256(
            json.dumps(constraint_payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "native_state_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "initial_state_derivatives_sha256": hashlib.sha256(
            json.dumps(initial_derivatives, allow_nan=False).encode()
        ).hexdigest(),
        "final_state_derivatives_sha256": hashlib.sha256(
            json.dumps(final_derivatives, allow_nan=False).encode()
        ).hexdigest(),
        "final_muscle_actuation_sha256": hashlib.sha256(
            json.dumps(final_forces, allow_nan=False).encode()
        ).hexdigest(),
        "mechanics_observation_sha256": hashlib.sha256(
            json.dumps(observations, allow_nan=False).encode()
        ).hexdigest(),
        "native_joint_function_evaluations": native_joint_evaluations,
        "discrete_state_sha256": hashlib.sha256(
            json.dumps(discrete_values, allow_nan=False).encode()
        ).hexdigest(),
    }
    if not math.isclose(reached_time, start + step, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("Native capability probe did not reach its exact end time")
    evidence_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return evidence_sha, reached_time


def _run_native_short_capability_probe(
    model: Any,
    state: Any,
    horizon_s: float,
    source_sha256: str,
    runtime_sha256: str,
    declared: DeclaredColdStart | None,
    stage: dict[str, Any],
) -> tuple[str, float, float]:
    """Run finite native dynamics and source-bound observations for one short window."""
    import opensim as osim

    observations: list[tuple[str, float]] = []
    native_joint_evaluations: list[tuple[str, str]] = []
    model.realizePosition(state)
    stage["name"] = "native-joint-and-function-observation"
    _observe_joint_functions(model, state, osim, observations, native_joint_evaluations)
    stage["name"] = "native-initial-dynamics-observation"
    initial_derivatives = _observe_initial_dynamics(model, state, observations)
    muscles = model.getMuscles()
    (
        start,
        initial_state,
        control_values,
        initial_discrete,
        constraint_payload,
        actuators,
    ) = _audit_initial_inputs(model, state, declared, stage)
    step, final_state, manager = _integrate_native_window(
        model, state, horizon_s, start, stage, osim
    )
    values, final_derivatives, final_forces, discrete_values = _observe_final_state(
        model, final_state, muscles, stage
    )
    reached_time = float(final_state.getTime())
    evidence_sha, reached_time = _build_probe_payload(
        source_sha256,
        runtime_sha256,
        start,
        step,
        reached_time,
        initial_state,
        control_values,
        actuators,
        initial_discrete,
        constraint_payload,
        initial_derivatives,
        values,
        final_derivatives,
        final_forces,
        observations,
        native_joint_evaluations,
        discrete_values,
    )
    del manager
    return evidence_sha, start, reached_time
