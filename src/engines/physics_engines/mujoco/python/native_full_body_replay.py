"""Frozen generalized-effort replay through the full-body MuJoCo owner.

This provider uses the repository's spec-built full-body model, shared
zero-order-hold RK4 integrator, and model-owned contact/weld dynamics. It does
not call ``mj_step`` and does not qualify the model or trajectory.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.same_input import (
    ROOT_COORDINATES,
    StepPolicy,
    VectorPlant,
    integrate,
    open_loop,
)

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

Array: TypeAlias = NDArray[np.float64]
PROVIDER_ID = "mujoco-full-body-generalized-rk4"
PROVIDER_VERSION = "1.0.0"
_SUBSTEPS = 8
DUAL_GRIP_WELD_CLOSURE_DIMENSION = 6
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DeclaredFullBodyFeedbackPolicy:
    """Caller-declared controller source and frozen input provenance."""

    name: str
    version: str
    source_path: Path
    source_sha256: str
    configuration_sha256: str
    reference_sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.name, self.version)
        ):
            raise ValueError("feedback policy name and version must be nonempty")
        for value in (
            self.source_sha256,
            self.configuration_sha256,
            self.reference_sha256,
        ):
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError("feedback policy digests must be lowercase SHA-256")
        object.__setattr__(
            self, "source_path", Path(self.source_path).resolve(strict=True)
        )


@dataclass(frozen=True)
class NativeFullBodyFeedbackRequest:
    """Complete source, initial-state, clock and T01 identity for recording."""

    spec_path: Path
    q0: Array
    v0: Array
    steps: int
    dt_s: float
    controller_policy: DeclaredFullBodyFeedbackPolicy
    model_id: str
    variant_id: str
    experiment_id: str


@dataclass(frozen=True)
class _FeedbackRunContext:
    request: NativeFullBodyFeedbackRequest
    path: Path
    spec_bytes: bytes
    coordinate_order: tuple[str, ...]
    q_initial: Array
    v_initial: Array
    function: Any
    implementation: tuple[Path, Any]
    root_indices: NDArray[np.intp]
    controller_digest: str


@dataclass(frozen=True)
class NativeFullBodyFeedbackRecording:
    """Feedback-produced T01 inputs and their unqualified producer evidence."""

    bundle: Any
    controller_policy: DeclaredFullBodyFeedbackPolicy
    time_seconds: Array
    qpos: Array
    qvel: Array
    applied_generalized_efforts: Array
    pose_closure_residuals: Array
    initial_pose_closure_residual: Array
    initial_rate_closure_residual: Array
    controller_sha256: str
    applied_effort_sha256: str
    trajectory_sha256: str
    qualification_status: str = "unqualified"


@dataclass(frozen=True)
class NativeFullBodyReplay:
    """Unqualified full-body trajectory and observed closure residuals."""

    time_seconds: Array
    qpos: Array
    qvel: Array
    applied_generalized_efforts: Array
    generalized_efforts: Array
    initial_pose_closure_residual: Array
    initial_rate_closure_residual: Array
    pose_closure_residuals: Array
    input_sha256: str
    policy_sha256: str


def validate_generalized_effort_history(
    coordinate_order: tuple[str, ...] | list[str],
    time_seconds: Array,
    generalized_efforts: Array,
) -> float:
    """Validate a full ordered effort table and its terminal ZOH sentinel."""
    names = tuple(coordinate_order)
    times = np.asarray(time_seconds, dtype=np.float64)
    values = np.asarray(generalized_efforts, dtype=np.float64)
    if names[: len(ROOT_COORDINATES)] != ROOT_COORDINATES:
        raise ValueError("coordinate order must begin with the six root coordinates")
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError("time grid must contain finite start and terminal samples")
    intervals = np.diff(times)
    if times[0] != 0.0 or np.any(intervals <= 0.0):
        raise ValueError("time grid must start at zero and increase strictly")
    from src.engines.native_replay_contracts import native_clock_interval_matches

    if not all(
        native_clock_interval_matches(float(interval), float(intervals[0]))
        for interval in intervals
    ):
        raise ValueError("time grid must be fixed-step")
    if values.shape != (times.size, len(names)) or not np.isfinite(values).all():
        raise ValueError("generalized efforts must cover every named coordinate")
    if np.any(values[:, : len(ROOT_COORDINATES)] != 0.0):
        raise ValueError("root generalized efforts must be zero")
    if not np.array_equal(values[-1], values[-2]):
        raise ValueError("terminal ZOH sentinel must equal the last applied effort")
    return float(intervals[0])


def _source_coordinate_effort_units(spec_bytes: bytes) -> tuple[tuple[str, str], ...]:
    """Resolve every declared source primitive to its generalized-effort unit."""
    spec = json.loads(spec_bytes)
    names = tuple(spec["coordinate_order"])
    primitive_pairs = [
        (item["coordinate"], item["primitive"])
        for joint in spec["joints"]
        for item in joint["primitives"]
    ]
    if (
        len(primitive_pairs) != len(names)
        or len({name for name, _ in primitive_pairs}) != len(names)
        or {name for name, _ in primitive_pairs} != set(names)
    ):
        raise ValueError("every source coordinate needs one primitive mapping")
    primitive_units = {
        "Px": "N",
        "Py": "N",
        "Pz": "N",
        "Rx": "N*m",
        "Ry": "N*m",
        "Rz": "N*m",
    }
    by_name = dict(primitive_pairs)
    if any(primitive not in primitive_units for primitive in by_name.values()):
        raise ValueError("source primitive must be a translational or rotational axis")
    return tuple((name, primitive_units[by_name[name]]) for name in names)


def _validate_full_body_input_contract(
    bundle: ExperimentReplayBundle, spec_bytes: bytes, contracts: Any
) -> None:
    """Require frozen generalized-effort semantics and exact source units."""
    if (
        bundle.input_history.input_kind
        is not contracts.ActuationInputKind.GENERALIZED_EFFORT
    ):
        raise ValueError("full-body replay requires generalized-effort input")
    if (
        bundle.input_history.interpolation
        is not contracts.InputInterpolation.ZERO_ORDER_HOLD
    ):
        raise ValueError("full-body replay requires zero-order-held input")
    if bundle.input_history.timebase_id != "simulation_relative":
        raise ValueError("full-body replay requires simulation_relative time")
    expected = tuple(
        contracts.InputChannel(name, name, unit, name)
        for name, unit in _source_coordinate_effort_units(spec_bytes)
    )
    if tuple(bundle.input_history.channels) != expected:
        raise ValueError("input channels differ from exact source primitive mapping")


def integrate_full_body_history(
    plant: VectorPlant,
    q0: Array,
    v0: Array,
    time_seconds: Array,
    generalized_efforts: Array,
    *,
    substeps: int = 8,
    input_sha256: str = "",
    policy_sha256: str = "",
) -> NativeFullBodyReplay:
    """Replay a complete state and fixed generalized-effort history.

    The initial closure residual is observed through the plant's public API.
    No assembly, projection, reset, controller, or state-feedback callback is
    run. The residuals are reported as diagnostics without a qualification
    threshold.
    """
    dt_s = validate_generalized_effort_history(
        plant.coordinate_order, time_seconds, generalized_efforts
    )
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    q, v = _complete_initial_state(q0, v0, len(plant.coordinate_order))
    initial_pose, initial_rate = _initial_closure_observation(plant, q, v)
    rollout = open_loop(
        plant,
        q,
        v,
        np.asarray(generalized_efforts, dtype=np.float64)[:-1],
        dt_s=dt_s,
        policy=StepPolicy(substeps=substeps, project=False),
    )
    expected_times = np.asarray(time_seconds, dtype=np.float64)
    if not np.array_equal(rollout.time_s, expected_times):
        raise ValueError("integrator output does not cover the frozen time grid")
    if (
        rollout.q.shape != (expected_times.size, len(plant.coordinate_order))
        or rollout.v.shape != rollout.q.shape
        or not np.isfinite(rollout.q).all()
        or not np.isfinite(rollout.v).all()
    ):
        raise ValueError("integrator output contains incomplete or nonfinite q/v state")
    expected_efforts = np.asarray(generalized_efforts, dtype=np.float64)[:-1]
    if (
        rollout.efforts.shape != expected_efforts.shape
        or not np.isfinite(rollout.efforts).all()
        or not np.array_equal(rollout.efforts, expected_efforts)
    ):
        raise ValueError(
            "integrator output differs from applied frozen generalized effort"
        )
    if (
        rollout.pose_drift.shape != (expected_times.size - 1,)
        or not np.isfinite(rollout.pose_drift).all()
    ):
        raise ValueError("integrator output contains incomplete closure magnitudes")
    return NativeFullBodyReplay(
        np.asarray(time_seconds, dtype=np.float64).copy(),
        rollout.q,
        rollout.v,
        rollout.efforts,
        rollout.efforts,
        np.asarray(initial_pose, dtype=np.float64),
        np.asarray(initial_rate, dtype=np.float64),
        rollout.pose_drift,
        input_sha256,
        policy_sha256,
    )


def _immutable_array(values: Array) -> Array:
    copied = np.asarray(values, dtype=np.float64)
    return np.frombuffer(copied.tobytes(), dtype=np.float64).reshape(copied.shape)


def _complete_initial_state(q0: Array, v0: Array, nv: int) -> tuple[Array, Array]:
    q = np.asarray(q0, dtype=np.float64).copy()
    v = np.asarray(v0, dtype=np.float64).copy()
    expected_shape = (nv,)
    if q.shape != expected_shape or v.shape != expected_shape:
        raise ValueError("complete named position and velocity state is required")
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("initial position and velocity state must be finite")
    return q, v


def _initial_closure_observation(
    plant: VectorPlant, q: Array, v: Array
) -> tuple[Array, Array]:
    pose, rate = plant.closure_residuals(q, v)
    pose = np.asarray(pose, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    if pose.shape != (DUAL_GRIP_WELD_CLOSURE_DIMENSION,) or rate.shape != (
        DUAL_GRIP_WELD_CLOSURE_DIMENSION,
    ):
        raise ValueError(
            "initial dual-grip weld closure observations must have six components"
        )
    if not np.isfinite(pose).all() or not np.isfinite(rate).all():
        raise ValueError("initial closure observation must be finite")
    return pose.copy(), rate.copy()


def _feedback_callable_source(controller: Callable[..., Any]) -> tuple[Any, Any]:
    function = (
        controller if inspect.isfunction(controller) else type(controller).__call__
    )
    source_path = inspect.getsourcefile(function)
    code = getattr(function, "__code__", None)
    if source_path is None or code is None:
        raise ValueError("feedback controller must have inspectable Python source")
    return function, (Path(source_path).resolve(strict=True), code)


def _verify_feedback_callable(
    controller: Callable[..., Any],
    policy: DeclaredFullBodyFeedbackPolicy,
    function: Any,
    implementation: tuple[Path, Any],
) -> None:
    current_function, current_implementation = _feedback_callable_source(controller)
    if current_function is not function or current_implementation != implementation:
        raise ValueError("feedback controller implementation changed during recording")
    source_path, _code = implementation
    if source_path != policy.source_path:
        raise ValueError("feedback callable differs from declared source file")
    if _sha256(source_path.read_bytes()) != policy.source_sha256:
        raise ValueError("feedback controller source differs from declared digest")


def _array_sha256(values: Array) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def record_native_full_body_feedback(
    request: NativeFullBodyFeedbackRequest,
    controller: Callable[[float, Array, Array], Array],
) -> NativeFullBodyFeedbackRecording:
    """Record one state-dependent generalized effort per fixed interval.

    The feedback callback sees detached copies of the current state.
    Its validated output is held by the shared RK4 integrator with the same
    eight substeps used by fresh full-body replay. This records an unqualified
    source run and creates the ordinary T01 frozen-input bundle; replay never
    invokes the callback.
    """
    context = _prepare_feedback_context(request, controller)
    rollout, applied_history, initial_pose, initial_rate = _integrate_feedback(
        context, controller
    )
    return _build_feedback_recording(
        context, rollout, applied_history, initial_pose, initial_rate
    )


def _prepare_feedback_context(
    request: NativeFullBodyFeedbackRequest,
    controller: Callable[[float, Array, Array], Array],
) -> _FeedbackRunContext:
    if type(request.steps) is not int or request.steps <= 0:
        raise ValueError("feedback step budget must be a positive integer")
    if not np.isfinite(request.dt_s) or request.dt_s <= 0.0:
        raise ValueError("feedback timestep must be finite and positive")
    if not callable(controller) or not isinstance(
        request.controller_policy, DeclaredFullBodyFeedbackPolicy
    ):
        raise TypeError("feedback requires a callable and declared policy provenance")

    path = Path(request.spec_path)
    spec_bytes = path.read_bytes()
    coordinate_order = tuple(json.loads(spec_bytes)["coordinate_order"])
    if coordinate_order[: len(ROOT_COORDINATES)] != ROOT_COORDINATES:
        raise ValueError(
            "source coordinate order must begin with the six root coordinates"
        )
    _source_coordinate_effort_units(spec_bytes)
    nv = len(coordinate_order)
    q_initial, v_initial = _complete_initial_state(request.q0, request.v0, nv)

    function, implementation = _feedback_callable_source(controller)
    policy = request.controller_policy
    _verify_feedback_callable(controller, policy, function, implementation)
    controller_digest = _controller_policy_sha256(policy)
    roots = np.asarray(
        [coordinate_order.index(name) for name in ROOT_COORDINATES], dtype=np.intp
    )
    return _FeedbackRunContext(
        request,
        path,
        spec_bytes,
        coordinate_order,
        q_initial,
        v_initial,
        function,
        implementation,
        roots,
        controller_digest,
    )


def _controller_policy_sha256(policy: DeclaredFullBodyFeedbackPolicy) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "name": policy.name,
                "version": policy.version,
                "source_path": str(policy.source_path),
                "source_sha256": policy.source_sha256,
                "configuration_sha256": policy.configuration_sha256,
                "reference_sha256": policy.reference_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _integrate_feedback(
    context: _FeedbackRunContext,
    controller: Callable[[float, Array, Array], Array],
) -> tuple[Any, Array, Array, Array]:
    request = context.request
    policy = request.controller_policy
    plant = VectorPlant("mujoco", context.spec_bytes, kkt_regularization=0.0)
    initial_pose, initial_rate = _initial_closure_observation(
        plant, context.q_initial, context.v_initial
    )
    applied: list[Array] = []

    def effort_source(index: int, time_s: float, q: Array, v: Array) -> Array:
        if index != len(applied) or not np.isfinite(time_s):
            raise ValueError("feedback integrator interval order is invalid")
        _verify_feedback_callable(
            controller, policy, context.function, context.implementation
        )
        q_snapshot = np.asarray(q, dtype=np.float64).copy()
        v_snapshot = np.asarray(v, dtype=np.float64).copy()
        # Copy immediately because conversion can invoke caller-defined __array__.
        effort = np.asarray(
            controller(time_s, q_snapshot, v_snapshot), dtype=np.float64
        ).copy()
        _verify_feedback_callable(
            controller, policy, context.function, context.implementation
        )
        if (
            effort.shape != (len(context.coordinate_order),)
            or not np.isfinite(effort).all()
        ):
            raise ValueError("feedback effort must be a finite full-coordinate vector")
        if np.any(effort[context.root_indices] != 0.0):
            raise ValueError("feedback root generalized efforts must be exactly zero")
        applied.append(effort.copy())
        return effort

    rollout = integrate(
        plant,
        context.q_initial,
        context.v_initial,
        effort_source,
        steps=request.steps,
        dt_s=float(request.dt_s),
        policy=StepPolicy(substeps=_SUBSTEPS, project=False, stop_on_failure=False),
    )
    if len(applied) != request.steps or rollout.efforts.shape != (
        request.steps,
        len(context.coordinate_order),
    ):
        raise ValueError("feedback recorder did not cover every requested interval")
    applied_history = np.stack(applied)
    if not np.array_equal(rollout.efforts, applied_history):
        raise ValueError("integrator applied efforts differ from feedback snapshots")
    if (
        rollout.q.shape != (request.steps + 1, len(context.coordinate_order))
        or rollout.v.shape != (request.steps + 1, len(context.coordinate_order))
        or not np.isfinite(rollout.q).all()
        or not np.isfinite(rollout.v).all()
        or rollout.pose_drift.shape != (request.steps,)
        or not np.isfinite(rollout.pose_drift).all()
    ):
        raise ValueError("feedback integration produced incomplete or nonfinite state")
    if context.path.read_bytes() != context.spec_bytes:
        raise ValueError("source spec changed during feedback recording")
    return rollout, applied_history, initial_pose, initial_rate


def _build_feedback_recording(
    context: _FeedbackRunContext,
    rollout: Any,
    applied_history: Array,
    initial_pose: Array,
    initial_rate: Array,
) -> NativeFullBodyFeedbackRecording:
    request = context.request
    policy = request.controller_policy
    steps = request.steps
    times = np.arange(steps + 1, dtype=np.float64) * float(request.dt_s)
    frozen_efforts = np.vstack((applied_history, applied_history[-1]))
    bundle = build_native_full_body_bundle(
        context.path,
        context.q_initial,
        context.v_initial,
        times,
        frozen_efforts,
        model_id=request.model_id,
        variant_id=request.variant_id,
        experiment_id=request.experiment_id,
    )
    from src.engines.native_replay_contracts import (
        native_replay_contract_types,
        validate_native_replay_bundle,
    )

    contracts = native_replay_contract_types()
    bundle = validate_native_replay_bundle(bundle, contracts)
    _validate_full_body_input_contract(bundle, context.spec_bytes, contracts)
    if bundle.model.source_model_sha256 != _sha256(context.spec_bytes):
        raise ValueError("T01 bundle source identity differs from feedback source")
    bundled_values = np.asarray(bundle.input_history.values, dtype=np.float64)
    if not np.array_equal(bundled_values, frozen_efforts):
        raise ValueError("T01 bundle differs from the recorded applied effort history")

    return NativeFullBodyFeedbackRecording(
        bundle,
        policy,
        _immutable_array(times),
        _immutable_array(rollout.q),
        _immutable_array(rollout.v),
        _immutable_array(applied_history),
        _immutable_array(rollout.pose_drift),
        _immutable_array(initial_pose),
        _immutable_array(initial_rate),
        context.controller_digest,
        _array_sha256(applied_history),
        hashlib.sha256(
            (_array_sha256(rollout.q) + _array_sha256(rollout.v)).encode("ascii")
        ).hexdigest(),
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _provider_sha256() -> str:
    """Bind the provider, dynamics, integrator and actual MuJoCo extension."""
    import mujoco
    import mujoco._functions

    from src.engines.native_replay_contracts import native_replay_admission_bytes
    from src.engines.physics_engines.mujoco.python import (
        full_body_mjcf,
        full_body_model,
        native_model,
    )
    from src.shared.python.motion_matching import contact_law
    from src.shared.python.motion_matching.same_input import integrator, plant

    files = (
        Path(__file__),
        Path(full_body_mjcf.__file__),
        Path(full_body_model.__file__),
        Path(native_model.__file__),
        Path(contact_law.__file__),
        Path(plant.__file__),
        Path(integrator.__file__),
    )
    digest = hashlib.sha256()
    digest.update(PROVIDER_VERSION.encode())
    digest.update(str(mujoco.__version__).encode())
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    extension_path = Path(mujoco._functions.__file__)
    digest.update(extension_path.read_bytes())
    digest.update(native_replay_admission_bytes())
    return digest.hexdigest()


def _identity(
    spec_bytes: bytes,
    model_id: str,
    variant_id: str,
    time_step_s: float,
    contracts: Any,
) -> tuple[Any, Any]:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    spec = json.loads(spec_bytes)
    coordinate_order = tuple(spec["coordinate_order"])
    if coordinate_order[: len(ROOT_COORDINATES)] != ROOT_COORDINATES:
        raise ValueError(
            "source coordinate order must begin with the six root coordinates"
        )
    generated_xml, metadata = export_full_body_mjcf(spec_bytes)
    source_sha = _sha256(spec_bytes)
    generated_sha = _sha256(generated_xml.encode("utf-8"))
    if (
        metadata.get("model_sha256") != source_sha
        or metadata.get("mjcf_sha256") != generated_sha
    ):
        raise ValueError("full-body generator metadata differs from exact source bytes")
    provider_sha = _provider_sha256()
    state_schema = contracts.InitialStateSchema(
        "mujoco-full-body-qv",
        PROVIDER_VERSION,
        (
            contracts.StateComponentSpec(
                "qpos",
                contracts.StateComponentRole.POSITION,
                len(coordinate_order),
                "SI",
                "complete-spec-coordinate-order",
            ),
            contracts.StateComponentSpec(
                "qvel",
                contracts.StateComponentRole.VELOCITY,
                len(coordinate_order),
                "SI",
                "complete-spec-coordinate-order",
            ),
        ),
    )
    model = contracts.ModelIdentity(
        "mujoco",
        model_id,
        variant_id,
        PROVIDER_VERSION,
        source_sha,
        PROVIDER_ID,
        PROVIDER_VERSION,
        provider_sha,
        state_schema,
        coordinate_order,
        generated_sha,
    )
    from src.engines.physics_engines.mujoco.python import (
        full_body_model,
        native_model,
    )
    from src.shared.python.motion_matching import contact_law

    contact_digest = hashlib.sha256(b"full-body-own-contact-and-weld-v1\0")
    contact_digest.update(spec_bytes)
    for source in (full_body_model, native_model, contact_law):
        source_path = source.__file__
        if not source_path:
            raise RuntimeError("a contact-policy source module has no file path")
        contact_digest.update(Path(source_path).read_bytes())
    contact_sha = contact_digest.hexdigest()
    policy = contracts.ReplayExecutionPolicy(
        replay_mode=contracts.ReplayMode.NATIVE_OWN_CONTACT,
        solver_id="full-body-weld-kkt",
        solver_version=PROVIDER_VERSION,
        integration_method=(
            f"python-classical-rk4-{_SUBSTEPS}-substeps-over-engine-accelerations"
        ),
        step_policy="fixed",
        step_size_seconds=time_step_s,
        initialization_policy_id="complete-qv-no-assembly-or-reset",
        initialization_policy_version=PROVIDER_VERSION,
        input_player_id="coordinate-generalized-effort-zoh",
        input_player_version=PROVIDER_VERSION,
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="spec-owned-ground-contact-and-dual-grip-weld",
        contact_policy_version=PROVIDER_VERSION,
        contact_policy_sha256=contact_sha,
    )
    return model, policy


def build_native_full_body_bundle(
    spec_path: str | Path,
    q0: Array,
    v0: Array,
    time_seconds: Array,
    generalized_efforts: Array,
    *,
    model_id: str,
    variant_id: str,
    experiment_id: str,
) -> ExperimentReplayBundle:
    """Bind exact source spec, complete q/v state, and frozen generalized efforts."""
    from src.engines.native_replay_contracts import native_replay_contract_types

    contracts = native_replay_contract_types()
    path = Path(spec_path)
    spec_bytes = path.read_bytes()
    times = np.asarray(time_seconds, dtype=np.float64)
    efforts = np.asarray(generalized_efforts, dtype=np.float64)
    coordinate_order = tuple(json.loads(spec_bytes)["coordinate_order"])
    step_s = validate_generalized_effort_history(coordinate_order, times, efforts)
    model, policy = _identity(spec_bytes, model_id, variant_id, step_s, contracts)
    names = tuple(model.ordered_input_channel_ids)
    source_units = dict(_source_coordinate_effort_units(spec_bytes))
    if tuple(source_units) != names:
        raise ValueError(
            "source effort-unit mapping differs from model coordinate order"
        )
    channels = tuple(
        contracts.InputChannel(name, name, source_units[name], name) for name in names
    )
    return contracts.build_experiment_replay_bundle(
        experiment_id,
        model,
        (
            contracts.CapabilityDeclaration(
                "full-body-generalized-effort-rk4",
                True,
                contracts.CapabilitySupport.SUPPORTED,
                contracts.CapabilityAvailability.AVAILABLE,
            ),
        ),
        (
            ("qpos", tuple(np.asarray(q0, dtype=np.float64))),
            (
                "qvel",
                tuple(np.asarray(v0, dtype=np.float64)),
            ),
        ),
        channels,
        contracts.ActuationInputKind.GENERALIZED_EFFORT,
        contracts.InputInterpolation.ZERO_ORDER_HOLD,
        tuple(times),
        tuple(tuple(row) for row in efforts),
        policy,
    )


def replay_native_full_body_bundle(
    bundle: ExperimentReplayBundle,
    spec_path: str | Path,
) -> NativeFullBodyReplay:
    """Revalidate and replay a frozen T01 bundle through a fresh VectorPlant."""
    from src.engines.native_replay_contracts import (
        native_replay_contract_types,
        validate_native_replay_bundle,
    )

    contracts = native_replay_contract_types()
    bundle = validate_native_replay_bundle(bundle, contracts)
    path = Path(spec_path)
    spec_bytes = path.read_bytes()
    source_sha = _sha256(spec_bytes)
    if source_sha != bundle.model.source_model_sha256:
        raise ValueError("full-body source spec differs from the frozen bundle")
    if bundle.model.engine_id != "mujoco" or bundle.model.provider_id != PROVIDER_ID:
        raise ValueError("bundle is not bound to the full-body MuJoCo provider")
    _validate_full_body_input_contract(bundle, spec_bytes, contracts)
    times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    dt_s = validate_generalized_effort_history(
        tuple(json.loads(spec_bytes)["coordinate_order"]),
        times,
        np.asarray(bundle.input_history.values, dtype=np.float64),
    )
    expected_model, expected_policy = _identity(
        spec_bytes,
        bundle.model.model_id,
        bundle.model.variant_id,
        dt_s,
        contracts,
    )
    if bundle.model != expected_model or bundle.policy != expected_policy:
        raise ValueError("source model, provider, contact or execution policy differs")
    if tuple(channel.channel_id for channel in bundle.input_history.channels) != tuple(
        json.loads(spec_bytes)["coordinate_order"]
    ):
        raise ValueError("input channels must match complete spec coordinate order")
    state = {
        item.component_id: np.asarray(item.values, dtype=np.float64)
        for item in bundle.initial_state
    }
    if tuple(state) != ("qpos", "qvel"):
        raise ValueError(
            "complete qpos and qvel state is required without hidden state"
        )
    plant = VectorPlant("mujoco", spec_bytes, kkt_regularization=0.0)
    if tuple(plant.coordinate_order) != tuple(bundle.model.ordered_input_channel_ids):
        raise ValueError("loaded full-body coordinate order differs from T01")
    return integrate_full_body_history(
        plant,
        state["qpos"],
        state["qvel"],
        times,
        np.asarray(bundle.input_history.values, dtype=np.float64),
        substeps=_SUBSTEPS,
        input_sha256=bundle.applied_input_sha256,
        policy_sha256=bundle.policy_sha256,
    )
