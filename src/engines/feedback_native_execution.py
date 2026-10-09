"""Strict execution boundary for frozen Tools experiment/replay bundles.

Receipts prove that an explicitly bound native adapter ran a complete frozen
input history. They do not score observations or qualify model physics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from numpy.typing import NDArray

from src.engines.feedback_comparison import ComparisonRow, DriveMode
from src.engines.model_inventory import TARGET_ENGINES

if TYPE_CHECKING:
    from src.engines.physics_engines.drake.python.native_torque_replay import (
        NativeDrakeTorqueReplay,
    )
    from src.engines.physics_engines.mujoco.python.native_torque_replay import (
        NativeTorqueReplay,
    )


@dataclass(frozen=True)
class NativeAdapterBinding:
    """Explicit bridge from one F01 row to a separately identified T01 adapter."""

    package_id: str
    variant_id: str
    drive_mode: DriveMode
    native_model_id: str
    native_variant_id: str
    native_execution_provider_id: str
    native_execution_provider_sha256: str
    source_model_sha256: str
    loaded_native_model_sha256: str
    state_schema_sha256: str
    input_channel_schema_sha256: str
    ordered_input_channel_ids: tuple[str, ...]


@dataclass(frozen=True)
class NativeReplayRequest:
    """A bundle plus local source path; paths are never copied to receipts."""

    binding: NativeAdapterBinding
    bundle: Any
    model_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))

    def as_dict(self) -> dict[str, str]:
        drive_mode = self.binding.drive_mode
        return {
            "row_key": f"{self.binding.package_id}/{self.binding.variant_id}/{drive_mode.value}",
            "bundle_schema": str(self.bundle.schema_version),
        }


@dataclass(frozen=True)
class NativeReplayExecution:
    """Validated receipt and actual output from one native replay invocation."""

    receipt: NativeExecutionReceipt
    output: NativeTorqueReplay | NativeDrakeTorqueReplay


@dataclass(frozen=True)
class NativeExecutionReceipt:
    """Integrity and run facts without a numerical acceptance verdict."""

    schema_version: str
    package_id: str
    variant_id: str
    drive_mode: str
    engine: str
    required: bool
    support: str
    availability: str
    qualification: str
    source_model_sha256: str
    inventory_provider_id: str
    inventory_provider_sha256: str
    native_model_id: str
    native_variant_id: str
    native_execution_provider_id: str
    native_execution_provider_sha256: str
    loaded_native_model_sha256: str
    state_schema_sha256: str
    initial_state_sha256: str
    input_channel_schema_sha256: str
    applied_input_sha256: str
    policy_sha256: str
    time_grid_sha256: str
    output_state_sha256: str
    channel_ids: tuple[str, ...]
    input_kind: str
    interpolation: str
    timebase_id: str
    evidence_mode: str
    horizon_s: float
    state_sample_count: int
    applied_input_sample_count: int
    nq: int
    nv: int
    full_state: bool
    full_horizon: bool
    state_reset_allowed: bool
    state_reset_count: int | None


@dataclass(frozen=True)
class NativeReplayRowResult:
    package_id: str
    variant_id: str
    drive_mode: str
    engine: str
    required: bool
    support: str
    availability: str
    qualification: str
    status: str
    receipt: NativeExecutionReceipt | None = None
    reason: str = ""


@dataclass(frozen=True)
class NativeReplayReport:
    rows: tuple[NativeReplayRowResult, ...]
    required_engine_ids: tuple[str, ...]

    @property
    def missing_engine_ids(self) -> tuple[str, ...]:
        replayed = {
            row.engine for row in self.rows if row.required and row.status == "replayed"
        }
        return tuple(
            engine for engine in self.required_engine_ids if engine not in replayed
        )

    @property
    def executed_row_count(self) -> int:
        return sum(row.status == "replayed" for row in self.rows)

    @property
    def is_complete(self) -> bool:
        return (
            bool(self.rows)
            and not self.missing_engine_ids
            and all(row.status == "replayed" for row in self.rows if row.required)
        )


def _digest_matches(actual: str, expected: str, name: str) -> None:
    if not expected or actual != expected:
        raise ValueError(f"native {name} identity differs")


def _binding_row(row: ComparisonRow, binding: NativeAdapterBinding) -> None:
    if (row.package_id, row.variant_id, row.drive_mode) != (
        binding.package_id,
        binding.variant_id,
        binding.drive_mode,
    ):
        raise ValueError("binding does not identify the declared inventory row")
    if row.engine not in TARGET_ENGINES:
        raise ValueError("inventory engine is outside the required parity set")
    if row.support != "supported":
        raise ValueError("inventory row does not declare this drive mode supported")
    if row.availability != "available":
        raise ValueError("inventory row is not declared available for execution")
    if not row.provider_id:
        raise ValueError("inventory provider identity is unavailable")
    if len(row.provider_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in row.provider_sha256.lower()
    ):
        raise ValueError("inventory provider SHA-256 identity is invalid")
    for value, name in (
        (binding.native_execution_provider_sha256, "native execution provider"),
        (binding.source_model_sha256, "source model"),
        (binding.loaded_native_model_sha256, "loaded native model"),
        (binding.state_schema_sha256, "state schema"),
        (binding.input_channel_schema_sha256, "input channel schema"),
    ):
        if len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value.lower()
        ):
            raise ValueError(f"{name} SHA-256 identity is invalid")
    if (
        not binding.native_model_id
        or not binding.native_variant_id
        or not binding.native_execution_provider_id
    ):
        raise ValueError("native adapter binding identifiers are required")
    if not binding.ordered_input_channel_ids or len(
        set(binding.ordered_input_channel_ids)
    ) != len(binding.ordered_input_channel_ids):
        raise ValueError("native ordered channel mapping is invalid")


def _validate_bundle(
    row: ComparisonRow, binding: NativeAdapterBinding, bundle: Any
) -> None:
    from sidekick.lab.mocap import ActuationInputKind, InputInterpolation, ReplayMode

    _binding_row(row, binding)
    if bundle.schema_version != "experiment-replay/1.0.0":
        raise ValueError("unsupported frozen replay bundle schema")
    if bundle.blocking_capabilities:
        raise ValueError("required replay capability is unavailable")
    if bundle.policy.replay_mode is not ReplayMode.NATIVE_OWN_CONTACT:
        raise ValueError("native execution requires native own-contact replay policy")
    if bundle.policy.observation_access or bundle.policy.state_feedback_access:
        raise ValueError("observation and state-feedback access are forbidden")
    if bundle.policy.state_reset_allowed:
        raise ValueError("state resets are forbidden by native replay policy")
    if bundle.input_history.input_kind is not ActuationInputKind.ACTUATOR_TORQUE:
        raise ValueError("current native adapters require actuator torque input")
    if bundle.input_history.interpolation is not InputInterpolation.ZERO_ORDER_HOLD:
        raise ValueError("current native adapters require zero-order-held torque")
    if bundle.input_history.timebase_id != "simulation_relative":
        raise ValueError("native replay requires simulation_relative time")
    if row.drive_mode is not DriveMode.TORQUE:
        raise ValueError("actuator torque cannot satisfy a muscle-excitation row")
    model = bundle.model
    if model.engine_id != row.engine:
        raise ValueError("native adapter engine differs from inventory row")
    _digest_matches(model.source_model_sha256, row.source_model_sha256, "source model")
    _digest_matches(
        model.source_model_sha256, binding.source_model_sha256, "bound source model"
    )
    if model.model_id != binding.native_model_id:
        raise ValueError("bundle native model identity differs from explicit binding")
    if model.variant_id != binding.native_variant_id:
        raise ValueError("bundle native variant identity differs from explicit binding")
    if model.provider_id != binding.native_execution_provider_id:
        raise ValueError(
            "bundle native execution provider differs from explicit binding"
        )
    if model.provider_sha256 != binding.native_execution_provider_sha256:
        raise ValueError(
            "bundle native execution provider hash differs from explicit binding"
        )
    _digest_matches(
        model.loaded_native_model_sha256 or "",
        binding.loaded_native_model_sha256,
        "loaded native model",
    )
    _digest_matches(
        bundle.state_schema_sha256, binding.state_schema_sha256, "state schema"
    )
    _digest_matches(
        bundle.input_channel_schema_sha256,
        binding.input_channel_schema_sha256,
        "input channel schema",
    )
    channels = tuple(channel.channel_id for channel in bundle.input_history.channels)
    if (
        channels != binding.ordered_input_channel_ids
        or channels != model.ordered_input_channel_ids
    ):
        raise ValueError("native ordered input channel mapping differs")


def _output_digest(
    output: NativeTorqueReplay | NativeDrakeTorqueReplay,
) -> str:
    digest = hashlib.sha256()
    for name in (
        "time_seconds",
        "qpos",
        "qvel",
        "integration_states",
        "discrete_states",
        "applied_actuator_torques",
        "generalized_actuator_torques",
    ):
        if not hasattr(output, name):
            continue
        array = np.asarray(getattr(output, name), dtype="<f8")
        if not np.isfinite(array).all():
            raise ValueError("native output contains nonfinite evidence")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_native_replay_output(
    row: ComparisonRow,
    binding: NativeAdapterBinding,
    bundle: Any,
    output: NativeTorqueReplay | NativeDrakeTorqueReplay,
) -> NativeExecutionReceipt:
    """Validate the adapter's complete actual output against the frozen bundle."""
    if row.engine == "mujoco":
        from src.engines.physics_engines.mujoco.python.native_torque_replay import (
            NativeTorqueReplay,
        )

        if not isinstance(output, NativeTorqueReplay):
            raise ValueError("MuJoCo native adapter returned an unknown output type")
    elif row.engine == "drake":
        from src.engines.physics_engines.drake.python.native_torque_replay import (
            NativeDrakeTorqueReplay,
        )

        if not isinstance(output, NativeDrakeTorqueReplay):
            raise ValueError("Drake native adapter returned an unknown output type")
    else:
        raise ValueError("no native output contract is registered for this engine")
    _validate_bundle(row, binding, bundle)
    times = np.asarray(output.time_seconds, dtype=np.float64)
    expected_times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    inputs = np.asarray(output.applied_actuator_torques, dtype=np.float64)
    expected_inputs = np.asarray(bundle.input_history.values[:-1], dtype=np.float64)
    qpos = np.asarray(output.qpos, dtype=np.float64)
    qvel = np.asarray(output.qvel, dtype=np.float64)
    if times.shape != expected_times.shape or not np.array_equal(times, expected_times):
        raise ValueError("native output does not cover the exact full time grid")
    if inputs.shape != expected_inputs.shape or not np.array_equal(
        inputs, expected_inputs
    ):
        raise ValueError("native output actual inputs differ from frozen bundle")
    qpos = np.asarray(output.qpos, dtype=np.float64)
    qvel = np.asarray(output.qvel, dtype=np.float64)
    generalized_effort = np.asarray(output.generalized_actuator_torques)
    if generalized_effort.shape != (
        len(inputs),
        qvel.shape[1] if qvel.ndim == 2 else -1,
    ):
        raise ValueError("native generalized actuator effort dimensions differ")
    _validate_native_output_state(bundle, output, times, qpos, qvel)
    if (
        output.input_sha256 != bundle.applied_input_sha256
        or output.policy_sha256 != bundle.policy_sha256
    ):
        raise ValueError("native output input or executed policy digest differs")
    output_sha = _output_digest(output)
    return _native_execution_receipt(
        row, binding, bundle, times, inputs, qpos, qvel, output_sha
    )


def _validate_native_output_state(
    bundle: Any,
    output: NativeTorqueReplay | NativeDrakeTorqueReplay,
    times: NDArray[np.float64],
    qpos: NDArray[np.float64],
    qvel: NDArray[np.float64],
) -> None:
    """Check full physical and numerical state against the frozen bundle."""
    if (
        qpos.ndim != 2
        or qvel.ndim != 2
        or qpos.shape[0] != len(times)
        or qvel.shape[0] != len(times)
    ):
        raise ValueError("native output state does not cover the full horizon")
    if qpos.shape[1] <= 0 or qvel.shape[1] <= 0:
        raise ValueError("native qpos and qvel dimensions must be positive")
    state_component = {item.component_id: item.values for item in bundle.initial_state}
    auxiliary_name = (
        "integration_states"
        if hasattr(output, "integration_states")
        else "discrete_states"
    )
    expected_auxiliary_name = (
        "integration" if auxiliary_name == "integration_states" else "discrete"
    )
    if tuple(state_component) != ("qpos", "qvel", expected_auxiliary_name):
        raise ValueError(
            "native adapter cannot preserve every declared state component"
        )
    if auxiliary_name not in ("integration_states", "discrete_states"):
        raise ValueError("native replay output lacks its complete numerical state")
    auxiliary = np.asarray(getattr(output, auxiliary_name), dtype=np.float64)
    if (
        auxiliary.ndim != 2
        or auxiliary.shape[0] != len(times)
        or auxiliary.shape[1] <= 0
    ):
        raise ValueError("native output lacks full integration initialization state")
    if auxiliary.shape[1] != len(state_component.get(expected_auxiliary_name, ())):
        raise ValueError("native output auxiliary state differs from T01 state schema")
    if not np.array_equal(
        qpos[0], state_component.get("qpos", ())
    ) or not np.array_equal(qvel[0], state_component.get("qvel", ())):
        raise ValueError(
            "native output initial physical state differs from frozen bundle"
        )
    if not np.array_equal(
        auxiliary[0], state_component.get(expected_auxiliary_name, ())
    ):
        raise ValueError(
            "native output initial numerical state differs from frozen bundle"
        )


def _native_execution_receipt(
    row: ComparisonRow,
    binding: NativeAdapterBinding,
    bundle: Any,
    times: NDArray[np.float64],
    inputs: NDArray[np.float64],
    qpos: NDArray[np.float64],
    qvel: NDArray[np.float64],
    output_sha: str,
) -> NativeExecutionReceipt:
    """Build an unqualified integrity receipt from validated native samples."""
    input_history = bundle.input_history
    replay_policy = bundle.policy
    return NativeExecutionReceipt(
        "native-execution/1.0.0",
        row.package_id,
        row.variant_id,
        row.drive_mode.value,
        row.engine,
        row.required,
        row.support,
        row.availability,
        "unqualified",
        row.source_model_sha256,
        row.provider_id,
        row.provider_sha256,
        binding.native_model_id,
        binding.native_variant_id,
        binding.native_execution_provider_id,
        binding.native_execution_provider_sha256,
        binding.loaded_native_model_sha256,
        bundle.state_schema_sha256,
        bundle.integrity.initial_state_sha256,
        bundle.input_channel_schema_sha256,
        bundle.applied_input_sha256,
        bundle.policy_sha256,
        bundle.time_grid_sha256,
        output_sha,
        tuple(binding.ordered_input_channel_ids),
        input_history.input_kind.value,
        input_history.interpolation.value,
        input_history.timebase_id,
        replay_policy.replay_mode.value,
        float(times[-1] - times[0]),
        len(times),
        len(inputs),
        qpos.shape[1],
        qvel.shape[1],
        True,
        True,
        replay_policy.state_reset_allowed,
        None,
    )


def execute_native_replay(
    request: NativeReplayRequest, registry: Any
) -> NativeExecutionReceipt:
    """Execute one frozen bundle through its exact reviewed native adapter."""
    return execute_native_replay_with_output(request, registry).receipt


def execute_native_replay_with_output(
    request: NativeReplayRequest, registry: Any
) -> NativeReplayExecution:
    """Execute once and return both the integrity receipt and native trajectory."""
    row = registry.get(
        request.binding.package_id,
        request.binding.variant_id,
        request.binding.drive_mode,
    )
    _validate_bundle(row, request.binding, request.bundle)
    output: NativeTorqueReplay | NativeDrakeTorqueReplay
    if (
        row.engine == "mujoco"
        and request.binding.native_execution_provider_id
        == "mujoco-native-torque-replay"
    ):
        from src.engines.physics_engines.mujoco.python.native_torque_replay import (
            replay_native_torque_bundle,
        )

        output = replay_native_torque_bundle(request.bundle, request.model_path)
    elif (
        row.engine == "drake"
        and request.binding.native_execution_provider_id == "drake-native-torque-replay"
    ):
        from src.engines.physics_engines.drake.python.native_torque_replay import (
            replay_native_drake_torque_bundle,
        )

        output = replay_native_drake_torque_bundle(request.bundle, request.model_path)
    else:
        raise ValueError("no reviewed native adapter is registered for this row")
    receipt = validate_native_replay_output(
        row, request.binding, request.bundle, output
    )
    return NativeReplayExecution(receipt, output)


def build_native_replay_report(
    registry: Any, requests: Iterable[NativeReplayRequest]
) -> NativeReplayReport:
    """Retain every inventory row; missing or unavailable evidence stays visible."""
    request_items = tuple(requests)
    by_key = {
        (
            request.binding.package_id,
            request.binding.variant_id,
            request.binding.drive_mode,
        ): request
        for request in request_items
    }
    if len(by_key) != len(request_items):
        raise ValueError("duplicate native replay request row")
    inventory_keys = {
        (row.package_id, row.variant_id, row.drive_mode) for row in registry.rows
    }
    if set(by_key) - inventory_keys:
        raise ValueError(
            "native replay request references an unregistered inventory row"
        )
    results: list[NativeReplayRowResult] = []
    for row in registry.rows:
        key = (row.package_id, row.variant_id, row.drive_mode)
        request = by_key.get(key)
        if request is None:
            results.append(
                _row_result(row, "missing_binding", "no explicit adapter binding")
            )
            continue
        if row.engine not in {"mujoco", "drake"}:
            results.append(
                _row_result(row, "unsupported_adapter", "no reviewed native adapter")
            )
            continue
        if row.availability != "available":
            results.append(
                _row_result(
                    row, "unavailable", "inventory availability is not available"
                )
            )
            continue
        try:
            receipt = execute_native_replay(request, registry)
        except (ImportError, OSError) as error:
            results.append(
                _row_result(row, "runtime_unavailable", type(error).__name__)
            )
        except (TypeError, ValueError, RuntimeError) as error:
            results.append(_row_result(row, "rejected", type(error).__name__))
        else:
            results.append(_row_result(row, "replayed", "", receipt))
    return NativeReplayReport(tuple(results), tuple(sorted(TARGET_ENGINES)))


def _row_result(
    row: ComparisonRow,
    status: str,
    reason: str,
    receipt: NativeExecutionReceipt | None = None,
) -> NativeReplayRowResult:
    return NativeReplayRowResult(
        row.package_id,
        row.variant_id,
        row.drive_mode.value,
        row.engine,
        row.required,
        row.support,
        row.availability,
        row.qualification,
        status,
        receipt,
        reason,
    )
