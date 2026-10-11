"""Owned-file boundary for trusted R2025b operating-point producers (#11942).

This boundary performs no MAT decoding, native simulation or qualification.
MATLAB requires filenames; temporary files are created from verified immutable
bytes, never by copying or reopening caller source paths. Native consumers must
still verify the actual decoded class, clock, configuration and compatibility.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from src.engines.native_replay_contracts import native_replay_contract_types

if TYPE_CHECKING:
    from sidekick.lab.mocap import (
        NativeStateExecution,
        NativeStateIdentity,
        NativeStateReplayEnvelope,
        OwnedNativeStateArtifact,
    )


@dataclass(frozen=True)
class SimscapeRestartBinding:
    """Expected producer identity, distinct from unverified artifact metadata.

    producer_input_sha256 identifies the capture's input prehistory; replay
    inputs can differ after the saved native time and have separate digests.
    The caller supplies this binding from its reviewed trusted native producer.
    """

    model_name: str
    identity: NativeStateIdentity
    execution: NativeStateExecution
    producer_input_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,62}", self.model_name
        ):
            raise ValueError("native model name must be a bounded MATLAB identifier")
        if not isinstance(self.producer_input_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.producer_input_sha256
        ):
            raise ValueError("producer input digest must be lowercase SHA-256")


def _admit(
    envelope: NativeStateReplayEnvelope,
    owned: OwnedNativeStateArtifact,
    source: bytes,
    binding: SimscapeRestartBinding,
) -> tuple[Any, Any, bytes]:
    c = native_replay_contract_types()
    if not hasattr(c, "NativeStateReplayEnvelope"):
        raise RuntimeError("native replay requires the reviewed Tools T04 dependency")
    serialized = c.dumps_native_state_replay_envelope(envelope)
    validated = c.load_native_state_replay_envelope(serialized)
    model = validated.model
    if (
        binding.model_name != "native_restart_fixture_11921"
        or model.model_id != f"simscape/{binding.model_name}"
        or model.variant_id != "diagnostic"
        or model.model_version != "1.0.0"
    ):
        raise ValueError("unsupported diagnostic model inventory binding")
    if validated.blocking_capabilities:
        raise ValueError("required native replay capabilities are unavailable")
    descriptor = validated.artifact
    clock = descriptor.clock
    if clock.clock_id != "simulation_absolute":
        raise ValueError("native clock must use simulation_absolute")
    if owned.descriptor != descriptor:
        raise ValueError("owned artifact descriptor differs from envelope binding")
    frozen = c.freeze_native_state_artifact(descriptor, owned.payload)
    identity, execution = descriptor.identity, descriptor.execution
    if identity != binding.identity or execution != binding.execution:
        raise ValueError("native producer or execution binding differs")
    if identity.engine_id != "simscape" or "R2025b" not in identity.runtime_id:
        raise ValueError("native provider requires an explicit R2025b binding")
    if hashlib.sha256(source).hexdigest() != identity.source_model_sha256:
        raise ValueError("native model bytes differ from source binding")
    encoding = descriptor.encoding
    if (
        encoding.format_id != "matlab-mat-v7.3"
        or encoding.format_version != "7.3.0"
        or encoding.native_class != "Simulink.op.ModelOperatingPoint"
    ):
        raise ValueError("unsupported native operating-point encoding")
    history, policy = validated.input_history, validated.policy
    if (
        policy.external_loads_sha256
        != hashlib.sha256(_native_input_rows(validated)).hexdigest()
    ):
        raise ValueError("external load identity differs from executed force rows")
    channels = history.channels
    if (
        history.input_kind is not c.ActuationInputKind.ACTUATOR_FORCE
        or history.interpolation is not c.InputInterpolation.ZERO_ORDER_HOLD
        or len(channels) != 1
        or channels[0].target_id != "native_force_input"
        or channels[0].unit != "N"
        or channels[0].frame_id != "fixture-translational-axis"
    ):
        raise ValueError("native adapter requires one held force channel in N")
    if (
        policy.solver_id != "ode23t"
        or policy.integration_method != "ode23t"
        or policy.step_policy != "adaptive"
        or policy.step_size_seconds is not None
        or policy.initialization_policy_id != "simscape-model-operating-point"
        or policy.initialization_policy_version != "1.0.0"
        or policy.input_player_id != "simscape-from-workspace-zoh"
        or policy.input_player_version != "1.0.0"
        or policy.replay_mode is not c.ReplayMode.EXTERNALLY_FORCED
        or policy.contact_policy_id is not None
    ):
        raise ValueError("unsupported native diagnostic replay policy")
    return validated, frozen, serialized.encode("utf-8")


def _native_input_rows(envelope: Any) -> bytes:
    history = envelope.input_history
    return "".join(
        f"{time:.17g},{value[0]:.17g}\n"
        for time, value in zip(
            envelope.native_time_seconds, history.values, strict=True
        )
    ).encode("ascii")


def _write_request(
    directory: Path, envelope: Any, binding: SimscapeRestartBinding, serialized: bytes
) -> None:
    descriptor = envelope.artifact
    clock, execution = descriptor.clock, descriptor.execution
    identity = binding.identity
    integrity = envelope.integrity
    times = envelope.native_time_seconds
    rows = _native_input_rows(envelope)
    (directory / "replay-input.csv").write_bytes(rows)
    (directory / "native-envelope.json").write_bytes(serialized)
    request = {
        "schema_version": "simscape-owned-replay-request/1.0.0",
        "model_name": binding.model_name,
        "model_sha256": identity.source_model_sha256,
        "loaded_model_sha256": identity.loaded_model_sha256,
        "snapshot_sha256": descriptor.payload_sha256,
        "snapshot_time_s": clock.snapshot_time_seconds,
        "start_time_s": clock.start_time_seconds,
        "clock_id": clock.clock_id,
        "stop_time_s": times[-1],
        "runtime_id": identity.runtime_id,
        "provider_sha256": identity.provider_sha256,
        "producer_input_sha256": binding.producer_input_sha256,
        "replay_input_file_sha256": hashlib.sha256(rows).hexdigest(),
        "envelope_sha256": integrity.envelope_sha256,
        "envelope_file_sha256": hashlib.sha256(serialized).hexdigest(),
        "solver_id": execution.solver_id,
        "solver_version": execution.solver_version,
        "effective_configuration_sha256": execution.effective_configuration_sha256,
        "compatibility_sha256": execution.compatibility_sha256,
    }
    (directory / "native-request.json").write_bytes(
        (json.dumps(request, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )


@contextmanager
def owned_simscape_replay_files(
    envelope: NativeStateReplayEnvelope,
    owned: OwnedNativeStateArtifact,
    source_model: bytes | bytearray | memoryview,
    binding: SimscapeRestartBinding,
) -> Iterator[Path]:
    """Admit immutable bytes and clean owned files on success or native failure.

    Only the diagnostic held-force input player is supported. A returned path
    is a scoped resource for a trusted native decoder, not proof that the MAT
    payload is safe to execute or that its metadata describes actual physics.
    """
    if not isinstance(source_model, (bytes, bytearray, memoryview)):
        raise TypeError("source_model must contain owned bytes")
    if not isinstance(binding, SimscapeRestartBinding):
        raise TypeError("binding must be SimscapeRestartBinding")
    source = bytes(source_model)
    validated, frozen, serialized = _admit(envelope, owned, source, binding)
    with TemporaryDirectory(prefix="simscape-native-replay-") as temporary:
        directory = Path(temporary)
        (directory / f"{binding.model_name}.slx").write_bytes(source)
        (directory / "native-operating-point.mat").write_bytes(frozen.payload)
        _write_request(directory, validated, binding, serialized)
        yield directory
