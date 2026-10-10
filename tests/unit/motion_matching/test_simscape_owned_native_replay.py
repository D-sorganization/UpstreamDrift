"""Owned-byte admission tests, deliberately independent of MATLAB availability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.engines.native_replay_contracts import native_replay_contract_types
from src.engines.Simscape_Multibody_Models.python import (
    SimscapeRestartBinding,
    owned_simscape_replay_files,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def declaration() -> tuple[Any, Any, bytearray, SimscapeRestartBinding]:
    """Synthetic opaque bytes test ownership, never native restart validity."""
    c = native_replay_contract_types()
    source = bytearray(b"synthetic model bytes, not an SLX")
    digest = hashlib.sha256(source).hexdigest()
    identity = c.NativeStateIdentity(
        "simscape", "R2025b-test", "1" * 64, digest, digest
    )
    execution = c.NativeStateExecution(
        "ode23t",
        "25.2.0",
        "simscape-model-operating-point",
        "1.0.0",
        "2" * 64,
        "3" * 64,
    )
    artifact = c.NativeStateArtifact(
        c.NativeStateEncoding(
            "matlab-mat-v7.3", "7.3.0", "Simulink.op.ModelOperatingPoint"
        ),
        identity,
        c.NativeStateClock(0.0, 0.2, "simulation_absolute"),
        execution,
        hashlib.sha256(b"opaque test bytes").hexdigest(),
        len(b"opaque test bytes"),
        c.NativeStateRole.COMPLETE_NATIVE_RESTART,
    )
    history = c.InputHistory(
        c.ActuationInputKind.ACTUATOR_FORCE,
        "simulation_relative",
        c.InputInterpolation.ZERO_ORDER_HOLD,
        (0.0, 0.1, 0.2),
        (
            c.InputChannel(
                "force",
                "native_force_input",
                "N",
                frame_id="fixture-translational-axis",
            ),
        ),
        ((0.3,), (0.4,), (0.4,)),
    )
    policy = c.ReplayExecutionPolicy(
        c.ReplayMode.EXTERNALLY_FORCED,
        "ode23t",
        "25.2.0",
        "ode23t",
        "adaptive",
        None,
        "simscape-model-operating-point",
        "1.0.0",
        "simscape-from-workspace-zoh",
        "1.0.0",
        False,
        False,
        False,
        external_loads_sha256=hashlib.sha256(
            "".join(
                f"{artifact.clock.snapshot_time_seconds + time:.17g},{row[0]:.17g}\n"
                for time, row in zip(history.time_seconds, history.values, strict=True)
            ).encode("ascii")
        ).hexdigest(),
    )
    envelope = c.build_native_state_replay_envelope(
        "synthetic-ownership-only",
        c.NativeReplayModel(
            "simscape/native_restart_fixture_11921", "diagnostic", "1.0.0", ("force",)
        ),
        artifact,
        (
            c.CapabilityDeclaration(
                "native_restart",
                True,
                c.CapabilitySupport.SUPPORTED,
                c.CapabilityAvailability.AVAILABLE,
            ),
        ),
        history,
        policy,
    )
    binding = SimscapeRestartBinding(
        "native_restart_fixture_11921", identity, execution, "5" * 64
    )
    return (
        envelope,
        c.freeze_native_state_artifact(artifact, b"opaque test bytes"),
        source,
        binding,
    )


def test_owns_bytes_and_absolute_inputs_without_reopening_source_paths(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
) -> None:
    envelope, owned, source, binding = declaration
    expected = bytes(source)
    with owned_simscape_replay_files(envelope, owned, source, binding) as directory:
        source[:] = b"mutated caller storage"
        assert (directory / "native_restart_fixture_11921.slx").read_bytes() == expected
        assert (directory / "native-operating-point.mat").read_bytes() == owned.payload
        manifest = json.loads(
            (directory / "native-request.json").read_text(encoding="utf-8")
        )
        assert manifest["snapshot_time_s"] == 0.2
        assert manifest["stop_time_s"] == 0.4
        assert manifest["producer_input_sha256"] == "5" * 64
        assert manifest["envelope_sha256"] == envelope.integrity.envelope_sha256
        c = native_replay_contract_types()
        retained = (directory / "native-envelope.json").read_text(encoding="utf-8")
        assert c.load_native_state_replay_envelope(retained) == envelope
        assert (
            manifest["envelope_file_sha256"]
            == hashlib.sha256(retained.encode("utf-8")).hexdigest()
        )
        rows = (directory / "replay-input.csv").read_text().splitlines()
        assert float(rows[0].split(",")[0]) == 0.2
        assert float(rows[-1].split(",")[0]) == 0.4
        root = directory
    assert not root.exists()


def test_rejects_unexecuted_external_load_declaration(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
) -> None:
    envelope, owned, source, binding = declaration
    c = native_replay_contract_types()
    changed = c.build_native_state_replay_envelope(
        envelope.experiment_id,
        envelope.model,
        envelope.artifact,
        envelope.capabilities,
        envelope.input_history,
        replace(envelope.policy, external_loads_sha256="6" * 64),
    )
    with pytest.raises(ValueError, match="external load"):
        with owned_simscape_replay_files(changed, owned, source, binding):
            pytest.fail("unexecuted external-load identity was admitted")


@pytest.mark.parametrize(
    "field,value", [("model_id", "simscape/driver"), ("variant_id", "default")]
)
def test_diagnostic_bytes_cannot_claim_a_production_inventory_row(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
    field: str,
    value: str,
) -> None:
    envelope, owned, source, binding = declaration
    c = native_replay_contract_types()
    changed = c.build_native_state_replay_envelope(
        envelope.experiment_id,
        replace(envelope.model, **{field: value}),
        envelope.artifact,
        envelope.capabilities,
        envelope.input_history,
        envelope.policy,
    )
    with pytest.raises(ValueError, match="diagnostic model"):
        with owned_simscape_replay_files(changed, owned, source, binding):
            pytest.fail("synthetic diagnostic was mislabeled as production")


@pytest.mark.parametrize("field", ["identity", "execution"])
def test_rejects_other_producer_or_execution_before_materializing(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
    field: str,
) -> None:
    envelope, owned, source, binding = declaration
    if field == "identity":
        changed = replace(binding.identity, runtime_id="another runtime")
    else:
        changed = replace(binding.execution, compatibility_sha256="6" * 64)
    with pytest.raises(ValueError, match="binding"):
        with owned_simscape_replay_files(
            envelope, owned, source, replace(binding, **{field: changed})
        ):
            pytest.fail("incompatible native data reached decoder boundary")


def test_rejects_wrong_source_bytes(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
) -> None:
    envelope, owned, _, binding = declaration
    with pytest.raises(ValueError, match="model bytes"):
        with owned_simscape_replay_files(envelope, owned, b"wrong", binding):
            pytest.fail("source mismatch reached decoder")


def test_cleans_owned_directory_after_native_decoder_failure(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
) -> None:
    envelope, owned, source, binding = declaration
    with pytest.raises(RuntimeError, match="native failed"):
        with owned_simscape_replay_files(envelope, owned, source, binding) as directory:
            root = directory
            raise RuntimeError("native failed")
    assert not root.exists()


def test_rejects_model_path_traversal(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
) -> None:
    _, _, _, binding = declaration
    with pytest.raises(ValueError, match="model name"):
        replace(binding, model_name="../outside")


@pytest.mark.parametrize("mutation", ["frame", "policy", "unavailable"])
def test_rejects_unsupported_execution_before_decoder(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
    mutation: str,
) -> None:
    envelope, owned, source, binding = declaration
    c = native_replay_contract_types()
    history, policy, capabilities = (
        envelope.input_history,
        envelope.policy,
        envelope.capabilities,
    )
    if mutation == "frame":
        channel = replace(history.channels[0], frame_id="another-axis")
        history = replace(history, channels=(channel,))
    elif mutation == "policy":
        policy = replace(policy, integration_method="another-method")
    else:
        capabilities = (
            replace(
                capabilities[0],
                availability=c.CapabilityAvailability.UNAVAILABLE,
                reason="required runtime absent",
            ),
        )
    changed = c.build_native_state_replay_envelope(
        envelope.experiment_id,
        envelope.model,
        envelope.artifact,
        capabilities,
        history,
        policy,
    )
    with pytest.raises(ValueError, match="force channel|policy|unavailable"):
        with owned_simscape_replay_files(changed, owned, source, binding):
            pytest.fail("unsupported declaration reached native decoder")


def test_revalidates_tampered_owned_payload_before_decoder(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
) -> None:
    envelope, owned, source, binding = declaration
    object.__setattr__(owned, "payload", b"x" * len(owned.payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        with owned_simscape_replay_files(envelope, owned, source, binding):
            pytest.fail("tampered bytes reached native decoder")


def test_revalidates_tampered_input_integrity_before_decoder(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
) -> None:
    envelope, owned, source, binding = declaration
    object.__setattr__(envelope.input_history, "values", ((0.0,), (0.0,), (0.0,)))
    with pytest.raises(ValueError, match="integrity hash mismatch"):
        with owned_simscape_replay_files(envelope, owned, source, binding):
            pytest.fail("tampered inputs reached native decoder")


def test_rejects_non_simulation_native_clock_before_materialization(
    declaration: tuple[Any, Any, bytearray, SimscapeRestartBinding],
) -> None:
    envelope, _, source, binding = declaration
    c = native_replay_contract_types()
    artifact = replace(
        envelope.artifact,
        clock=replace(envelope.artifact.clock, clock_id="capture_wall_clock"),
    )
    changed = c.build_native_state_replay_envelope(
        envelope.experiment_id,
        envelope.model,
        artifact,
        envelope.capabilities,
        envelope.input_history,
        envelope.policy,
    )
    owned = c.freeze_native_state_artifact(artifact, b"opaque test bytes")
    with pytest.raises(ValueError, match="simulation_absolute"):
        with owned_simscape_replay_files(changed, owned, source, binding):
            pytest.fail("capture clock was interpreted as native simulation time")
