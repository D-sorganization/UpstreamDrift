"""Source-bound persistence for an already-returned passive readiness audit."""

from array import array
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit

from src.engines.physics_engines.opensim.python.tour_matching.native_constraint_state import (
    ActuatorObservation,
    ConstraintObservation,
    CoordinateConstraintObservation,
    NativeConstraintStateAudit,
)
from src.engines.physics_engines.opensim.python.tour_matching import (
    native_passive_readiness,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_passive_readiness import (
    NativePassiveMuscleObservation,
    NativePassiveReadinessAudit,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_passive_receipt import (
    NativePassiveObservationReceiptReference,
    validate_native_passive_observation_receipt,
    write_native_passive_observation_receipt,
)

OBSERVER_SHA256 = hashlib.sha256(
    Path(native_passive_readiness.__file__).read_bytes()
).hexdigest()


def _audit() -> NativePassiveReadinessAudit:
    native = NativeConstraintStateAudit(
        loaded_model_sha256="a" * 64,
        runtime="OpenSim 4.6 fixture",
        runtime_extension_sha256="b" * 64,
        simbody_extension_sha256="c" * 64,
        observer_sha256="d" * 64,
        time_seconds=0.0,
        coordinates=(
            CoordinateConstraintObservation(
                path="/joint/value",
                motion_type=1,
                value=0.25,
                speed=0.0,
                locked=False,
                prescribed=False,
                clamped=False,
                dependent=False,
                source_default_value=0.0,
                source_default_locked=False,
                declared_lock_target=None,
            ),
        ),
        constraints=(
            ConstraintObservation(
                path="/joint/weld",
                concrete_class="WeldConstraint",
                enforced=True,
                serialized_sha256="f" * 64,
            ),
        ),
        actuators=(
            ActuatorObservation(
                path="/forceset/muscle",
                concrete_class="Thelen2003Muscle",
                is_muscle=True,
                enabled=True,
                control=0.05,
                actuation=0.0,
                power_w=0.0,
                overridden=False,
            ),
        ),
        named_state=(("/joint/value", 0.25),),
        registered_discrete=(),
        modeling_options=(("/muscle/ignore_tendon_compliance", 0.0),),
        derivatives=(("/joint/value", 0.0),),
        muscle_forces=(("/forceset/muscle", 1.0),),
        position_errors=(0.0,),
        velocity_errors=(),
        acceleration_errors=(),
        constraint_jacobian_shape=(0, 0),
        blockers=("complete-native-state-unverified",),
    )
    muscle = NativePassiveMuscleObservation(
        path="/forceset/muscle",
        concrete_class="Thelen2003Muscle",
        path_length_m=0.31,
        fiber_length_m=0.12,
        normalized_fiber_length=1.0,
        fiber_velocity_m_per_s=0.0,
        maximum_isometric_force_n=1000.0,
        passive_fiber_force_n=0.0,
        passive_elastic_fiber_force_n=0.0,
        passive_damping_fiber_force_n=0.0,
        passive_force_cancellation_n=0.0,
        tendon_force_n=0.0,
        fiber_along_tendon_force_n=0.0,
        activation=0.05,
        excitation=0.05,
        ignore_tendon_compliance=False,
        ignore_activation_dynamics=False,
    )
    return NativePassiveReadinessAudit(
        native_state=native,
        muscles=(muscle,),
        policy_sha256=None,
        observer_sha256=OBSERVER_SHA256,
        blockers=("passive-policy-unavailable",),
    )


def _write(
    tmp_path: Path, audit: NativePassiveReadinessAudit | None = None
) -> tuple[NativePassiveObservationReceiptReference, NativePassiveReadinessAudit]:
    observed = audit or _audit()
    reference = write_native_passive_observation_receipt(
        tmp_path,
        observed,
        request_sha256="1" * 64,
        source_sha256="2" * 64,
        runtime_sha256="3" * 64,
        observer_sha256=OBSERVER_SHA256,
    )
    return reference, observed


def _validate(
    tmp_path: Path, reference: NativePassiveObservationReceiptReference
) -> None:
    validate_native_passive_observation_receipt(
        tmp_path,
        reference,
        request_sha256="1" * 64,
        source_sha256="2" * 64,
        loaded_model_sha256="a" * 64,
        runtime_sha256="3" * 64,
        policy_sha256=None,
        observer_sha256=OBSERVER_SHA256,
    )


def test_receipt_roundtrip_preserves_audit_digest_and_explicit_none_policy(
    tmp_path: Path,
) -> None:
    original = _audit()
    original_digest = original.observation_sha256
    reference = write_native_passive_observation_receipt(
        tmp_path,
        original,
        request_sha256="1" * 64,
        source_sha256="2" * 64,
        runtime_sha256="3" * 64,
        observer_sha256=OBSERVER_SHA256,
    )

    payload = json.loads((tmp_path / reference.relative_path).read_text())
    assert original.observation_sha256 == original_digest
    assert reference.audit_sha256 == original_digest
    assert reference.size_bytes == (tmp_path / reference.relative_path).stat().st_size
    assert (
        reference.sha256
        == hashlib.sha256((tmp_path / reference.relative_path).read_bytes()).hexdigest()
    )
    assert payload["policy_sha256"] is None
    assert payload["observation_sha256"] == original_digest
    assert (
        reference.audit_sha256
        == payload["observation_sha256"]
        == original.observation_sha256
    )
    assert payload["observation"] == json.loads(json.dumps(asdict(original)))
    _validate(tmp_path, reference)


def test_sink_captures_the_same_audit_once_without_reobserving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_preparation,
    )

    observed = _audit()
    calls = 0

    def audit_once(model: Any, state: Any, policy: Any) -> NativePassiveReadinessAudit:
        nonlocal calls
        calls += 1
        return observed

    captured: list[NativePassiveReadinessAudit] = []
    monkeypatch.setattr(
        native_moco_preparation, "audit_native_passive_readiness", audit_once
    )
    observation_sha256, blockers = native_moco_preparation._audit_native_passive(
        object(), object(), None, passive_audit_sink=captured.append
    )

    assert calls == 1
    assert len(captured) == 1
    assert captured[0] is observed
    assert observation_sha256 == observed.observation_sha256
    assert blockers == ["passive-policy-unavailable"]


def test_receipt_rejects_stale_identity_tampering_and_malformed_bytes(
    tmp_path: Path,
) -> None:
    reference, _ = _write(tmp_path)
    with pytest.raises(ValueError, match="request"):
        validate_native_passive_observation_receipt(
            tmp_path,
            reference,
            request_sha256="9" * 64,
            source_sha256="2" * 64,
            loaded_model_sha256="a" * 64,
            runtime_sha256="3" * 64,
            policy_sha256=None,
            observer_sha256=OBSERVER_SHA256,
        )

    receipt = tmp_path / reference.relative_path
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        _validate(tmp_path, reference)
    receipt.write_bytes(b"{not-json")
    malformed = replace(
        reference,
        size_bytes=len(b"{not-json"),
        sha256=hashlib.sha256(b"{not-json").hexdigest(),
    )
    with pytest.raises(ValueError, match="JSON"):
        _validate(tmp_path, malformed)


def test_receipt_rejects_nonfinite_payload_even_with_recomputed_outer_digest(
    tmp_path: Path,
) -> None:
    reference, _ = _write(tmp_path)
    path = tmp_path / reference.relative_path
    payload = json.loads(path.read_text())
    payload["observation"]["native_state"]["time_seconds"] = 1e999
    raw = json.dumps(payload, sort_keys=True, allow_nan=True).encode() + b"\n"
    path.write_bytes(raw)
    changed = replace(
        reference,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    with pytest.raises(ValueError, match="finite|JSON"):
        _validate(tmp_path, changed)


def test_receipt_refuses_path_escape_and_atomic_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, audit = _write(tmp_path)
    escaped = replace(reference, relative_path="../outside.json")
    with pytest.raises(ValueError, match="relative|owned"):
        _validate(tmp_path, escaped)

    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_passive_receipt,
    )

    def fail_atomic_write(path: Path, raw: bytes) -> Path:
        raise OSError("synthetic atomic write failure")

    monkeypatch.setattr(native_passive_receipt, "atomic_write_bytes", fail_atomic_write)
    with pytest.raises(OSError, match="synthetic atomic write failure"):
        write_native_passive_observation_receipt(
            tmp_path / "write-failure",
            audit,
            request_sha256="1" * 64,
            source_sha256="2" * 64,
            runtime_sha256="3" * 64,
            observer_sha256=OBSERVER_SHA256,
        )
    assert not (tmp_path / "write-failure" / "native_passive_observation.json").exists()


@pytest.mark.parametrize("allow_unqualified", [False, True])
def test_actual_solver_consumer_requires_and_validates_passive_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_unqualified: bool,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_runner,
    )

    reference, _ = _write(tmp_path)
    times = array("d", [0.0])
    expected_clock = hashlib.sha256(times.tobytes()).hexdigest()
    monkeypatch.setattr(
        native_moco_runner, "_require_physical_body_audit", lambda *args: None
    )
    monkeypatch.setattr(
        native_moco_runner,
        "_sha",
        lambda path: (
            OBSERVER_SHA256
            if Path(path).name == "native_passive_readiness.py"
            else {
                "model.osim": "2" * 64,
                "capture.trc": "capture",
                "guess.sto": "guess",
                "registered.trc": "registered",
            }[Path(path).name]
        ),
    )
    monkeypatch.setattr(
        native_moco_runner, "read_trc", lambda _path: SimpleNamespace(time_s=times)
    )
    request = SimpleNamespace(
        identity_sha256="1" * 64,
        allow_unqualified_passive_diagnostic=allow_unqualified,
        model_path=tmp_path / "model.osim",
        model_sha256="2" * 64,
        trc_path=tmp_path / "capture.trc",
        trc_sha256="capture",
        states_guess_path=tmp_path / "guess.sto",
        states_guess_sha256="guess",
        passive_policy=None,
    )
    prepared = SimpleNamespace(
        request_sha256="1" * 64,
        allow_unqualified_passive_diagnostic=allow_unqualified,
        physical_body_inertia_qualified=True,
        physical_body_inertia_sha256="physical",
        ready_for_software_solve=True,
        passive_observation_sha256=reference.audit_sha256,
        passive_observation_receipt=reference,
        native_runtime_sha256="3" * 64,
        loaded_model_sha256="a" * 64,
        source_sha256="2" * 64,
        registered_trc_sha256="registered",
        observation_clock_sha256=expected_clock,
    )

    assert native_moco_runner._require_current_inputs(request, prepared, tmp_path) == (
        tmp_path / "registered.trc"
    )
    (tmp_path / reference.relative_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        native_moco_runner._require_current_inputs(request, prepared, tmp_path)

    prepared.passive_observation_receipt = None
    with pytest.raises(ValueError, match="receipt is missing"):
        native_moco_runner._require_current_inputs(request, prepared, tmp_path)


@pytest.mark.parametrize("fail_receipt_write", [False, True])
@pytest.mark.parametrize(
    "identity_issue",
    [None, "missing-runtime", "missing-loaded", "loaded-mismatch", "digest-mismatch"],
)
def test_prepare_orchestration_persists_the_same_single_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_receipt_write: bool,
    identity_issue: str | None,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_preparation,
    )

    observed = _audit()
    audit_to_capture = observed
    loaded_sha256: str | None = "a" * 64
    runtime_sha256: str | None = "3" * 64
    passive_sha256: str | None = observed.observation_sha256
    if identity_issue == "missing-runtime":
        runtime_sha256 = None
    elif identity_issue == "missing-loaded":
        loaded_sha256 = None
    elif identity_issue == "loaded-mismatch":
        audit_to_capture = replace(
            observed,
            native_state=replace(observed.native_state, loaded_model_sha256="b" * 64),
        )
        passive_sha256 = audit_to_capture.observation_sha256
    elif identity_issue == "digest-mismatch":
        passive_sha256 = "f" * 64
    calls = 0

    def fake_prepare_model(
        request: Any,
        actual: Any,
        blockers: list[str],
        wrapped_path_auditor: Any,
        probe_callable: Any,
        *,
        passive_audit_sink: Any,
        run_capability_probe: bool = True,
    ) -> Any:
        nonlocal calls
        calls += 1
        passive_audit_sink(audit_to_capture)
        return SimpleNamespace(
            loaded_sha256=loaded_sha256,
            reference_sha256=None,
            passive_sha256=passive_sha256,
            runtime_sha256=runtime_sha256,
            probe_sha256=None,
            probe_failure=None,
            probe_attempted=False,
            probe_start=None,
            probe_end=None,
            wrapped_findings=(),
        )

    preflight = SimpleNamespace(
        actual_digests={"model": "2" * 64, "capture": None, "guess": "guess"},
        source_blockers=(),
        blockers=[],
        physical_sha256=None,
        physical_qualified=False,
        capture=None,
        clock_sha256=None,
        registered_sha256=None,
    )
    request = SimpleNamespace(
        identity_sha256="1" * 64,
        model_sha256="2" * 64,
        allow_unqualified_passive_diagnostic=True,
        config=SimpleNamespace(allow_unused_references=False, t_start_s=0.0),
        marker_weights={},
        excluded_markers=(),
        states_guess_sha256="guess",
        qualification_context=None,
        effective_diagnostic_policy=None,
    )
    monkeypatch.setattr(
        native_moco_preparation, "_collect_native_preflight", lambda *args: preflight
    )
    monkeypatch.setattr(
        native_moco_preparation, "_prepare_native_model", fake_prepare_model
    )
    if fail_receipt_write:
        from src.engines.physics_engines.opensim.python.tour_matching import (
            native_passive_receipt,
        )

        def fail_atomic_write(path: Path, raw: bytes) -> Path:
            raise OSError("synthetic receipt write failure")

        monkeypatch.setattr(
            native_passive_receipt, "atomic_write_bytes", fail_atomic_write
        )

    report = native_moco_preparation.prepare_native_moco(
        request, tmp_path / "preparation", wrapped_path_auditor=lambda *args: None
    )

    assert calls == 1
    assert report.passive_observation_sha256 == passive_sha256
    assert report.loaded_model_sha256 == loaded_sha256
    assert report.native_runtime_sha256 == runtime_sha256
    if fail_receipt_write or identity_issue is not None:
        assert report.passive_observation_receipt is None
        assert not report.ready_for_software_solve
        assert "native-passive-observation-receipt-unavailable" in report.blockers
        assert report.diagnostic_assessment is not None
        expected_blocker_digest = hashlib.sha256(
            json.dumps(report.blockers, allow_nan=False).encode()
        ).hexdigest()
        assert (
            report.diagnostic_assessment.blocker_receipt_sha256
            == expected_blocker_digest
        )
        assert not report.diagnostic_assessment.execution_permitted
        return

    reference = report.passive_observation_receipt
    assert reference is not None
    assert reference.audit_sha256 == audit_to_capture.observation_sha256
    raw = (tmp_path / "preparation" / reference.relative_path).read_bytes()
    payload = json.loads(raw)
    assert payload["observation"] == json.loads(json.dumps(asdict(audit_to_capture)))
    assert payload["observation_sha256"] == report.passive_observation_sha256
    assert report.passive_observation_sha256 == audit_to_capture.observation_sha256
    assert payload["loaded_model_sha256"] == report.loaded_model_sha256
    assert payload["runtime_sha256"] == report.native_runtime_sha256


@pytest.mark.parametrize("payload", [[], None, "not-an-object"])
def test_receipt_rejects_non_object_top_level_json(
    tmp_path: Path, payload: Any
) -> None:
    reference, _ = _write(tmp_path)
    path = tmp_path / reference.relative_path
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    changed = replace(
        reference,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    with pytest.raises(ValueError, match="top-level"):
        _validate(tmp_path, changed)


def test_writer_rejects_malformed_muscle_and_actuator_paths_or_flags(
    tmp_path: Path,
) -> None:
    original = _audit()
    native = original.native_state
    muscle = original.muscles[0]
    malformed_inputs = (
        replace(original, muscles=(replace(muscle, path=""),)),
        replace(
            original,
            native_state=replace(
                native, actuators=(replace(native.actuators[0], path=""),)
            ),
        ),
        replace(
            original,
            native_state=replace(
                native, actuators=(replace(native.actuators[0], is_muscle=1),)
            ),
        ),
    )
    for index, audit in enumerate(malformed_inputs):
        with pytest.raises(ValueError, match="paths|flags"):
            write_native_passive_observation_receipt(
                tmp_path / str(index),
                audit,
                request_sha256="1" * 64,
                source_sha256="2" * 64,
                runtime_sha256="3" * 64,
                observer_sha256=OBSERVER_SHA256,
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_sha256", "9" * 64),
        ("loaded_model_sha256", "9" * 64),
        ("runtime_sha256", "9" * 64),
        ("policy_sha256", "9" * 64),
        ("observer_sha256", "9" * 64),
    ],
)
def test_receipt_rejects_stale_expected_provider_identities(
    tmp_path: Path, field: str, value: str
) -> None:
    reference, _ = _write(tmp_path)
    expected = {
        "request_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "loaded_model_sha256": "a" * 64,
        "runtime_sha256": "3" * 64,
        "policy_sha256": None,
        "observer_sha256": OBSERVER_SHA256,
    }
    expected[field] = value
    with pytest.raises(ValueError, match="identity|reference"):
        validate_native_passive_observation_receipt(tmp_path, reference, **expected)


def test_receipt_rejects_nested_state_rebinding_and_incomplete_muscle_coverage(
    tmp_path: Path,
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_passive_receipt,
    )

    reference, audit = _write(tmp_path)
    path = tmp_path / reference.relative_path
    payload = json.loads(path.read_text())
    payload["observation"]["native_state"]["loaded_model_sha256"] = "9" * 64
    payload["observation_sha256"] = native_passive_receipt._audit_digest(
        payload["observation"]
    )
    raw = native_passive_receipt._canonical_json(payload)
    path.write_bytes(raw)
    rebound = replace(
        reference,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        audit_sha256=payload["observation_sha256"],
    )
    with pytest.raises(ValueError, match="prepared-state"):
        _validate(tmp_path, rebound)

    with pytest.raises(ValueError, match="muscle coverage"):
        write_native_passive_observation_receipt(
            tmp_path / "incomplete",
            replace(audit, muscles=()),
            request_sha256="1" * 64,
            source_sha256="2" * 64,
            runtime_sha256="3" * 64,
            observer_sha256=OBSERVER_SHA256,
        )
