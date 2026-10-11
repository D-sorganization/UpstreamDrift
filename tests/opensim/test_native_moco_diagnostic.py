"""Explicit numerical feasibility retains unavailable physiological admission."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_request import (
    load_native_moco_request,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
    NativeMocoRequest,
    prepare_native_moco,
    solve_native_moco,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_diagnostic_contract import (
    DiagnosticCode,
    ExecutionEffect,
    NativeDiagnosticFinding,
    NativeDiagnosticPolicy,
    NativeQualificationContext,
    PhysicalQualification,
    QualificationEffect,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_replay import (
    export_native_moco_bundle,
    replay_native_moco_bundle,
    score_native_moco_replay,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_mechanics_profile import (
    NativeMechanicsComponentIdentity,
    NativeMechanicsExecutionProfile,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CaptureRegistration,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
    DeclaredColdStart,
)
from tests.opensim.test_moco_initial_bindings import _native_inputs
from tests.opensim.test_native_moco_runner import _write_request
from tests.opensim.test_native_moco_inertia_preflight import _ready_request

pytest_plugins = ("tests.opensim.test_native_muscle_replay",)
pytestmark = pytest.mark.unit


def _request(
    fixture: tuple[Path, dict[str, float]], directory: Path
) -> NativeMocoRequest:
    model, trc, guess, bindings = _native_inputs(
        fixture, directory, times=np.linspace(0.0, 0.01, 11)
    )

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return NativeMocoRequest(
        model,
        trc,
        guess,
        sha(model),
        sha(trc),
        sha(guess),
        bindings,
        MocoTrackingConfig(
            horizon_s=0.01,
            mesh_interval_s=0.005,
            optim_max_iterations=30,
            allow_unused_references=False,
        ),
        {"load_marker": 2.0},
        {"load_marker": ("/bodyset/load", (0.0, 0.0, 0.0))},
        CaptureRegistration(np.eye(3), np.zeros(3)),
        "/ground",
        None,
        {},
    )


def test_opt_in_solves_and_replays_without_claiming_passive_admission(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    ordinary = _request(muscle_fixture, tmp_path)
    blocked = prepare_native_moco(ordinary, tmp_path / "ordinary")
    assert not blocked.ready_for_software_solve
    request = replace(ordinary, allow_unqualified_passive_diagnostic=True)
    assert request.identity_sha256 != ordinary.identity_sha256
    directory = tmp_path / "diagnostic"
    prepared = prepare_native_moco(request, directory)
    assert prepared.blockers == ("passive-policy-unavailable",)
    assert prepared.passive_observation_sha256
    assert prepared.ready_for_software_solve
    assert prepared.qualification == "not-qualified-for-muscle-matching"
    saved = json.loads((directory / "preparation.json").read_text())
    assert saved["allow_unqualified_passive_diagnostic"] is True
    assert saved["blockers"] == ["passive-policy-unavailable"]
    solved = solve_native_moco(request, prepared, directory)
    assert solved.success
    assert solved.qualification == "not-qualified-for-muscle-matching"
    exported = export_native_moco_bundle(request, prepared, solved, directory)
    replayed = replay_native_moco_bundle(exported, request.model_path, directory)
    score = score_native_moco_replay(
        request, prepared, solved, exported, replayed, directory
    )
    assert np.isfinite(score.marker_rmse_m)
    assert score.observed_samples == 11
    assert score.qualification == "synthetic-or-exploratory-executed-match-only"
    replay_receipt = json.loads((directory / "replay.json").read_text())
    assert (
        replay_receipt["qualification"] == "independent-native-input-reproduction-only"
    )


def test_diagnostic_never_waives_other_blockers_or_physical_qualification(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    request = replace(
        _request(muscle_fixture, tmp_path), allow_unqualified_passive_diagnostic=True
    )
    directory = tmp_path / "prepared"
    prepared = prepare_native_moco(request, directory)
    # Passive admission is explicitly unqualified, but may be opted into for
    # software execution. Removing its optional observation does not certify it.
    without_passive_observation = replace(prepared, passive_observation_sha256=None)
    assert without_passive_observation.ready_for_software_solve
    assert (
        without_passive_observation.diagnostic_assessment.physical_qualification
        is PhysicalQualification.UNQUALIFIED
    )
    for changed in (
        replace(prepared, physical_body_inertia_qualified=False),
        replace(
            prepared, blockers=prepared.blockers + ("native-state-binding-coverage",)
        ),
    ):
        assert not changed.ready_for_software_solve
        with pytest.raises(ValueError):
            solve_native_moco(request, changed, directory)
    forged = replace(prepared, allow_unqualified_passive_diagnostic=False)
    with pytest.raises(ValueError, match="diagnostic disposition"):
        solve_native_moco(request, forged, directory)
    with pytest.raises(ValueError, match="request changed"):
        solve_native_moco(
            replace(request, allow_unqualified_passive_diagnostic=False),
            prepared,
            directory,
        )


@pytest.mark.parametrize("value", [1, "true", None])
def test_diagnostic_disposition_requires_boolean(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path, value: object
) -> None:
    with pytest.raises(TypeError, match="boolean"):
        replace(
            _request(muscle_fixture, tmp_path),
            allow_unqualified_passive_diagnostic=value,
        )


def test_file_request_explicit_opt_in_round_trips(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    ordinary = _request(muscle_fixture, tmp_path)
    path = tmp_path / "request.json"
    _write_request(ordinary, path)
    assert not load_native_moco_request(path).allow_unqualified_passive_diagnostic
    payload = json.loads(path.read_text())
    payload["allow_unqualified_passive_diagnostic"] = True
    path.write_text(json.dumps(payload))
    request = load_native_moco_request(path)
    assert request.allow_unqualified_passive_diagnostic is True
    assert (
        request.identity_sha256
        == replace(ordinary, allow_unqualified_passive_diagnostic=True).identity_sha256
    )


def test_native_mechanics_profile_json_recomputes_digest_and_rejects_unknown_fields() -> (
    None
):
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_request import (
        _parse_native_mechanics_profile,
    )

    profile = NativeMechanicsExecutionProfile(
        "native-sdk-mechanics-v1",
        "a" * 64,
        (NativeMechanicsComponentIdentity("/jointset/base", "CustomJoint", "b" * 64),),
    )
    payload = {
        "schema_version": profile.schema_version,
        "source_sha256": profile.source_sha256,
        "components": [
            {
                "absolute_path": item.absolute_path,
                "concrete_class": item.concrete_class,
                "serialized_sha256": item.serialized_sha256,
            }
            for item in profile.components
        ],
        "profile_sha256": profile.profile_sha256,
    }
    restored = _parse_native_mechanics_profile(payload)
    assert restored == profile
    with pytest.raises(ValueError, match="digest"):
        _parse_native_mechanics_profile({**payload, "profile_sha256": "c" * 64})
    with pytest.raises(ValueError, match="unknown fields"):
        _parse_native_mechanics_profile({**payload, "unexpected": True})


def test_native_contact_declaration_json_recomputes_digest_and_rejects_unknown_fields() -> (
    None
):
    from src.engines.physics_engines.opensim.python.tour_matching.native_contact_policy import (
        NativeContactExecutionDeclaration,
        NativeContactForceIdentity,
        NativeContactGeometryIdentity,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_request import (
        _parse_native_contact_declaration,
    )

    declaration = NativeContactExecutionDeclaration(
        "native-sphere-halfspace-contact-v1",
        "a" * 64,
        (
            NativeContactForceIdentity(
                "/forceset/contact", "HuntCrossleyForce", "b" * 64
            ),
        ),
        (
            NativeContactGeometryIdentity(
                "/contactgeometryset/floor", "ContactHalfSpace", "c" * 64
            ),
            NativeContactGeometryIdentity(
                "/contactgeometryset/sphere", "ContactSphere", "d" * 64
            ),
        ),
    )
    payload = {
        "schema_version": declaration.schema_version,
        "source_sha256": declaration.source_sha256,
        "forces": [
            {
                "absolute_path": item.absolute_path,
                "concrete_class": item.concrete_class,
                "serialized_sha256": item.serialized_sha256,
            }
            for item in declaration.forces
        ],
        "geometries": [
            {
                "absolute_path": item.absolute_path,
                "concrete_class": item.concrete_class,
                "serialized_sha256": item.serialized_sha256,
            }
            for item in declaration.geometries
        ],
        "declaration_sha256": declaration.declaration_sha256,
    }
    assert _parse_native_contact_declaration(payload) == declaration
    with pytest.raises(ValueError, match="digest"):
        _parse_native_contact_declaration({**payload, "declaration_sha256": "e" * 64})
    with pytest.raises(ValueError, match="unknown fields"):
        _parse_native_contact_declaration({**payload, "extra": True})


def test_native_mechanics_profile_requires_scoped_typed_request_policy(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    request = _request(muscle_fixture, tmp_path)
    profile = NativeMechanicsExecutionProfile(
        "native-sdk-mechanics-v1",
        request.model_sha256,
        (NativeMechanicsComponentIdentity("/jointset/base", "CustomJoint", "b" * 64),),
    )
    with pytest.raises(ValueError, match="constrained-muscle execution"):
        replace(request, native_mechanics_profile=profile)
    declared = DeclaredColdStart(
        request.model_path.resolve(),
        request.model_sha256,
        request.bindings.initial_state,
        request.config.t_start_s,
        {},
        {},
        {},
        1e-9,
    )
    constrained = replace(request, constrained_cold_start=declared)
    with pytest.raises(ValueError, match="typed diagnostic policy"):
        replace(constrained, native_mechanics_profile=profile)
    explicit = replace(
        constrained,
        diagnostic_policy=NativeDiagnosticPolicy(
            request.model_sha256,
            (DiagnosticCode.GUIDE_WORK_BUDGET_UNAVAILABLE,),
            "source-bound diagnostic execution",
        ),
        native_mechanics_profile=profile,
    )
    assert explicit.identity_sha256 != constrained.identity_sha256
    assert (
        explicit.identity_sha256
        != replace(explicit, native_mechanics_profile=None).identity_sha256
    )


def test_invalid_native_mechanics_profile_blocks_before_probe(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner as runner

    request = _request(muscle_fixture, tmp_path)
    declared = DeclaredColdStart(
        request.model_path.resolve(),
        request.model_sha256,
        request.bindings.initial_state,
        request.config.t_start_s,
        {},
        {},
        {},
        1e-9,
    )
    policy = NativeDiagnosticPolicy(
        request.model_sha256,
        (DiagnosticCode.GUIDE_WORK_BUDGET_UNAVAILABLE,),
        "source-bound diagnostic execution",
    )
    profile = NativeMechanicsExecutionProfile(
        "native-sdk-mechanics-v1",
        request.model_sha256,
        (
            NativeMechanicsComponentIdentity(
                "/jointset/undeclared", "CustomJoint", "b" * 64
            ),
        ),
    )
    request = replace(
        request,
        constrained_cold_start=declared,
        diagnostic_policy=policy,
        native_mechanics_profile=profile,
    )

    def fail_if_probe_runs(*args: object, **kwargs: object) -> tuple[str, float, float]:
        raise AssertionError("invalid profile must block native integration")

    monkeypatch.setattr(runner, "_native_short_capability_probe", fail_if_probe_runs)
    prepared = prepare_native_moco(request, tmp_path / "invalid-profile")

    assert not prepared.ready_for_software_solve
    assert "native-mechanics-profile-invalid" in prepared.blockers
    assert prepared.diagnostic_assessment is not None
    assert prepared.diagnostic_assessment.runtime_viability.value == "not-attempted"


def test_typed_source_bound_policy_round_trips_without_changing_default_identity(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    ordinary = _request(muscle_fixture, tmp_path)
    assert ordinary.identity_sha256 == replace(ordinary).identity_sha256
    path = tmp_path / "typed-policy.json"
    _write_request(ordinary, path)
    payload = json.loads(path.read_text())
    payload["diagnostic_policy"] = {
        "source_sha256": ordinary.model_sha256,
        "allowed_unqualified_findings": ["passive_policy_unavailable"],
        "provenance": "reviewed bounded diagnostic request",
    }
    path.write_text(json.dumps(payload))

    loaded = load_native_moco_request(path)

    assert loaded.diagnostic_policy == NativeDiagnosticPolicy(
        source_sha256=ordinary.model_sha256,
        allowed_unqualified_findings=(DiagnosticCode.PASSIVE_POLICY_UNAVAILABLE,),
        provenance="reviewed bounded diagnostic request",
    )
    assert loaded.identity_sha256 != ordinary.identity_sha256
    assert loaded.effective_diagnostic_policy == loaded.diagnostic_policy


def test_source_bound_qualification_context_round_trips_strictly(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    ordinary = _request(muscle_fixture, tmp_path)
    evidence = "c" * 64
    context = NativeQualificationContext(
        source_sha256=ordinary.model_sha256,
        findings=(
            NativeDiagnosticFinding(
                code=DiagnosticCode.SUPPORT_CONTACT_GRIP_UNQUALIFIED,
                source_sha256=ordinary.model_sha256,
                evidence_sha256=evidence,
                execution_effect=ExecutionEffect.PROCEED,
                qualification_effect=QualificationEffect.UNQUALIFIED,
            ),
        ),
        provenance_references=((evidence, ("private://component-inventory",)),),
    )
    with_context = replace(ordinary, qualification_context=context)
    path = tmp_path / "qualification-context.json"
    _write_request(ordinary, path)
    payload = json.loads(path.read_text())
    payload["qualification_context"] = {
        "source_sha256": context.source_sha256,
        "findings": [
            {
                "code": context.findings[0].code.value,
                "source_sha256": context.findings[0].source_sha256,
                "evidence_sha256": evidence,
                "execution_effect": "proceed",
                "qualification_effect": "unqualified",
            }
        ],
        "provenance_references": [
            {
                "evidence_sha256": evidence,
                "references": ["private://component-inventory"],
            }
        ],
    }
    path.write_text(json.dumps(payload))
    loaded = load_native_moco_request(path)
    assert loaded.qualification_context == context
    assert loaded.identity_sha256 == with_context.identity_sha256
    assert loaded.identity_sha256 != ordinary.identity_sha256

    payload["qualification_context"]["findings"][0]["execution_effect"] = "block"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="effect is not registered"):
        load_native_moco_request(path)


@pytest.mark.parametrize("invalid_input", ["guess", "clock-policy"])
def test_outer_admission_blockers_skip_native_time_integration(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_input: str,
) -> None:
    import src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner as runner

    request = _request(muscle_fixture, tmp_path)
    if invalid_input == "guess":
        request = replace(request, states_guess_sha256="0" * 64)
    else:
        request = replace(
            request,
            config=replace(request.config, allow_unused_references=True),
        )

    def fail_if_probe_runs(*args: object, **kwargs: object) -> tuple[str, float, float]:
        raise AssertionError("native probe must not run after an outer blocker")

    monkeypatch.setattr(runner, "_native_short_capability_probe", fail_if_probe_runs)
    prepared = prepare_native_moco(request, tmp_path / f"blocked-{invalid_input}")

    assert not prepared.ready_for_software_solve
    assert prepared.diagnostic_assessment is not None
    assert prepared.diagnostic_assessment.runtime_viability.value == "not-attempted"


def test_wrapped_path_closure_block_skips_native_time_integration(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner as runner

    request = _request(muscle_fixture, tmp_path)
    failure = NativeDiagnosticFinding(
        code=DiagnosticCode.WRAPPED_PATH_SOURCE_CLOSURE_INVALID,
        source_sha256=request.model_sha256,
        evidence_sha256="d" * 64,
        execution_effect=ExecutionEffect.BLOCK,
        qualification_effect=QualificationEffect.NONE,
    )
    monkeypatch.setattr(
        runner,
        "audit_native_wrapped_path_source",
        lambda *args, **kwargs: SimpleNamespace(findings=(failure,)),
    )

    def fail_if_probe_runs(*args: object, **kwargs: object) -> tuple[str, float, float]:
        raise AssertionError("native probe must not run after wrap closure failure")

    monkeypatch.setattr(runner, "_native_short_capability_probe", fail_if_probe_runs)
    prepared = prepare_native_moco(request, tmp_path / "blocked-wrap")

    assert not prepared.ready_for_software_solve
    assert DiagnosticCode.WRAPPED_PATH_SOURCE_CLOSURE_INVALID in {
        finding.code for finding in prepared.diagnostic_assessment.findings
    }


def test_native_probe_failure_distinguishes_last_observation_from_reached_time(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner as runner

    request = _request(muscle_fixture, tmp_path)

    def fail_during_native_joint_functions(
        *args: object, **kwargs: object
    ) -> tuple[str, float, float]:
        raise runner.NativeProbeFailure(
            "native-short-integration",
            "RuntimeError",
            "synthetic native function failure",
            None,
            attempted_window_seconds=(0.0, 0.001),
            last_observed_time_seconds=0.0,
            cause_chain=(("ValueError", "bounded synthetic cause"),),
        )

    monkeypatch.setattr(
        runner, "_native_short_capability_probe", fail_during_native_joint_functions
    )
    directory = tmp_path / "failed-probe"
    prepared = prepare_native_moco(request, directory)

    failure = json.loads((directory / "native_probe_failure_private.json").read_text())
    assert failure["stage"] == "native-short-integration"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "synthetic native function failure"
    assert failure["observed_window_seconds"] is None
    assert failure["attempted_window_seconds"] == [0.0, 0.001]
    assert failure["last_observed_time_seconds"] == 0.0
    assert failure["reached_time_seconds"] is None
    assert failure["cause_chain"] == [
        {"exception_type": "ValueError", "message": "bounded synthetic cause"}
    ]
    assert prepared.diagnostic_assessment is not None
    assert prepared.diagnostic_assessment.runtime_viability.value == "failed"
    assert prepared.diagnostic_assessment.runtime_window_seconds is None


def test_supplied_limits_cannot_be_overridden_by_diagnostic_disposition(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    request = _ready_request(muscle_fixture, tmp_path)
    with pytest.raises(ValueError, match="unavailable passive policy"):
        replace(request, allow_unqualified_passive_diagnostic=True)


def test_native_runtime_identity_binds_shared_library_formats(tmp_path: Path) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_capability_probe import (
        _native_runtime_identity,
    )

    package = tmp_path / "opensim"
    package.mkdir()
    binding = package / "__init__.py"
    binding.write_text("# fixture binding\n", encoding="utf-8")
    library_names = (
        "native.dll",
        "native.pyd",
        "libnative.so",
        "libnative.so.3",
        "libnative.dylib",
    )
    libraries = {}
    for name in library_names:
        path = package / name
        path.write_bytes(b"original-native-library")
        libraries[name] = path

    class FakeOpenSim:
        __file__ = str(binding)
        __version__ = "4.6-fixture"

    opensim = FakeOpenSim()
    _native_runtime_identity.cache_clear()
    original_identity = _native_runtime_identity(opensim)
    for path in libraries.values():
        original_bytes = path.read_bytes()
        path.write_bytes(original_bytes + b"changed")
        _native_runtime_identity.cache_clear()
        assert _native_runtime_identity(opensim) != original_identity, path.name
        path.write_bytes(original_bytes)

    _native_runtime_identity.cache_clear()
