"""SDK-free tests for explicitly skipping the native capability probe."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import (
    native_moco_preparation,
)
from src.engines.physics_engines.opensim.python.tour_matching.cli import main
from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
    MocoInitialBindings,
)
from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_diagnostic_contract import (
    DiagnosticCode,
    RuntimeViability,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_preparation import (
    NativeMocoPreparation,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_request import (
    NativeMocoRequest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_skip_probe_report_and_gate_are_explicit(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model.osim"
    trc_path = tmp_path / "capture.trc"
    guess_path = tmp_path / "guess.sto"
    model_path.write_bytes(b"synthetic model")
    trc_path.write_bytes(b"synthetic capture")
    guess_path.write_bytes(b"synthetic guess")
    request = NativeMocoRequest(
        model_path=model_path,
        trc_path=trc_path,
        states_guess_path=guess_path,
        model_sha256=_sha(model_path),
        trc_sha256=_sha(trc_path),
        states_guess_sha256=_sha(guess_path),
        bindings=MocoInitialBindings({}, {}, {}),
        config=MocoTrackingConfig(horizon_s=0.01, allow_unused_references=False),
        marker_weights={},
        marker_bindings={},
        registration=None,
        reference_frame_path="/ground",
        passive_policy=None,
        excluded_markers={},
    )
    preflight = native_moco_preparation._NativePreflight(
        actual_digests={
            "model": request.model_sha256,
            "capture": request.trc_sha256,
            "guess": request.states_guess_sha256,
        },
        source_blockers=[],
        blockers=[],
        physical_sha256="e" * 64,
        physical_qualified=True,
        capture=None,
        clock_sha256=None,
        registered_sha256=None,
    )
    received: list[bool] = []
    monkeypatch.setattr(
        native_moco_preparation,
        "_collect_native_preflight",
        lambda request, directory, blockers: preflight,
    )

    def prepare_model(
        request, actual, blockers, auditor, probe_callable, *, run_capability_probe
    ):
        received.append(run_capability_probe)
        return native_moco_preparation._NativeModelPreparation(
            loaded_sha256=None,
            reference_sha256=None,
            passive_sha256=None,
            runtime_sha256=None,
            probe_sha256=None,
            probe_attempted=False,
            probe_start=None,
            probe_end=None,
            probe_failure=None,
            wrapped_findings=(),
        )

    monkeypatch.setattr(native_moco_preparation, "_prepare_native_model", prepare_model)
    report = native_moco_preparation.prepare_native_moco(
        request, tmp_path / "output", run_capability_probe=False
    )

    assert received == [False]
    saved = json.loads(
        (tmp_path / "output" / "preparation.json").read_text(encoding="utf-8")
    )
    assessment = saved["diagnostic_assessment"]
    assert saved["capability_probe_requested"] is False
    assert assessment["runtime_viability"] == RuntimeViability.NOT_ATTEMPTED.value
    assert assessment["runtime_phase"] is None
    assert assessment["runtime_window_seconds"] is None
    assert all(
        finding["code"] != DiagnosticCode.NATIVE_SHORT_PROBE_COMPLETED.value
        for finding in assessment["findings"]
    )
    assert report.ready_for_software_solve is False


@pytest.mark.parametrize("invalid_value", ["false", None, 0])
def test_non_boolean_probe_option_rejects_before_preflight(
    invalid_value: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_preflight(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid option must fail before preflight")

    monkeypatch.setattr(
        native_moco_preparation, "_collect_native_preflight", unexpected_preflight
    )
    output_dir = tmp_path / "must-not-be-created"

    with pytest.raises(TypeError, match="run_capability_probe must be a bool"):
        native_moco_preparation.prepare_native_moco(
            object(), output_dir, run_capability_probe=invalid_value
        )

    assert not output_dir.exists()


def test_capability_probe_request_is_an_independent_software_gate() -> None:
    blockers: tuple[str, ...] = ()
    blocker_sha = hashlib.sha256(json.dumps(blockers).encode()).hexdigest()
    admitted = SimpleNamespace(
        blocker_receipt_sha256=blocker_sha,
        execution_permitted=True,
    )
    prepared = NativeMocoPreparation(
        request_sha256="a" * 64,
        source_sha256="b" * 64,
        capture_sha256="c" * 64,
        guess_sha256="d" * 64,
        loaded_model_sha256="e" * 64,
        reference_observation_sha256="f" * 64,
        passive_observation_sha256=None,
        observation_clock_sha256=None,
        registered_trc_sha256=None,
        physical_body_inertia_sha256="1" * 64,
        physical_body_inertia_qualified=True,
        marker_count=0,
        observation_count=0,
        excluded_marker_count=0,
        blockers=blockers,
        wall_seconds=0.0,
        cpu_seconds=0.0,
        diagnostic_assessment=admitted,
        capability_probe_requested=False,
    )

    assert prepared.ready_for_software_solve is False
    assert replace(prepared, capability_probe_requested=True).ready_for_software_solve


def test_cli_skip_probe_is_prepare_only_and_reaches_api(
    monkeypatch, tmp_path: Path
) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_request,
        native_moco_runner,
    )

    request = object()
    calls: list[bool] = []
    monkeypatch.setattr(
        native_moco_request, "load_native_moco_request", lambda path: request
    )

    def prepare(request, output_dir, *, run_capability_probe=True):
        calls.append(run_capability_probe)
        return SimpleNamespace(ready_for_software_solve=False, blockers=("not-ready",))

    monkeypatch.setattr(native_moco_runner, "prepare_native_moco", prepare)
    args = [
        "moco-native",
        "--request",
        str(tmp_path / "request.json"),
        "--output-dir",
        str(tmp_path / "output"),
        "--prepare-only",
        "--skip-capability-probe",
    ]
    assert main(args) == 2
    assert calls == [False]

    calls.clear()
    assert main(args[:-1]) == 2
    assert calls == [True]

    calls.clear()
    invalid_args = args[:-2] + ["--skip-capability-probe"]
    assert main(invalid_args) == 2
    assert calls == []
