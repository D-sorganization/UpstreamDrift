"""Physical source admission is mandatory before native Moco optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from defusedxml import ElementTree
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_passive_readiness import (
    MusclePassiveLimits,
    PassiveReadinessPolicy,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
    NativeMocoRequest,
    prepare_native_moco,
    solve_native_moco,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CaptureRegistration,
)
from tests.opensim.test_moco_initial_bindings import _native_inputs
from tests.opensim.test_native_muscle_replay import muscle_fixture as muscle_fixture
from src.shared.python.estimation import dime_global_calibration

pytestmark = pytest.mark.integration


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(
    fixture: tuple[Path, dict[str, float]], directory: Path
) -> NativeMocoRequest:
    model, capture, guess, bindings = _native_inputs(fixture, directory)
    return NativeMocoRequest(
        model_path=model,
        trc_path=capture,
        states_guess_path=guess,
        model_sha256=_sha(model),
        trc_sha256=_sha(capture),
        states_guess_sha256=_sha(guess),
        bindings=bindings,
        config=MocoTrackingConfig(horizon_s=0.01, allow_unused_references=False),
        marker_weights={"load_marker": 2.0},
        marker_bindings={"load_marker": ("/bodyset/load", (0.0, 0.0, 0.0))},
        registration=CaptureRegistration(np.eye(3), np.zeros(3)),
        reference_frame_path="/ground",
        passive_policy=None,
        excluded_markers={},
    )


def _ready_request(
    fixture: tuple[Path, dict[str, float]], directory: Path
) -> NativeMocoRequest:
    osim = pytest.importorskip("opensim")
    request = _request(fixture, directory)
    model = osim.Model(str(request.model_path))
    model.initSystem()
    policy = PassiveReadinessPolicy(
        hashlib.sha256(model.dump().encode()).hexdigest(),
        "synthetic fixture only; no human physiology claim",
        (
            MusclePassiveLimits(
                "/forceset/flexor",
                (0.1, 10.0),
                1000.0,
                1000.0,
                False,
                False,
                "predeclared synthetic fixture envelope",
            ),
        ),
    )
    return replace(request, passive_policy=policy)


def test_valid_physical_body_audit_is_bound_to_preparation(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    request = _request(muscle_fixture, tmp_path)

    report = prepare_native_moco(request, tmp_path / "prepared")

    assert report.physical_body_inertia_qualified is True
    assert report.physical_body_inertia_sha256 == _sha(
        tmp_path / "prepared" / "physical_body_inertia.json"
    )
    assert report.blockers == ("passive-policy-unavailable",)


def test_rotated_unphysical_body_cannot_reach_moco_solve(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner as runner

    def fail_if_probe_runs(*args: object, **kwargs: object) -> tuple[str, float, float]:
        raise AssertionError("invalid physical inertia must block native integration")

    monkeypatch.setattr(runner, "_native_short_capability_probe", fail_if_probe_runs)
    request = _request(muscle_fixture, tmp_path)
    tree = ElementTree.parse(request.model_path)
    inertia = tree.find("./Model/BodySet/objects/Body/inertia")
    assert inertia is not None
    # SPD in the source axes but principal moments (0.02, 0.02, 0.08)
    # violate the physical triangle. Rotation hides the defect on the diagonal.
    inertia.text = "0.05 0.02 0.05 0 -0.03 0"
    request.model_path.write_bytes(
        ElementTree.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    )
    request = replace(request, model_sha256=_sha(request.model_path))

    report = prepare_native_moco(request, tmp_path / "prepared")

    assert report.physical_body_inertia_qualified is False
    assert "physical-body-inertia-unqualified" in report.blockers
    physical = json.loads(
        (tmp_path / "prepared" / "physical_body_inertia.json").read_text()
    )
    assert "triangle inequality" in str(physical)
    with pytest.raises(ValueError, match="Physical body inertia"):
        solve_native_moco(request, report, tmp_path / "prepared")
    assert not (tmp_path / "prepared" / "solution.sto").exists()


def test_native_solve_rejects_sidecar_and_validator_drift(
    muscle_fixture: tuple[Path, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _ready_request(muscle_fixture, tmp_path)
    validator_copy = tmp_path / "validator.py"
    validator_copy.write_bytes(Path(dime_global_calibration.__file__).read_bytes())
    monkeypatch.setattr(dime_global_calibration, "__file__", str(validator_copy))
    directory = tmp_path / "prepared"
    report = prepare_native_moco(request, directory)
    assert report.ready_for_software_solve, report.blockers
    sidecar = directory / "physical_body_inertia.json"
    original = sidecar.read_bytes()
    sidecar.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Physical body inertia audit sidecar"):
        solve_native_moco(request, report, directory)
    sidecar.write_bytes(original)
    validator_copy.write_bytes(validator_copy.read_bytes() + b"\n# changed")
    with pytest.raises(ValueError, match="Physical body inertia audit identity"):
        solve_native_moco(request, report, directory)
    assert not (directory / "solution.sto").exists()


def test_missing_physical_audit_cannot_upgrade_old_preparation(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    request = _ready_request(muscle_fixture, tmp_path)
    directory = tmp_path / "prepared"
    report = prepare_native_moco(request, directory)
    assert report.ready_for_software_solve, report.blockers
    stale = replace(report, physical_body_inertia_sha256=None)

    with pytest.raises(ValueError, match="Physical body inertia admission"):
        solve_native_moco(request, stale, directory)


def test_malformed_source_retains_physical_reason_and_other_blockers(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    request = _request(muscle_fixture, tmp_path)
    tree = ElementTree.parse(request.model_path)
    body = tree.find("./Model/BodySet/objects/Body")
    assert body is not None
    mass = body.find("mass")
    inertia = body.find("inertia")
    assert mass is not None and inertia is not None
    mass.text = "0"
    inertia.text = "0.01 0.01 0.01 0 0 0"
    request.model_path.write_bytes(
        ElementTree.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    )
    request = replace(request, model_sha256=_sha(request.model_path))

    report = prepare_native_moco(request, tmp_path / "prepared")

    assert report.physical_body_inertia_qualified is False
    assert "physical-body-inertia-audit-unavailable" in report.blockers
    assert "passive-policy-unavailable" in report.blockers
    physical = json.loads(
        (tmp_path / "prepared" / "physical_body_inertia.json").read_text()
    )
    assert "massless body has nonzero inertia" in physical["reason"]
