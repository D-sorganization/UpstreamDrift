"""Explicit numerical feasibility retains unavailable physiological admission."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

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
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_replay import (
    export_native_moco_bundle,
    replay_native_moco_bundle,
    score_native_moco_replay,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CaptureRegistration,
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


def test_diagnostic_never_waives_other_blockers_or_missing_native_observation(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    request = replace(
        _request(muscle_fixture, tmp_path), allow_unqualified_passive_diagnostic=True
    )
    directory = tmp_path / "prepared"
    prepared = prepare_native_moco(request, directory)
    for changed in (
        replace(prepared, passive_observation_sha256=None),
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


def test_supplied_limits_cannot_be_overridden_by_diagnostic_disposition(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    request = _ready_request(muscle_fixture, tmp_path)
    with pytest.raises(ValueError, match="unavailable passive policy"):
        replace(request, allow_unqualified_passive_diagnostic=True)
