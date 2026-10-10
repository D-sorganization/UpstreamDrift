"""Source-bound numerical Moco guesses preserve native names and the frozen clock."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.cli import main
from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
    build_moco_study,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_guess import (
    audit_native_moco_guess,
    materialize_native_moco_guess,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
    NativeMocoRequest,
    prepare_native_moco,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CaptureRegistration,
)
from tests.opensim.test_moco_initial_bindings import _native_inputs

pytest_plugins = ("tests.opensim.test_native_muscle_replay",)
pytestmark = pytest.mark.unit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_native_guess_roundtrips_and_populates_actual_moco_problem(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    osim = pytest.importorskip("opensim")
    clock = np.linspace(0.0, 0.01, 11)
    path, trc, _, bindings = _native_inputs(muscle_fixture, tmp_path, clock)
    receipt = materialize_native_moco_guess(path, _sha(path), clock, tmp_path / "guess")
    guess_path = tmp_path / "guess" / "native_initial_guess.sto"
    assert receipt.source_sha256 == _sha(path)
    assert receipt.guess_sha256 == _sha(guess_path)
    assert receipt.state_count == len(bindings.initial_state)
    assert receipt.knot_count == len(clock)
    assert receipt.qualification == "numerical-seed-only"
    table = osim.TimeSeriesTable(str(guess_path))
    names = tuple(table.getColumnLabels())
    assert names == tuple(bindings.initial_state)
    assert np.array_equal(np.asarray(table.getIndependentColumn()), clock)
    assert np.isfinite(table.getMatrix().to_numpy()).all()
    assert audit_native_moco_guess(
        path, guess_path, _sha(path), clock
    ).state_count == len(names)
    study = build_moco_study(
        str(path),
        str(trc),
        str(guess_path),
        MocoTrackingConfig(horizon_s=0.01, mesh_interval_s=0.005),
        initial_bindings=bindings,
    )
    moco_guess = osim.MocoCasADiSolver.safeDownCast(study.updSolver()).getGuess()
    assert set(moco_guess.getStateNames()) == set(bindings.state_bounds)
    assert set(moco_guess.getControlNames()) == set(bindings.control_bounds)


def test_source_and_clock_changes_reject_before_writing(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    path, _ = muscle_fixture
    clock = np.array([0.0, 0.005, 0.01])
    with pytest.raises(ValueError, match="source"):
        materialize_native_moco_guess(path, "0" * 64, clock, tmp_path / "stale")
    assert not (tmp_path / "stale" / "native_initial_guess.sto").exists()
    with pytest.raises(ValueError, match="clock"):
        materialize_native_moco_guess(
            path, _sha(path), np.array([0.0, 0.005, 0.005]), tmp_path / "clock"
        )
    assert not (tmp_path / "clock" / "native_initial_guess.sto").exists()


def test_native_guess_audit_rejects_missing_or_nonfinite_state(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    osim = pytest.importorskip("opensim")
    path, _ = muscle_fixture
    clock = np.array([0.0, 0.005, 0.01])
    materialize_native_moco_guess(path, _sha(path), clock, tmp_path)
    guess_path = tmp_path / "native_initial_guess.sto"
    table = osim.TimeSeriesTable(str(guess_path))
    table.removeColumnAtIndex(0)
    osim.STOFileAdapter.write(table, str(tmp_path / "missing.sto"))
    with pytest.raises(ValueError, match="state names"):
        audit_native_moco_guess(path, tmp_path / "missing.sto", _sha(path), clock)
    lines = guess_path.read_text(encoding="utf-8").splitlines()
    first_data = lines.index("endheader") + 2
    fields = lines[first_data].split("\t")
    fields[1] = "nan"
    lines[first_data] = "\t".join(fields)
    (tmp_path / "nonfinite.sto").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite|invalid"):
        audit_native_moco_guess(path, tmp_path / "nonfinite.sto", _sha(path), clock)
    with pytest.raises(ValueError, match="clock"):
        audit_native_moco_guess(
            path, guess_path, _sha(path), np.array([0.0, 0.004, 0.01])
        )


def test_preparation_distinguishes_real_guess_from_scientific_blockers(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    clock = np.linspace(0.0, 0.01, 11)
    path, trc, _, bindings = _native_inputs(muscle_fixture, tmp_path, clock)
    receipt = materialize_native_moco_guess(path, _sha(path), clock, tmp_path / "seed")
    request = NativeMocoRequest(
        path,
        trc,
        tmp_path / "seed" / "native_initial_guess.sto",
        _sha(path),
        _sha(trc),
        receipt.guess_sha256,
        bindings,
        MocoTrackingConfig(horizon_s=0.01, mesh_interval_s=0.005),
        {"load_marker": 1.0},
        {"load_marker": ("/bodyset/load", (0.0, 0.0, 0.0))},
        CaptureRegistration(np.eye(3), np.zeros(3)),
        "/ground",
        None,
        {},
    )
    report = prepare_native_moco(request, tmp_path / "prepare")
    assert "guess-sha256-mismatch" not in report.blockers
    assert "native-guess-invalid" not in report.blockers
    assert "passive-policy-unavailable" in report.blockers
    assert not report.ready_for_software_solve
    corrupted_path = tmp_path / "seed" / "truncated_guess.sto"
    original = request.states_guess_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    corrupted_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    corrupted = replace(
        request,
        states_guess_path=corrupted_path,
        states_guess_sha256=_sha(corrupted_path),
    )
    malformed = prepare_native_moco(corrupted, tmp_path / "malformed")
    assert "native-guess-invalid" in malformed.blockers


def test_actual_moco_builder_rejects_incomplete_seed_state_columns(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    osim = pytest.importorskip("opensim")
    path, trc, _, bindings = _native_inputs(muscle_fixture, tmp_path)
    incomplete = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    only = next(iter(bindings.state_bounds))
    labels.append(only)
    incomplete.setColumnLabels(labels)
    for instant in (0.0, 0.005, 0.01):
        row = osim.RowVector(1)
        row[0] = bindings.initial_state[only]
        incomplete.appendRow(instant, row)
    path_guess = tmp_path / "missing_states.sto"
    osim.STOFileAdapter.write(incomplete, str(path_guess))
    with pytest.raises(ValueError, match="state names"):
        build_moco_study(
            str(path),
            str(trc),
            str(path_guess),
            MocoTrackingConfig(horizon_s=0.01),
            initial_bindings=bindings,
        )


def test_native_guess_cli_consumes_file_inputs_without_capture_inference(
    muscle_fixture: tuple[Path, dict[str, float]], tmp_path: Path
) -> None:
    clock = np.linspace(0.0, 0.01, 11)
    path, trc, _, _ = _native_inputs(muscle_fixture, tmp_path, clock)
    output = tmp_path / "cli_guess"
    assert (
        main(
            [
                "moco-native-guess",
                "--model",
                str(path),
                "--source-sha256",
                _sha(path),
                "--trc",
                str(trc),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert audit_native_moco_guess(
        path, output / "native_initial_guess.sto", _sha(path), clock
    ).knot_count == len(clock)
