"""SHOW-C6 (#14562): swing-variant artefact with #14436 run identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.shot_presets import preset
from src.shared.python.impact_optimization.swing_variant import (
    build_variant,
    load_variant,
)
from src.sim_run import RunManifestError, validate_run_manifest
from tests.unit.impact_optimization.test_trajectory_optimiser import (
    _envelope,
    _solve,
)

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures/sim_run/run_manifest_valid_v1.json"
)


def _manifest(shot: str = "draw"):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["parameters"] = {**data["parameters"], "preset": shot, "curve_deg": 4.0}
    return validate_run_manifest(data)


@pytest.fixture(scope="module")
def solution():
    return _solve(_envelope())


def test_variant_records_preset_target_solution_and_run_identity(solution) -> None:
    v = build_variant(preset("draw"), solution, _manifest("draw"))
    d = v.to_dict()
    assert d["preset"] == "draw"
    assert d["target"]["face_deg"] == pytest.approx(preset("draw").target.face_deg)
    assert d["solution"]["deviation_from_fit"] == pytest.approx(
        solution.deviation_from_fit
    )
    assert d["solution"]["knots"] and d["assumptions"]
    assert d["run_manifest"]["seed"] == 20261104
    assert d["run_manifest"]["model"]["revision"].startswith("sha256:")


def test_save_and_load_round_trips_and_revalidates_identity(
    solution, tmp_path: Path
) -> None:
    path = tmp_path / "draw_variant.json"
    build_variant(preset("draw"), solution, _manifest("draw")).save(path)
    loaded = load_variant(path)
    assert loaded["preset"] == "draw"
    assert loaded["run_manifest"]["engine"]["name"] == "mujoco"


def test_load_rejects_a_tampered_run_identity(solution, tmp_path: Path) -> None:
    path = tmp_path / "v.json"
    build_variant(preset("fade"), solution, _manifest("fade")).save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["run_manifest"]["ud_commit"] = "main"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RunManifestError):
        load_variant(path)


def test_manifest_must_name_the_preset_it_was_run_for(solution) -> None:
    with pytest.raises(ContractViolationError):
        build_variant(preset("draw"), solution, _manifest("fade"))
