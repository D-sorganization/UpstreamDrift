"""Exports publish atomically with pre-run identity and complete populations."""

import csv
import gzip
import json
from pathlib import Path
import pytest
from src.tools.shot_pattern_analysis import matched_export as module
from src.tools.shot_pattern_analysis.core import AnalysisCancelled
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_physics import UnsupportedShot
from src.tools.shot_pattern_analysis.matched_runner import (
    ExperimentConfig,
    run_matched_analysis,
)
from src.tools.shot_pattern_analysis.noise import NoiseSpec
from src.tools.shot_pattern_analysis.provenance import manifest_files_match
from tests.tools.shot_pattern_analysis.test_matched_runner import AnalyticPhysics

pytestmark = pytest.mark.unit


@pytest.fixture
def request_config():
    return ExperimentConfig(
        matched_preset("seven_iron"), NoiseSpec((1, 0, 0, 0.00635, 0, 0, 0)), n_shots=12
    )


@pytest.fixture
def snapshot(monkeypatch):
    value = {"source_sha256": {"fake.py": "abc"}, "native_binary_sha256": "def"}
    monkeypatch.setattr(module, "matched_source_snapshot", lambda: value.copy())
    return value


def test_export_preserves_unsupported_rows_and_hashes(
    tmp_path, request_config, snapshot
):
    result = run_matched_analysis(
        request_config, physics=AnalyticPhysics(), include_scoring=False
    )
    module.export_matched_analysis(result, tmp_path, execution=snapshot, figures=False)
    assert manifest_files_match(tmp_path / "manifest.json")
    with gzip.open(tmp_path / "shots.csv.gz", "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 36
    assert any(r["status"] != "supported" and r["carry_x_m"] == "" for r in rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["input_diagnostics"]["paired_inputs"]
    (tmp_path / "summary.json").write_text("{}")
    assert not manifest_files_match(tmp_path / "manifest.json")


def test_cancel_does_not_publish(tmp_path, request_config, snapshot):
    destination = tmp_path / "result"
    with pytest.raises(AnalysisCancelled):
        module.execute_and_export(
            request_config, destination, cancelled=lambda: True, figures=False
        )
    assert not destination.exists()


def test_nominal_unsupported_is_a_complete_unavailable_record(
    tmp_path, request_config, snapshot, monkeypatch
):
    def unsupported(*args, **kwargs):
        raise UnsupportedShot("nonairborne_launch")

    monkeypatch.setattr(module, "run_matched_analysis", unsupported)
    destination = module.execute_and_export(
        request_config, tmp_path / "result", figures=False
    )
    assert manifest_files_match(destination / "manifest.json")
    result = json.loads((destination / "summary.json").read_text())
    assert result["status"] == "unavailable"
    assert result["reason"] == "nonairborne_launch"


def test_existing_data_never_overwritten(tmp_path, request_config, snapshot):
    (tmp_path / "keep.txt").write_text("private unrelated data")
    with pytest.raises(FileExistsError):
        module.execute_and_export(request_config, tmp_path, figures=False)
    assert (tmp_path / "keep.txt").read_text() == "private unrelated data"


def test_changed_source_aborts_publication(tmp_path, request_config, monkeypatch):
    values = iter([{"version": 1}, {"version": 2}])
    monkeypatch.setattr(module, "matched_source_snapshot", lambda: next(values))
    result = run_matched_analysis(
        request_config, physics=AnalyticPhysics(), include_scoring=False
    )
    monkeypatch.setattr(module, "run_matched_analysis", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError, match="source changed"):
        module.execute_and_export(request_config, tmp_path / "result", figures=False)
    assert not (tmp_path / "result").exists()


def test_source_snapshot_records_contact_backend(monkeypatch):
    monkeypatch.setattr(module, "source_snapshot", lambda: {"source_sha256": {}})
    monkeypatch.setattr(
        module,
        "contact_backend_info",
        lambda: {"active_backend": "numba", "numba_version": "0.68.0"},
    )
    snapshot = module.matched_source_snapshot()
    assert snapshot["contact_backend"]["active_backend"] == "numba"
    assert snapshot["contact_backend"]["numba_version"] == "0.68.0"
    assert snapshot["numerical_runtime"]["python"]
    assert snapshot["numerical_runtime"]["numpy"]
    assert snapshot["numerical_runtime"]["scipy"]


def test_receipt_records_selected_contact_backend(tmp_path, request_config, snapshot):
    from dataclasses import replace

    club = replace(request_config.club, contact_backend="python_reference")
    request = replace(request_config, club=club)
    result = run_matched_analysis(
        request, physics=AnalyticPhysics(), include_scoring=False
    )
    module.export_matched_analysis(result, tmp_path, execution=snapshot, figures=False)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["contact_execution"] == {
        "contact_law": "incremental_coulomb_energetic",
        "requested_backend": "python_reference",
        "selected_backend": "python_reference",
        "steps_per_compression": 128,
    }


@pytest.mark.slow
def test_real_figure_export_contract(tmp_path, request_config, snapshot):
    from dataclasses import replace

    result = run_matched_analysis(
        request_config, physics=AnalyticPhysics(), include_scoring=False
    )
    paths = dict.fromkeys(result.nominals, (((0.0, 0.0), (100.0, 0.0)),))
    result = replace(result, flight_samples=paths)
    module.export_matched_analysis(result, tmp_path, execution=snapshot, figures=True)
    assert all(
        (tmp_path / (name + suffix)).is_file()
        for name in ("density", "overhead")
        for suffix in (".png", ".pdf", ".svg")
    )
    assert manifest_files_match(tmp_path / "manifest.json")
