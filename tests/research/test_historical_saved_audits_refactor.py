"""Saved-audit parity uses synthetic analytical endpoints, never historical studies."""

import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import pytest

from docs.research.shot_pattern_analysis.phase2.validation import (
    independent_final_audit as final_audit,
)
from docs.research.shot_pattern_analysis.phase2.validation import (
    independent_pilot_audit as pilot_audit,
)
from src.tools.shot_pattern_analysis import matched_export
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_physics import UnsupportedShot
from src.tools.shot_pattern_analysis.matched_runner import (
    ExperimentConfig,
    run_matched_analysis,
)
from src.tools.shot_pattern_analysis.noise import NoiseSpec
from tests.research.test_historical_geometry_refactor import _archived_module
from tests.tools.shot_pattern_analysis.test_matched_runner import AnalyticPhysics

pytestmark = pytest.mark.unit
EXECUTION = {"source_sha256": {"synthetic": "fixed"}}


def _synthetic_cell(folder, monkeypatch, mode):
    strike = 0.00635 if mode == "conditional" else 0.0
    request = ExperimentConfig(
        matched_preset("seven_iron"),
        NoiseSpec((1, 0.5, 0.44704, strike, 0, 0, 0)),
        n_shots=12,
    )
    config = json.loads(json.dumps(asdict(request)))
    monkeypatch.setattr(matched_export, "matched_source_snapshot", lambda: EXECUTION)
    if mode == "unavailable":
        folder.mkdir(parents=True)
        (folder / "run_start.json").write_text(
            json.dumps({"config": config, "execution": EXECUTION})
        )
        (folder / "summary.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "status": "unavailable",
                    "shot_rows": 0,
                    "reason": "synthetic unavailable",
                }
            )
        )
        matched_export._write_manifest(folder)
        return config

    class SyntheticPhysics(AnalyticPhysics):
        def simulate_delivery(self, **kwargs):
            if (
                mode == "insufficient"
                and abs(kwargs.get("shaft_rotation_deg", 0)) > 0.0001
            ):
                raise UnsupportedShot("off_face")
            return super().simulate_delivery(**kwargs)

    result = run_matched_analysis(
        request,
        physics=SyntheticPhysics(),
        include_scoring=mode != "unavailable_scoring",
    )
    matched_export.export_matched_analysis(
        result, folder, execution=EXECUTION, figures=False
    )
    if mode == "unavailable_scoring":
        path = folder / "summary.json"
        summary = json.loads(path.read_text())
        summary["course_scoring"].update(
            status="unavailable", reason="synthetic scoring domain"
        )
        path.write_text(json.dumps(summary))
        matched_export._write_manifest(folder)
    if mode == "wrong_summary":
        path = folder / "summary.json"
        summary = json.loads(path.read_text())
        summary["summary"]["Straight"]["mean_carry_m"] += 0.1
        path.write_text(json.dumps(summary))
        matched_export._write_manifest(folder)
    if mode == "corrupt_manifest":
        (folder / "summary.json").write_text("{}")
    return config


def _audit(module, folder, config):
    maxima = dict.fromkeys(module.LIMITS, 0.0)
    try:
        result = module.audit_cell(
            folder,
            config,
            EXECUTION,
            {},
            module.build_broadie_approx_baseline(),
            maxima,
        )
        error = None
    except (AssertionError, ValueError, KeyError, TypeError, OSError) as exc:
        result, error = None, repr(exc)
    return {"result": result, "error": error, "maxima": maxima}


@pytest.mark.parametrize(
    "mode",
    [
        "complete",
        "conditional",
        "insufficient",
        "unavailable",
        "unavailable_scoring",
        "corrupt_manifest",
        "wrong_summary",
    ],
)
def test_final_cell_audit_preserves_exact_values_failures_and_partial_maxima(
    tmp_path, monkeypatch, mode
):
    folder = tmp_path / "seven_iron_lean0_curve1_rotation1_combined2"
    config = _synthetic_cell(folder, monkeypatch, mode)
    before = _audit(_archived_module(final_audit), folder, config)
    after = _audit(final_audit, folder, config)
    assert json.dumps(after, allow_nan=False) == json.dumps(before, allow_nan=False)
    if mode in ("corrupt_manifest", "wrong_summary"):
        assert after["error"] is not None
    else:
        assert after["error"] is None


def _pilot_run(module, root, output, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["pilot-audit", str(root), "--output", str(output)]
    )
    stream = io.StringIO()
    with redirect_stdout(stream):
        module.main()
    return {"bytes": output.read_text(), "stdout": stream.getvalue()}


def test_pilot_audit_preserves_counts_maxima_hashes_and_failure_collection(
    tmp_path, monkeypatch
):
    root = tmp_path / "pilot"
    cells = {}
    for mode in ("complete", "unavailable", "wrong_summary", "corrupt_manifest"):
        name = f"{mode}_lean0_curve1_rotation1_combined2"
        cells[name] = _synthetic_cell(root / name, monkeypatch, mode)
    (root / "matrix_plan.json").write_text(
        json.dumps({"cells": cells, "execution": EXECUTION})
    )
    before = _pilot_run(
        _archived_module(pilot_audit), root, tmp_path / "before.json", monkeypatch
    )
    after = _pilot_run(pilot_audit, root, tmp_path / "after.json", monkeypatch)
    assert before == after
    receipt = json.loads(after["bytes"])
    assert receipt["rows_checked"] == 72
    assert receipt["direct_api_rows"] == 72
    assert receipt["counts"] == {"completed": 2, "unavailable": 1}
    assert len(receipt["failures"]) == 1
    assert not receipt["pass"] and receipt["maxima"]["statistics"] >= 0.1 - 1e-12


def test_population_audit_keeps_shared_supported_index_sets(tmp_path, monkeypatch):
    import pandas as pd
    from src.tools.shot_pattern_analysis.noise import sample_paired_noise

    folder = tmp_path / "synthetic_population"
    config = _synthetic_cell(folder, monkeypatch, "conditional")
    summary = json.loads((folder / "summary.json").read_text())
    data = pd.read_csv(folder / "shots.csv.gz", float_precision="round_trip")
    noise = sample_paired_noise(
        NoiseSpec(**config["noise"]), n_shots=config["n_shots"], seed=config["seed"]
    )
    maxima = dict.fromkeys(final_audit.LIMITS, 0.0)
    frames, supports = final_audit._audit_population(
        data, config, summary, noise, maxima
    )
    assert list(frames) == list(supports) == ["Straight", "Draw", "Fade"]
    for pattern in frames:
        assert frames[pattern].index.tolist() == list(range(config["n_shots"]))
        assert supports[pattern] == set(
            data.loc[
                (data["pattern"] == pattern) & (data["status"] == "supported"),
                "shot_index",
            ]
        )
    assert 2 <= len(set.intersection(*supports.values())) < config["n_shots"]


def test_pilot_accumulator_starts_with_all_original_counter_categories():
    state = pilot_audit._PilotAuditState()
    assert list(state.maximum) == [
        "statistics",
        "covariance",
        "noise",
        "nominal_residual",
        "direct_api_sg",
    ]
    assert state.hashes_checked == state.rows_checked == state.api_rows == 0
