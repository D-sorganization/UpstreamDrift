"""Figure contracts for articulated closed-state and trajectory uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.make_articulated_uncertainty_figure import (
    render_articulated_uncertainty_figure,
)

pytestmark = pytest.mark.scientific


def _write_evidence(directory: Path) -> tuple[Path, Path]:
    parameters = np.asarray(("height_scale", "joint_limit_scale"))
    metrics = np.asarray(("peak_force_n", "energy_residual"))
    failures = np.asarray(
        ("feasible", "slip_occurring", "feasible", "joint_limit_failure")
    )
    record = {
        "schema_version": "articulated-uncertainty-study/v2",
        "design": {"method": "deterministic_latin_hypercube"},
        "uncertainty_parameters": parameters.tolist(),
        "output_metrics": metrics.tolist(),
        "results": {
            "sample_count": 4,
            "analysis_included_count": 3,
            "failure_distribution": {
                "feasible": 2,
                "joint_limit_failure": 1,
                "slip_occurring": 1,
            },
        },
    }
    record_path = directory / "uncertainty.json"
    arrays_path = directory / "uncertainty.npz"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    np.savez_compressed(
        arrays_path,
        parameter_names=parameters,
        output_metric_names=metrics,
        parameter_samples=np.zeros((4, 2)),
        response_matrix=np.zeros((4, 2)),
        prcc_sensitivity_matrix=np.asarray(((0.7, -0.2), (-0.4, 0.5))),
        failure_classes=failures,
        analysis_included=np.asarray((True, True, True, False)),
    )
    return record_path, arrays_path


def test_render_articulated_uncertainty_figure(tmp_path: Path) -> None:
    record, arrays = _write_evidence(tmp_path)
    output = tmp_path / "figure"

    render_articulated_uncertainty_figure(record, arrays, output)

    assert output.with_suffix(".pdf").stat().st_size > 1_000
    svg = output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Articulated Closed-State and Local-Trajectory Uncertainty Screen" in svg
    assert "Joint Limit Failure" in svg


def test_figure_rejects_mismatched_committed_evidence(tmp_path: Path) -> None:
    record_path, arrays_path = _write_evidence(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["results"]["sample_count"] = 5
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="sample count"):
        render_articulated_uncertainty_figure(
            record_path, arrays_path, tmp_path / "figure"
        )
