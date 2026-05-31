"""Tests for the CC-11 differential report generator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPORT_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "validation"
    / "cross_engine_differential_report.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "cross_engine_differential_report",
    _REPORT_SCRIPT,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load {_REPORT_SCRIPT}")
_REPORT_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _REPORT_MODULE
_SPEC.loader.exec_module(_REPORT_MODULE)
generate_report = _REPORT_MODULE.generate_report


def test_generator_normalizes_cc7_harness_shape(tmp_path: Path) -> None:
    source = {
        "checks": [
            {
                "check_name": "differential_cross_engine_reference",
                "engine_name": "mujoco-canonical-v2",
                "passed": True,
                "validation": {
                    "metric_name": "position",
                    "max_deviation": 0.002,
                    "tolerance": 0.005,
                    "engine1": "mujoco-canonical-v2",
                    "engine2": "pinocchio-canonical-v2",
                    "severity": "WARNING",
                },
                "divergence": {
                    "id": "soft-vs-rigid-contact",
                    "tolerance": 0.01,
                    "rationale": "MuJoCo reports compliant contact separation.",
                },
            }
        ]
    }
    input_path = tmp_path / "cc7.json"
    input_path.write_text(json.dumps(source), encoding="utf-8")

    report = generate_report(
        input_path=input_path,
        json_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        generated_at="2026-05-31T00:00:00+00:00",
    )

    assert report["source_shape"] == "cc7"
    assert report["summary"]["comparison_count"] == 1
    assert report["summary"]["registered_divergence_count"] == 1
    assert report["comparisons"][0]["divergence_id"] == "soft-vs-rigid-contact"
    assert report["comparisons"][0]["tolerance_ratio"] == 0.2


def test_generator_quantifies_contact_free_torque_rows(tmp_path: Path) -> None:
    source = {
        "engines": ["mujoco-canonical-v2", "pinocchio-canonical-v2"],
        "comparisons": [
            {
                "check_name": "contact_free_inverse_dynamics",
                "phase": "contact_free",
                "metric_name": "torque_peak_pct",
                "engine1": "mujoco-canonical-v2",
                "engine2": "pinocchio-canonical-v2",
                "dof": "lead_shoulder_flexion",
                "max_deviation": 0.84,
                "rms_deviation_pct": 0.52,
                "peak_reference": 94.0,
                "tolerance": 2.0,
                "unit": "percent_peak_torque",
                "passed": True,
                "severity": "PASSED",
            }
        ],
    }
    input_path = tmp_path / "cc11.json"
    input_path.write_text(json.dumps(source), encoding="utf-8")

    report = generate_report(
        input_path=input_path,
        json_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        generated_at="2026-05-31T00:00:00+00:00",
    )

    assert report["status"] == "passed"
    assert report["summary"]["contact_free_torque_max_pct_of_peak"] == 0.52
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "lead_shoulder_flexion" in markdown
    assert "0.84 percent_peak_torque" in markdown


def test_default_report_documents_dependency_block(tmp_path: Path) -> None:
    report = generate_report(
        json_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        generated_at="2026-05-31T00:00:00+00:00",
    )

    assert report["status"] == "blocked_by_draft_dependencies"
    assert report["comparisons"] == []
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "No live adapter comparison rows are claimed yet" in markdown
    assert "draft PR #6828" in markdown
