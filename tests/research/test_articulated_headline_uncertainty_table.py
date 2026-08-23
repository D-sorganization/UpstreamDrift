"""Reviewer-table contracts for articulated headline uncertainty evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import (
    export_articulated_headline_uncertainty_table as exporter,
)

pytestmark = pytest.mark.scientific


def _record() -> dict[str, object]:
    return {
        "schema_version": "articulated-headline-uncertainty/v1",
        "study_id": "articulated-shaft-ground-headline-uncertainty",
        "status": "complete",
        "design": {
            "method": "registered_nominal_plus_one_at_a_time_low_high_corners",
            "corner_count": 3,
        },
        "corners": [
            {
                "corner_id": "nominal",
                "axis_name": "nominal",
                "level": "nominal",
                "value": 1.0,
                "pathways": ["shaft", "ground"],
                "shaft": {
                    "status": "completed",
                    "failure_class": None,
                    "failure_message": None,
                    "matched_cell_count": 126,
                    "matched_cell_count_change_from_nominal": 0,
                    "total_cell_count": 384,
                    "all_registered_gates_passed": True,
                },
                "ground": {
                    "status": "completed",
                    "failure_class": None,
                    "failure_message": None,
                    "matched_cell_count": 0,
                    "matched_cell_count_change_from_nominal": 0,
                    "total_cell_count": 384,
                    "all_registered_gates_passed": True,
                },
            },
            {
                "corner_id": "grip_stiffness_scale-low",
                "axis_name": "grip_stiffness_scale",
                "level": "low",
                "value": 0.6,
                "pathways": ["shaft", "ground"],
                "shaft": {
                    "status": "failed_retained",
                    "failure_class": "RuntimeError",
                    "failure_message": "linear shaft domain exceeded",
                    "matched_cell_count": None,
                    "matched_cell_count_change_from_nominal": None,
                    "total_cell_count": 384,
                    "all_registered_gates_passed": False,
                },
                "ground": {
                    "status": "completed",
                    "failure_class": None,
                    "failure_message": None,
                    "matched_cell_count": 3,
                    "matched_cell_count_change_from_nominal": 3,
                    "total_cell_count": 384,
                    "all_registered_gates_passed": True,
                },
            },
            {
                "corner_id": "ground_translation_stiffness_scale-high",
                "axis_name": "ground_translation_stiffness_scale",
                "level": "high",
                "value": 1.5,
                "pathways": ["ground"],
                "shaft": {
                    "status": "not_affected",
                    "matched_cell_count": None,
                    "matched_cell_count_change_from_nominal": None,
                    "total_cell_count": 384,
                    "all_registered_gates_passed": None,
                },
                "ground": {
                    "status": "failed_retained",
                    "failure_class": "RuntimeError",
                    "failure_message": "native ground integration failed",
                    "matched_cell_count": None,
                    "matched_cell_count_change_from_nominal": None,
                    "total_cell_count": 384,
                    "all_registered_gates_passed": False,
                },
            },
        ],
    }


def _write_record(path: Path, record: dict[str, object]) -> str:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_preserves_every_corner_pathway_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "headline.json"
    record_sha256 = _write_record(record_path, _record())
    source_set_sha256 = "a" * 64
    monkeypatch.setattr(
        exporter,
        "audit_headline_record",
        lambda _: {
            "status": "complete",
            "record_sha256": record_sha256,
            "source_set_sha256": source_set_sha256,
        },
    )
    output = tmp_path / "headline.csv"

    row_count = exporter.export_headline_uncertainty_table(record_path, output)

    assert row_count == 6
    first_bytes = output.read_bytes()
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert [(row["corner_id"], row["pathway"]) for row in rows] == [
        ("nominal", "shaft"),
        ("nominal", "ground"),
        ("grip_stiffness_scale-low", "shaft"),
        ("grip_stiffness_scale-low", "ground"),
        ("ground_translation_stiffness_scale-high", "shaft"),
        ("ground_translation_stiffness_scale-high", "ground"),
    ]
    assert {row["campaign_record_sha256"] for row in rows} == {record_sha256}
    assert {row["source_set_sha256"] for row in rows} == {source_set_sha256}
    assert {row["evidence_scope"] for row in rows} == {
        "summary_counts_not_trajectory_data"
    }
    failed = rows[2]
    assert failed["status"] == "failed_retained"
    assert failed["failure_class"] == "RuntimeError"
    assert failed["matched_cell_count"] == ""
    not_affected = rows[4]
    assert not_affected["pathway_registered"] == "false"
    assert not_affected["status"] == "not_affected"
    assert not_affected["all_registered_gates_passed"] == ""

    exporter.export_headline_uncertainty_table(record_path, output)
    assert output.read_bytes() == first_bytes


def test_export_rejects_partial_or_changed_campaign_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record_path = tmp_path / "headline.json"
    record_sha256 = _write_record(record_path, _record())
    output = tmp_path / "headline.csv"
    monkeypatch.setattr(
        exporter,
        "audit_headline_record",
        lambda _: {
            "status": "partial",
            "record_sha256": record_sha256,
            "source_set_sha256": "a" * 64,
        },
    )
    with pytest.raises(ValueError, match="complete campaign"):
        exporter.export_headline_uncertainty_table(record_path, output)

    monkeypatch.setattr(
        exporter,
        "audit_headline_record",
        lambda _: {
            "status": "complete",
            "record_sha256": "b" * 64,
            "source_set_sha256": "a" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="changed during export"):
        exporter.export_headline_uncertainty_table(record_path, output)
