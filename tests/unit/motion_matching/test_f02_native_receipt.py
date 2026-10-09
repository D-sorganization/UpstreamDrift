"""Fail-closed receipt extraction from actual native pytest evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.f02_native_manifold_receipt import extract_native_evidence

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _junit(tmp_path: Path, evidence: dict[str, object] | None) -> Path:
    path = tmp_path / "native.xml"
    property_xml = (
        ""
        if evidence is None
        else f"<properties><property name=\"f02_native_evidence\" value='{json.dumps(evidence)}'/></properties>"
    )
    path.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="test_native_f02_feedback_replays_full_state_and_beats_frozen_nominal">'
        f"{property_xml}</testcase></testsuite>",
        encoding="utf-8",
    )
    return path


def _evidence() -> dict[str, object]:
    return {
        **dict.fromkeys(
            (
                "source_model_sha256",
                "loaded_native_model_sha256",
                "initial_state_sha256",
                "policy_sha256",
                "time_grid_sha256",
                "state_schema_sha256",
                "input_channel_schema_sha256",
                "controlled_applied_input_sha256",
                "nominal_applied_input_sha256",
            ),
            "a" * 64,
        ),
        "controlled_final_hip_error_rad": 0.2,
        "nominal_final_hip_error_rad": 0.4,
        "max_full_state_replay_error": 0.0,
        "nominal_applied_input_sha256": "b" * 64,
    }


def test_receipt_requires_passed_native_property_and_improvement(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    assert extract_native_evidence(_junit(tmp_path, evidence)) == evidence
    with pytest.raises(ValueError, match="property"):
        extract_native_evidence(_junit(tmp_path, None))
    evidence["controlled_final_hip_error_rad"] = 0.5
    with pytest.raises(ValueError, match="improvement"):
        extract_native_evidence(_junit(tmp_path, evidence))
    evidence = _evidence()
    evidence["source_model_sha256"] = "unbound"
    with pytest.raises(ValueError, match="SHA"):
        extract_native_evidence(_junit(tmp_path, evidence))
