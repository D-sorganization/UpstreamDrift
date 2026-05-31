"""Tests for canonical-core setup validation and wizard progression."""

from __future__ import annotations

import pytest

from src.shared.python.config.setup_wizard import (
    SetupValidationIssue,
    SetupWizardViewModel,
    validate_canonical_setup_config,
)
from src.shared.python.launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    EmbedCapabilities,
    EmbeddableTool,
)
from src.tools.config_setup_wizard._embed_adapter import ConfigSetupWizardAdapter

pytestmark = [pytest.mark.unit]


def _valid_config() -> dict[str, object]:
    return {
        "convention": "canonical-v2",
        "units": {
            "length": "m",
            "mass": "kg",
            "time": "s",
            "angle": "rad",
            "force": "N",
            "torque": "N*m",
        },
        "frame": "world_Zup",
        "gravity": [0.0, 0.0, -9.80665],
        "model": {
            "canonical_id": "golf_humanoid_v1",
            "joint_names": ["hip_flexion", "knee_flexion"],
            "nq": 9,
            "nv": 8,
        },
        "calibration": {
            "status": "complete",
            "anthropometrics_ref": "subjects/golfer-001.json",
        },
    }


def test_valid_config_passes_without_issues() -> None:
    report = validate_canonical_setup_config(_valid_config())

    assert report.is_valid is True
    assert report.issues == ()


def test_units_and_frame_failures_include_plain_language_fixes() -> None:
    config = _valid_config()
    config["convention"] = "canonical-v1"
    config["units"] = {"length": "cm", "mass": "kg", "time": "s"}
    config["frame"] = "world_Yup"
    config["gravity"] = [0.0, -9.80665, 0.0]

    report = validate_canonical_setup_config(config)
    by_code = {issue.code: issue for issue in report.issues}

    assert report.is_valid is False
    assert by_code["CC36_CONVENTION"].message.startswith("This setup must use")
    assert by_code["CC36_CONVENTION"].suggested_fix == (
        'Set "convention" to "canonical-v2".'
    )
    assert any(
        issue.code == "CC36_UNIT_MISMATCH" and issue.field_path == "units.length"
        for issue in report.issues
    )
    assert by_code["CC36_WORLD_FRAME"].suggested_fix == (
        'Set "frame" or "world_frame" to "world_Zup".'
    )
    assert by_code["CC36_GRAVITY_FRAME"].suggested_fix == (
        'Use "gravity": [0.0, 0.0, -9.80665].'
    )


def test_model_dimension_failure_prevents_common_canonical_misconfig() -> None:
    config = _valid_config()
    config["model"] = {
        "canonical_id": "golf_humanoid_v1",
        "joint_names": ["hip_flexion"],
        "nq": 8,
        "nv": 8,
    }

    report = validate_canonical_setup_config(config)

    assert any(issue.code == "CC36_MODEL_DIMENSIONS" for issue in report.errors)
    assert report.issues_for_step("model")[0].field_path.startswith("model")


def test_missing_calibration_reports_required_subject_fix() -> None:
    config = _valid_config()
    config.pop("calibration")

    report = validate_canonical_setup_config(config)

    assert report.errors == (
        SetupValidationIssue(
            code="CC36_CALIBRATION_REQUIRED",
            field_path="calibration",
            message="Subject calibration is required before this run can start.",
            suggested_fix=(
                "Run or import the calibration step and attach a calibration "
                "block with status=complete and anthropometrics_ref."
            ),
        ),
    )


def test_incomplete_calibration_requires_completion_and_subject_reference() -> None:
    config = _valid_config()
    config["calibration"] = {"status": "draft"}

    report = validate_canonical_setup_config(config)
    codes = {issue.code for issue in report.errors}

    assert codes == {"CC36_CALIBRATION_INCOMPLETE", "CC36_CALIBRATION_SUBJECT"}


def test_wizard_blocks_on_current_step_then_advances_after_fix() -> None:
    wizard = SetupWizardViewModel()
    config = _valid_config()
    config["frame"] = "world_Yup"

    snapshot = wizard.advance(config)

    assert snapshot.current_step == "units_frames"
    assert snapshot.steps[0].status == "blocked"
    assert snapshot.steps[0].can_advance is False

    config["frame"] = "world_Zup"
    snapshot = wizard.advance(config)

    assert snapshot.current_step == "model"
    assert snapshot.steps[0].status == "complete"
    assert snapshot.steps[1].status == "ready"


def test_wizard_progresses_to_review_only_after_each_gate_passes() -> None:
    wizard = SetupWizardViewModel()
    config = _valid_config()

    assert wizard.advance(config).current_step == "model"
    assert wizard.advance(config).current_step == "calibration"
    assert wizard.advance(config).current_step == "review"
    assert wizard.advance(config).current_step == "review"


def test_embed_adapter_is_headless_protocol_compliant() -> None:
    adapter = ConfigSetupWizardAdapter()

    assert isinstance(adapter, EmbeddableTool)
    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.min_size == (720, 520)
    assert adapter.is_dirty() is False
    adapter.cleanup()
    adapter.cleanup()


def test_embed_adapter_registers_on_import() -> None:
    assert "config_setup_wizard" in EMBEDDABLE_TOOL_REGISTRY
