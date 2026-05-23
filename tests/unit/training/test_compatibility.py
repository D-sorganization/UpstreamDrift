"""Tests for :mod:`training.compatibility`."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import (
    CompatibilityChecker,
    CompatibilityIssue,
    CompatibilityReport,
    TrainingConfig,
    TrainingFramework,
)

pytestmark = pytest.mark.unit


def _pytorch_config() -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="m:train",
        output_dir=Path("/tmp/out"),
    )


def _gymnasium_config() -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.GYMNASIUM,
        entry_point="m:train",
        output_dir=Path("/tmp/out"),
    )


class TestCompatibilityIssue:
    def test_default_severity_is_error(self) -> None:
        issue = CompatibilityIssue(code="x", message="y")
        assert issue.severity == "error"

    def test_warning_severity(self) -> None:
        issue = CompatibilityIssue(code="x", message="y", severity="warning")
        assert issue.severity == "warning"

    def test_rejects_empty_code(self) -> None:
        with pytest.raises(ValueError):
            CompatibilityIssue(code="", message="y")

    def test_rejects_empty_message(self) -> None:
        with pytest.raises(ValueError):
            CompatibilityIssue(code="x", message="")

    def test_rejects_invalid_severity(self) -> None:
        with pytest.raises(ValueError):
            CompatibilityIssue(code="x", message="y", severity="critical")  # type: ignore[arg-type]


class TestCompatibilityReport:
    def test_empty_report_is_compatible(self) -> None:
        report = CompatibilityReport()
        assert report.is_compatible is True
        assert report.errors == ()
        assert report.warnings == ()

    def test_only_warnings_is_compatible(self) -> None:
        report = CompatibilityReport(
            issues=(CompatibilityIssue("x", "y", severity="warning"),)
        )
        assert report.is_compatible is True
        assert len(report.warnings) == 1
        assert report.errors == ()

    def test_error_blocks_compatibility(self) -> None:
        report = CompatibilityReport(
            issues=(CompatibilityIssue("x", "y", severity="error"),)
        )
        assert report.is_compatible is False
        assert len(report.errors) == 1

    def test_mixed_errors_and_warnings(self) -> None:
        report = CompatibilityReport(
            issues=(
                CompatibilityIssue("e1", "err", severity="error"),
                CompatibilityIssue("w1", "warn", severity="warning"),
                CompatibilityIssue("e2", "err2", severity="error"),
            )
        )
        assert report.is_compatible is False
        assert len(report.errors) == 2
        assert len(report.warnings) == 1


class TestCompatibilityChecker:
    def test_default_map_includes_core_engines(self) -> None:
        checker = CompatibilityChecker()
        assert "mujoco" in checker.known_engines
        assert "drake" in checker.known_engines
        assert "pinocchio" in checker.known_engines

    def test_mujoco_supports_pytorch_and_gymnasium(self) -> None:
        checker = CompatibilityChecker()
        assert checker.check(_pytorch_config(), "mujoco").is_compatible
        assert checker.check(_gymnasium_config(), "mujoco").is_compatible

    def test_drake_pytorch_compatible(self) -> None:
        checker = CompatibilityChecker()
        report = checker.check(_pytorch_config(), "drake")
        assert report.is_compatible

    def test_drake_gymnasium_incompatible_with_clear_code(self) -> None:
        """Drake doesn't ship an RL env wrapper in v1 — gymnasium must fail."""
        checker = CompatibilityChecker()
        report = checker.check(_gymnasium_config(), "drake")
        assert report.is_compatible is False
        codes = [i.code for i in report.errors]
        assert "framework_unsupported" in codes

    def test_unknown_engine_yields_specific_error_code(self) -> None:
        checker = CompatibilityChecker()
        report = checker.check(_pytorch_config(), "no_such_engine")
        assert report.is_compatible is False
        assert any(i.code == "unknown_engine" for i in report.errors)

    def test_engine_name_is_case_insensitive(self) -> None:
        checker = CompatibilityChecker()
        assert checker.check(_pytorch_config(), "MuJoCo").is_compatible
        assert checker.check(_pytorch_config(), "DRAKE").is_compatible

    def test_engine_name_whitespace_trimmed(self) -> None:
        checker = CompatibilityChecker()
        assert checker.check(_pytorch_config(), "  drake  ").is_compatible

    def test_rejects_non_config_input(self) -> None:
        checker = CompatibilityChecker()
        with pytest.raises(TypeError):
            checker.check("not a config", "mujoco")  # type: ignore[arg-type]

    def test_rejects_empty_engine_name(self) -> None:
        checker = CompatibilityChecker()
        with pytest.raises(ValueError):
            checker.check(_pytorch_config(), "")

    def test_custom_map_overrides_defaults(self) -> None:
        """Tests / downstream packages can inject a custom map."""
        custom_map = {"experimental_engine": frozenset({TrainingFramework.PYTORCH})}
        checker = CompatibilityChecker(engine_framework_map=custom_map)
        assert checker.known_engines == {"experimental_engine"}
        report = checker.check(_pytorch_config(), "experimental_engine")
        assert report.is_compatible
        # And defaults are NOT silently merged in.
        report2 = checker.check(_pytorch_config(), "mujoco")
        assert report2.is_compatible is False

    def test_error_message_mentions_supported_frameworks(self) -> None:
        checker = CompatibilityChecker()
        report = checker.check(_gymnasium_config(), "drake")
        msg = " ".join(i.message for i in report.errors)
        assert "pytorch" in msg.lower()
