"""
Tests for launcher diagnostics module.

Tests cover:
- Model registry verification
- Tile loading diagnostics
- Asset file verification
- Layout configuration validation
- Engine availability checking
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

# Try to import the launcher diagnostics module
try:
    from src.launchers.launcher_diagnostics import (
        DiagnosticResult,
        LauncherDiagnostics,
        reset_layout_config,
        run_cli_diagnostics,
    )
except ImportError as e:
    pytest.skip(
        f"Cannot import launcher_diagnostics module: {e}", allow_module_level=True
    )


@pytest.fixture(autouse=True)
def _reset_structured_logging() -> Generator[None, None, None]:
    """Reset the global structured-logging flag before every test.

    ``setup_structured_logging()`` in ``src.shared.python.core._core`` is
    guarded by a module-level ``_structured_logging_configured`` flag.  When
    other tests in the full suite import modules that trigger this function
    (e.g. ``EngineManager``), the flag stays ``True`` for the rest of the
    process, which can change logging behaviour and cause spurious failures
    for tests that indirectly invoke the same code path.

    Resetting the flag here ensures every test starts from the same state.
    """
    try:
        import src.shared.python.core._core as _core_mod

        saved = _core_mod._structured_logging_configured
        _core_mod._structured_logging_configured = False
        yield
        _core_mod._structured_logging_configured = saved
    except (ImportError, AttributeError):
        yield


class TestDiagnosticResult:
    """Tests for the DiagnosticResult dataclass."""

    def test_diagnostic_result_creation(self) -> None:
        """Test creating a DiagnosticResult."""
        result = DiagnosticResult(
            name="test_check",
            status="pass",
            message="Test passed",
            details={"key": "value"},
            duration_ms=1.5,
        )
        assert result.name == "test_check", (
            "Assertion failed: result.name == test_check"
        )
        assert result.status == "pass", "Assertion failed: result.status == pass"
        assert result.message == "Test passed", (
            "Assertion failed: result.message == Test passed"
        )
        assert result.details == {"key": "value"}, (
            "Assertion failed: result.details == {key: value}"
        )
        assert result.duration_ms == 1.5, "Assertion failed: result.duration_ms == 1.5"

    def test_diagnostic_result_to_dict(self) -> None:
        """Test converting DiagnosticResult to dictionary."""
        result = DiagnosticResult(
            name="test_check",
            status="warning",
            message="Warning message",
            details={"warning_code": 123},
            duration_ms=2.567,
        )
        d = result.to_dict()
        assert d["name"] == "test_check", "Assertion failed: d[name] == test_check"
        assert d["status"] == "warning", "Assertion failed: d[status] == warning"
        assert d["message"] == "Warning message", (
            "Assertion failed: d[message] == Warning message"
        )
        assert d["details"]["warning_code"] == 123, (
            "Assertion failed: d[details][warning_code] == 123"
        )
        assert (
            d["duration_ms"] == 2.57
        )  # Rounded to 2 decimal places, "Assertion failed: d[duration_ms] == 2.57  # Rounded to 2 decimal places"


class TestLauncherDiagnostics:
    """Tests for the LauncherDiagnostics class."""

    def test_expected_tile_ids(self) -> None:
        """Test that expected tile IDs are defined."""
        diag = LauncherDiagnostics()
        expected_ids = diag.EXPECTED_TILE_IDS

        assert len(expected_ids) == 17, "Assertion failed: len(expected_ids) == 17"
        assert "mujoco_unified" in expected_ids, (
            "Assertion failed: mujoco_unified in expected_ids"
        )
        assert "drake_golf" in expected_ids, (
            "Assertion failed: drake_golf in expected_ids"
        )
        assert "pinocchio_golf" in expected_ids, (
            "Assertion failed: pinocchio_golf in expected_ids"
        )
        assert "opensim_golf" in expected_ids, (
            "Assertion failed: opensim_golf in expected_ids"
        )
        assert "myosim_suite" in expected_ids, (
            "Assertion failed: myosim_suite in expected_ids"
        )
        assert "putting_green" in expected_ids, (
            "Assertion failed: putting_green in expected_ids"
        )
        assert "model_explorer" in expected_ids, (
            "Assertion failed: model_explorer in expected_ids"
        )
        assert "c3d_viewer" in expected_ids, (
            "Assertion failed: c3d_viewer in expected_ids"
        )

    def test_expected_tile_names(self) -> None:
        """Test that expected tile names are defined correctly."""
        diag = LauncherDiagnostics()
        names = diag.EXPECTED_TILE_NAMES

        assert names["mujoco_unified"] == "MuJoCo", (
            "Assertion failed: names[mujoco_unified] == MuJoCo"
        )
        assert names["drake_golf"] == "Drake", (
            "Assertion failed: names[drake_golf] == Drake"
        )
        assert names["putting_green"] == "Putting Green", (
            "Assertion failed: names[putting_green] == Putting Green"
        )

    def test_diagnostics_initialization(self) -> None:
        """Test LauncherDiagnostics initialization."""
        diag = LauncherDiagnostics()
        assert diag.results == [], "Assertion failed: diag.results == []"

    def test_run_all_checks_returns_summary(self) -> None:
        """Test that run_all_checks returns proper summary."""
        diag = LauncherDiagnostics()
        results = diag.run_all_checks()

        summary = results["summary"]
        assert "total_checks" in summary, "Assertion failed: total_checks in summary"
        assert "passed" in summary, "Assertion failed: passed in summary"
        assert "failed" in summary, "Assertion failed: failed in summary"
        assert "warnings" in summary, "Assertion failed: warnings in summary"
        assert "status" in summary, "Assertion failed: status in summary"
        assert "timestamp" in summary, "Assertion failed: timestamp in summary"
        assert "expected_tiles" in summary, (
            "Assertion failed: expected_tiles in summary"
        )
        assert summary["expected_tiles"] == 17, (
            "Assertion failed: summary[expected_tiles] == 17"
        )

        # Verify counts add up
        assert (
            summary["passed"] + summary["failed"] + summary["warnings"]
            == summary["total_checks"]
        )

    def test_check_python_environment(self) -> None:
        """Test Python environment check."""
        diag = LauncherDiagnostics()
        result = diag.check_python_environment()

        assert result.name == "python_environment", (
            "Assertion failed: result.name == python_environment"
        )
        assert result.status == "pass", "Assertion failed: result.status == pass"
        assert "python_version" in result.details, (
            "Assertion failed: python_version in result.details"
        )
        assert "platform" in result.details, (
            "Assertion failed: platform in result.details"
        )
        assert "repos_root" in result.details, (
            "Assertion failed: repos_root in result.details"
        )

    def test_check_models_yaml(self) -> None:
        """Test models.yaml configuration check."""
        diag = LauncherDiagnostics()
        result = diag.check_models_yaml()

        assert result.name == "models_yaml", (
            "Assertion failed: result.name == models_yaml"
        )
        assert "path" in result.details, "Assertion failed: path in result.details"
        assert "exists" in result.details, "Assertion failed: exists in result.details"

        # If the file exists, more details should be present
        if result.details.get("exists"):
            assert "model_count" in result.details or result.status == "fail", (
                "Assertion failed: model_count in result.details or result.status == fail"
            )

    def test_check_model_registry(self) -> None:
        """Test ModelRegistry loading check."""
        diag = LauncherDiagnostics()
        result = diag.check_model_registry()

        assert result.name == "model_registry", (
            "Assertion failed: result.name == model_registry"
        )
        # Status depends on whether registry can be loaded
        assert result.status in ("pass", "warning", "fail"), (
            "Assertion failed: result.status in (pass, warning, fail)"
        )

    def test_check_layout_config_no_file(self) -> None:
        """Test layout config check when no config file exists."""
        diag = LauncherDiagnostics()

        with patch(
            "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE"
        ) as mock_path:
            mock_path.exists.return_value = False
            result = diag.check_layout_config()

        assert result.name == "layout_config", (
            "Assertion failed: result.name == layout_config"
        )
        # Should pass with no saved layout (uses defaults)
        assert result.status == "pass", "Assertion failed: result.status == pass"
        assert "will use defaults" in result.message.lower(), (
            "Assertion failed: will use defaults in result.message.lower()"
        )

    def test_check_layout_config_with_file(self) -> None:
        """Test layout config check with existing config file."""
        diag = LauncherDiagnostics()

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            layout_data = {
                "model_order": ["mujoco_unified", "drake_golf", "pinocchio_golf"],
            }
            json.dump(layout_data, f)
            temp_path = Path(f.name)

        try:
            with patch(
                "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE", temp_path
            ):
                result = diag.check_layout_config()

            assert result.name == "layout_config", (
                "Assertion failed: result.name == layout_config"
            )
            # Should warn that layout is missing some tiles
            assert result.status == "warning", (
                "Assertion failed: result.status == warning"
            )
            assert "saved_model_order" in result.details, (
                "Assertion failed: saved_model_order in result.details"
            )
            assert len(result.details["saved_model_order"]) == 3, (
                "Assertion failed: len(result.details[saved_model_order]) == 3"
            )
        finally:
            temp_path.unlink()

    def test_check_layout_config_invalid_json(self) -> None:
        """Test layout config check with invalid JSON."""
        diag = LauncherDiagnostics()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)

        try:
            with patch(
                "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE", temp_path
            ):
                result = diag.check_layout_config()

            assert result.name == "layout_config", (
                "Assertion failed: result.name == layout_config"
            )
            assert result.status == "warning", (
                "Assertion failed: result.status == warning"
            )
            assert (
                "json_error" in result.details or "invalid" in result.message.lower()
            ), (
                "Assertion failed: json_error in result.details or invalid in result.message.lower()"
            )
        finally:
            temp_path.unlink()

    def test_check_asset_files(self) -> None:
        """Test asset files check."""
        diag = LauncherDiagnostics()
        result = diag.check_asset_files()

        assert result.name == "asset_files", (
            "Assertion failed: result.name == asset_files"
        )
        assert "assets_dir" in result.details, (
            "Assertion failed: assets_dir in result.details"
        )
        assert "assets_dir_exists" in result.details, (
            "Assertion failed: assets_dir_exists in result.details"
        )

        if result.details.get("assets_dir_exists"):
            assert "found_assets" in result.details, (
                "Assertion failed: found_assets in result.details"
            )
            assert "missing_assets" in result.details, (
                "Assertion failed: missing_assets in result.details"
            )

    def test_check_pyqt6_availability(self) -> None:
        """Test PyQt6 availability check."""
        diag = LauncherDiagnostics()
        result = diag.check_pyqt6_availability()

        assert result.name == "pyqt6_availability", (
            "Assertion failed: result.name == pyqt6_availability"
        )
        # Status depends on whether PyQt6 is installed
        assert result.status in ("pass", "fail"), (
            "Assertion failed: result.status in (pass, fail)"
        )

    def test_check_engine_availability(self) -> None:
        """Test engine availability check."""
        diag = LauncherDiagnostics()
        result = diag.check_engine_availability()

        assert result.name == "engine_availability", (
            "Assertion failed: result.name == engine_availability"
        )
        # Status depends on engine availability
        assert result.status in ("pass", "warning", "fail"), (
            "Assertion failed: result.status in (pass, warning, fail)"
        )

    def test_recommendations_generation(self) -> None:
        """Test that recommendations are generated based on results."""
        diag = LauncherDiagnostics()
        results = diag.run_all_checks()

        assert "recommendations" in results, (
            "Assertion failed: recommendations in results"
        )
        assert isinstance(results["recommendations"], list), (
            "Assertion failed: isinstance(results[recommendations], list)"
        )
        assert len(results["recommendations"]) > 0, (
            "Assertion failed: len(results[recommendations]) > 0"
        )

    def test_recommendations_for_layout_issues(self) -> None:
        """Test recommendations are generated for layout config issues."""
        diag = LauncherDiagnostics()

        # Add a warning result for layout_config
        warning_result = DiagnosticResult(
            name="layout_config",
            status="warning",
            message="Layout missing tiles",
            details={"missing_from_saved": ["mujoco_unified"]},
        )
        diag.results = [warning_result]

        recommendations = diag._generate_recommendations()
        assert any("layout" in rec.lower() for rec in recommendations), (
            "Assertion failed: any(layout in rec.lower() for rec in recommendations)"
        )


class TestLauncherDiagnosticsPerformance:
    """Performance tests for launcher diagnostics."""

    def test_diagnostics_complete_in_reasonable_time(self) -> None:
        """Test that all diagnostics complete within reasonable time."""
        diag = LauncherDiagnostics()

        start = time.time()
        results = diag.run_all_checks()
        elapsed = time.time() - start

        # All checks should complete within 10 seconds
        assert elapsed < 10.0, f"Diagnostics took too long: {elapsed:.2f}s"

        # Verify timing data is captured
        for check in results["checks"]:
            assert "duration_ms" in check, "Assertion failed: duration_ms in check"
            assert check["duration_ms"] >= 0, (
                "Assertion failed: check[duration_ms] >= 0"
            )

    def test_individual_check_performance(self) -> None:
        """Test individual check performance."""
        diag = LauncherDiagnostics()

        checks = [
            diag.check_python_environment,
            diag.check_models_yaml,
            diag.check_asset_files,
        ]

        for check_func in checks:
            start = time.time()
            result = check_func()
            elapsed = time.time() - start

            # Each check should complete within 2 seconds
            assert elapsed < 2.0, f"{result.name} took too long: {elapsed:.2f}s"


class TestResetLayoutConfig:
    """Tests for the reset_layout_config function."""

    def test_reset_nonexistent_config(self) -> None:
        """Test resetting when no config file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "nonexistent.json"
            with patch(
                "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE", config_file
            ):
                result = reset_layout_config()
                assert result is True, "Assertion failed: result is True"

    def test_reset_existing_config(self) -> None:
        """Test resetting when config file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "layout.json"
            config_file.write_text('{"model_order": ["test"]}')

            with patch(
                "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE", config_file
            ):
                result = reset_layout_config()
                assert result is True, "Assertion failed: result is True"

                # Original file should be renamed to .bak
                assert not config_file.exists(), (
                    "Assertion failed: not config_file.exists()"
                )
                backup_file = config_file.with_suffix(".json.bak")
                assert backup_file.exists(), "Assertion failed: backup_file.exists()"


class TestCLIDiagnostics:
    """Tests for CLI diagnostic output."""

    def test_run_cli_diagnostics_no_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test CLI diagnostics runs without errors.

        run_cli_diagnostics() emits output via the ``logging`` module
        (logger.info / logger.error / logger.warning), not via print().
        We therefore capture with ``caplog`` rather than ``capsys``.
        """
        with caplog.at_level(logging.INFO, logger="src.launchers.launcher_diagnostics"):
            run_cli_diagnostics()

        log_text = caplog.text
        assert "Golf Modeling Suite" in log_text, (
            "Assertion failed: Golf Modeling Suite in log_text"
        )
        assert "Status:" in log_text, "Assertion failed: Status: in log_text"
        assert "Recommendations:" in log_text, (
            "Assertion failed: Recommendations: in log_text"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
