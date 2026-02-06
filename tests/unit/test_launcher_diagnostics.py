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
import tempfile
import time
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
        assert result.name == "test_check"
        assert result.status == "pass"
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert result.duration_ms == 1.5

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
        assert d["name"] == "test_check"
        assert d["status"] == "warning"
        assert d["message"] == "Warning message"
        assert d["details"]["warning_code"] == 123
        assert d["duration_ms"] == 2.57  # Rounded to 2 decimal places


class TestLauncherDiagnostics:
    """Tests for the LauncherDiagnostics class."""

    def test_expected_tile_ids(self) -> None:
        """Test that expected tile IDs are defined."""
        diag = LauncherDiagnostics()
        expected_ids = diag.EXPECTED_TILE_IDS

        assert len(expected_ids) == 8
        assert "mujoco_unified" in expected_ids
        assert "drake_golf" in expected_ids
        assert "pinocchio_golf" in expected_ids
        assert "opensim_golf" in expected_ids
        assert "myosim_suite" in expected_ids
        assert "matlab_unified" in expected_ids
        assert "motion_capture" in expected_ids
        assert "model_explorer" in expected_ids

    def test_expected_tile_names(self) -> None:
        """Test that expected tile names are defined correctly."""
        diag = LauncherDiagnostics()
        names = diag.EXPECTED_TILE_NAMES

        assert names["mujoco_unified"] == "MuJoCo"
        assert names["drake_golf"] == "Drake"
        assert names["matlab_unified"] == "Matlab Models"

    def test_diagnostics_initialization(self) -> None:
        """Test LauncherDiagnostics initialization."""
        diag = LauncherDiagnostics()
        assert diag.results == []

    def test_run_all_checks_returns_summary(self) -> None:
        """Test that run_all_checks returns proper summary."""
        diag = LauncherDiagnostics()
        results = diag.run_all_checks()

        summary = results["summary"]
        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "warnings" in summary
        assert "status" in summary
        assert "timestamp" in summary
        assert "expected_tiles" in summary
        assert summary["expected_tiles"] == 8

        # Verify counts add up
        assert (
            summary["passed"] + summary["failed"] + summary["warnings"]
            == summary["total_checks"]
        )

    def test_check_python_environment(self) -> None:
        """Test Python environment check."""
        diag = LauncherDiagnostics()
        result = diag.check_python_environment()

        assert result.name == "python_environment"
        assert result.status == "pass"
        assert "python_version" in result.details
        assert "platform" in result.details
        assert "repos_root" in result.details

    def test_check_models_yaml(self) -> None:
        """Test models.yaml configuration check."""
        diag = LauncherDiagnostics()
        result = diag.check_models_yaml()

        assert result.name == "models_yaml"
        assert "path" in result.details
        assert "exists" in result.details

        # If the file exists, more details should be present
        if result.details.get("exists"):
            assert "model_count" in result.details or result.status == "fail"

    def test_check_model_registry(self) -> None:
        """Test ModelRegistry loading check."""
        diag = LauncherDiagnostics()
        result = diag.check_model_registry()

        assert result.name == "model_registry"
        # Status depends on whether registry can be loaded
        assert result.status in ("pass", "warning", "fail")

    def test_check_layout_config_no_file(self) -> None:
        """Test layout config check when no config file exists."""
        diag = LauncherDiagnostics()

        with patch(
            "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE"
        ) as mock_path:
            mock_path.exists.return_value = False
            result = diag.check_layout_config()

        assert result.name == "layout_config"
        # Should pass with no saved layout (uses defaults)
        assert result.status == "pass"
        assert "will use defaults" in result.message.lower()

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

            assert result.name == "layout_config"
            # Should warn that layout is missing some tiles
            assert result.status == "warning"
            assert "saved_model_order" in result.details
            assert len(result.details["saved_model_order"]) == 3
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

            assert result.name == "layout_config"
            assert result.status == "warning"
            assert "json_error" in result.details or "invalid" in result.message.lower()
        finally:
            temp_path.unlink()

    def test_check_asset_files(self) -> None:
        """Test asset files check."""
        diag = LauncherDiagnostics()
        result = diag.check_asset_files()

        assert result.name == "asset_files"
        assert "assets_dir" in result.details
        assert "assets_dir_exists" in result.details

        if result.details.get("assets_dir_exists"):
            assert "found_assets" in result.details
            assert "missing_assets" in result.details

    def test_check_pyqt6_availability(self) -> None:
        """Test PyQt6 availability check."""
        diag = LauncherDiagnostics()
        result = diag.check_pyqt6_availability()

        assert result.name == "pyqt6_availability"
        # Status depends on whether PyQt6 is installed
        assert result.status in ("pass", "fail")

    def test_check_engine_availability(self) -> None:
        """Test engine availability check."""
        diag = LauncherDiagnostics()
        result = diag.check_engine_availability()

        assert result.name == "engine_availability"
        # Status depends on engine availability
        assert result.status in ("pass", "warning", "fail")

    def test_recommendations_generation(self) -> None:
        """Test that recommendations are generated based on results."""
        diag = LauncherDiagnostics()
        results = diag.run_all_checks()

        assert "recommendations" in results
        assert isinstance(results["recommendations"], list)
        assert len(results["recommendations"]) > 0

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
        assert any("layout" in rec.lower() for rec in recommendations)


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
            assert "duration_ms" in check
            assert check["duration_ms"] >= 0

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
                assert result is True

    def test_reset_existing_config(self) -> None:
        """Test resetting when config file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "layout.json"
            config_file.write_text('{"model_order": ["test"]}')

            with patch(
                "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE", config_file
            ):
                result = reset_layout_config()
                assert result is True

                # Original file should be renamed to .bak
                assert not config_file.exists()
                backup_file = config_file.with_suffix(".json.bak")
                assert backup_file.exists()


class TestTileLoadingVerification:
    """Tests specifically for tile loading verification."""

    def test_all_eight_tiles_in_expected_ids(self) -> None:
        """Verify all 8 expected tiles are defined."""
        diag = LauncherDiagnostics()
        assert len(diag.EXPECTED_TILE_IDS) == 8

    def test_tile_ids_match_models_yaml(self) -> None:
        """Verify expected tile IDs match models.yaml configuration."""
        try:
            import yaml

            from src.launchers.launcher_diagnostics import REPOS_ROOT

            models_yaml_path = REPOS_ROOT / "src" / "config" / "models.yaml"
            if not models_yaml_path.exists():
                pytest.skip("models.yaml not found")

            with open(models_yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data or "models" not in data:
                pytest.skip("models.yaml is empty or invalid")

            yaml_model_ids = {m.get("id") for m in data["models"]}
            expected_ids = set(LauncherDiagnostics.EXPECTED_TILE_IDS)

            # All expected IDs should be in the YAML
            missing = expected_ids - yaml_model_ids
            assert len(missing) == 0, f"Missing from models.yaml: {missing}"

        except ImportError:
            pytest.skip("yaml not available")

    def test_model_registry_loads_all_tiles(self) -> None:
        """Verify ModelRegistry loads all expected tiles."""
        try:
            from src.launchers.launcher_diagnostics import REPOS_ROOT
            from src.shared.python.model_registry import ModelRegistry

            registry_path = REPOS_ROOT / "src" / "config" / "models.yaml"
            if not registry_path.exists():
                pytest.skip("models.yaml not found")

            registry = ModelRegistry(registry_path)
            all_models = registry.get_all_models()

            loaded_ids = {m.id for m in all_models}
            expected_ids = set(LauncherDiagnostics.EXPECTED_TILE_IDS)

            missing = expected_ids - loaded_ids
            assert len(missing) == 0, f"Registry missing: {missing}"
            assert (
                len(all_models) >= 8
            ), f"Expected at least 8 models, got {len(all_models)}"

        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

    def test_saved_layout_with_all_tiles(self) -> None:
        """Test layout config detection with all tiles present."""
        diag = LauncherDiagnostics()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            layout_data = {
                "model_order": list(LauncherDiagnostics.EXPECTED_TILE_IDS),
            }
            json.dump(layout_data, f)
            temp_path = Path(f.name)

        try:
            with patch(
                "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE", temp_path
            ):
                result = diag.check_layout_config()

            assert result.status == "pass"
            assert result.details["saved_model_count"] == 8
        finally:
            temp_path.unlink()

    def test_saved_layout_with_missing_tiles(self) -> None:
        """Test layout config detection with missing tiles."""
        diag = LauncherDiagnostics()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Only 4 tiles - simulating the reported issue
            layout_data = {
                "model_order": [
                    "mujoco_unified",
                    "drake_golf",
                    "pinocchio_golf",
                    "opensim_golf",
                ],
            }
            json.dump(layout_data, f)
            temp_path = Path(f.name)

        try:
            with patch(
                "src.launchers.launcher_diagnostics.LAYOUT_CONFIG_FILE", temp_path
            ):
                result = diag.check_layout_config()

            assert result.status == "warning"
            assert result.details["saved_model_count"] == 4
            assert len(result.details["missing_from_saved"]) == 4
        finally:
            temp_path.unlink()


class TestCLIDiagnostics:
    """Tests for CLI diagnostic output."""

    def test_run_cli_diagnostics_no_errors(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test CLI diagnostics runs without errors."""
        # This should not raise any exceptions
        run_cli_diagnostics()

        captured = capsys.readouterr()
        assert "Golf Modeling Suite" in captured.out
        assert "Status:" in captured.out
        assert "Recommendations:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
