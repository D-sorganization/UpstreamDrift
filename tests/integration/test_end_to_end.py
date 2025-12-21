"""Integration tests for Golf Modeling Suite.

These tests verify end-to-end functionality without heavy mocking,
ensuring the suite works in real-world scenarios.
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestLauncherIntegration:
    """Integration tests for launcher functionality."""

    def test_launch_golf_suite_imports(self):
        """Verify launch_golf_suite can import UnifiedLauncher."""
        # Add project root to path
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root))

        from launchers.unified_launcher import UnifiedLauncher

        launcher = UnifiedLauncher()
        assert launcher is not None
        assert hasattr(launcher, "mainloop")
        assert hasattr(launcher, "show_status")

    def test_launch_golf_suite_status(self):
        """Test launch_golf_suite.py --status command."""
        suite_root = Path(__file__).parent.parent.parent
        script = suite_root / "launch_golf_suite.py"

        result = subprocess.run(
            [sys.executable, str(script), "--status"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should complete successfully
        assert result.returncode == 0

        # Should contain status information
        assert "Golf Modeling Suite" in result.stdout or "Engine" in result.stdout

    def test_unified_launcher_show_status(self):
        """Test UnifiedLauncher.show_status() method."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root))

        from launchers.unified_launcher import UnifiedLauncher

        launcher = UnifiedLauncher()

        # Should not raise exception
        # Note: This will print to stdout, which is expected
        launcher.show_status()


class TestEngineProbes:
    """Integration tests for engine probe system."""

    def test_engine_manager_probes_available(self):
        """Verify EngineManager has probe functionality."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from engine_manager import EngineManager

        manager = EngineManager()

        assert hasattr(manager, "probes")
        assert hasattr(manager, "probe_all_engines")
        assert hasattr(manager, "get_diagnostic_report")

    def test_probe_all_engines(self):
        """Test probing all engines."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from engine_manager import EngineManager
        from engine_probes import ProbeStatus

        manager = EngineManager()
        results = manager.probe_all_engines()

        # Should have results for all probed engines
        assert len(results) > 0

        # Each result should have required fields
        for _engine_type, result in results.items():
            assert result.engine_name is not None
            assert isinstance(result.status, ProbeStatus)
            assert result.diagnostic_message is not None

    def test_diagnostic_report_generation(self):
        """Test generating diagnostic report."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from engine_manager import EngineManager

        manager = EngineManager()
        report = manager.get_diagnostic_report()

        # Report should be non-empty string
        assert isinstance(report, str)
        assert len(report) > 0

        # Should contain engine information
        assert "Engine Readiness Report" in report or "MUJOCO" in report.upper()

    def test_at_least_one_engine_available(self):
        """Verify at least one engine is available or properly diagnosed."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from engine_manager import EngineManager
        from engine_probes import ProbeStatus

        manager = EngineManager()
        results = manager.probe_all_engines()

        # Check if any engine is available
        available_engines = [
            r for r in results.values() if r.status == ProbeStatus.AVAILABLE
        ]

        # If no engines available, at least verify we get proper diagnostics
        if not available_engines:
            for result in results.values():
                # Should have diagnostic message
                assert result.diagnostic_message
                # Should have fix instructions
                assert result.get_fix_instructions()


class TestPhysicsParameters:
    """Integration tests for physics parameter registry."""

    def test_physics_parameters_accessible(self):
        """Verify physics parameters are accessible."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from physics_parameters import get_registry

        registry = get_registry()

        # Should have parameters
        assert len(registry.parameters) > 0

    def test_ball_parameters_present(self):
        """Test that ball parameters are defined."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from physics_parameters import get_registry

        registry = get_registry()

        # Check for key ball parameters
        ball_mass = registry.get("BALL_MASS")
        assert ball_mass is not None
        assert ball_mass.value == 0.04593
        assert ball_mass.unit == "kg"

        ball_diameter = registry.get("BALL_DIAMETER")
        assert ball_diameter is not None
        assert ball_diameter.value == 0.04267
        assert ball_diameter.unit == "m"

    def test_gravity_parameter(self):
        """Test gravity parameter."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from physics_parameters import get_registry

        registry = get_registry()

        gravity = registry.get("GRAVITY")
        assert gravity is not None
        assert gravity.value == 9.80665
        assert gravity.unit == "m/s²"
        assert gravity.is_constant is True  # Should be constant

    def test_parameter_validation(self):
        """Test parameter validation."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from physics_parameters import get_registry

        registry = get_registry()

        # Try to set a valid value
        success, error = registry.set("CLUB_MASS", 0.350)
        assert success is True

        # Try to set invalid value (too low)
        success, error = registry.set("CLUB_MASS", 0.050)
        assert success is False
        assert "must be >=" in error

        # Try to set constant (should fail)
        success, error = registry.set("GRAVITY", 10.0)
        assert success is False
        assert "constant" in error.lower()

    def test_parameter_categories(self):
        """Test parameter categorization."""
        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from physics_parameters import ParameterCategory, get_registry

        registry = get_registry()

        # Get ball parameters
        ball_params = registry.get_by_category(ParameterCategory.BALL)
        assert len(ball_params) > 0

        # All should be ball category
        for param in ball_params:
            assert param.category == ParameterCategory.BALL

    def test_show_physics_parameters_script(self):
        """Test show_physics_parameters.py script."""
        suite_root = Path(__file__).parent.parent.parent
        script = suite_root / "show_physics_parameters.py"

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should complete successfully
        assert result.returncode == 0

        # Should contain parameter information
        assert "Physics Parameter Registry" in result.stdout
        assert "BALL" in result.stdout or "ball" in result.stdout.lower()


class TestValidateSuite:
    """Integration tests for validate_suite.py."""

    def test_validate_suite_runs(self):
        """Test that validate_suite.py runs without errors."""
        suite_root = Path(__file__).parent.parent.parent
        script = suite_root / "validate_suite.py"

        if not script.exists():
            pytest.skip("validate_suite.py not found")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should complete (may have warnings but shouldn't crash)
        assert result.returncode in [0, 1]  # 0=success, 1=issues found

        # Should contain validation information
        assert len(result.stdout) > 0


class TestOutputManager:
    """Integration tests for output management."""

    def test_output_manager_real_save_load(self):
        """Test OutputManager with real file I/O."""
        import tempfile

        suite_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(suite_root / "shared" / "python"))

        from output_manager import OutputFormat, OutputManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)

            # Save data
            data = {"time": [0, 1, 2], "angle": [0.0, 0.5, 1.0]}

            # Note: OutputManager may add timestamps, so we can't use
            # deterministic names yet
            # This is a known limitation documented in the architecture plan
            path = manager.save_simulation_results(
                data, "test_sim", format_type=OutputFormat.JSON, engine="test"
            )

            # Verify file was created
            assert path.exists()

            # Load data back
            # Note: This test may need adjustment based on actual OutputManager API
            # For now, we just verify the file exists


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
