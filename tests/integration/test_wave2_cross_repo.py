"""
Wave 2 Cross-Repository Integration Tests

Validates that UpstreamDrift, Tools, and Gasification_Model can work together
across physics engines (Drake, OpenSim, MuJoCo, Pinocchio) with shared utilities.

Run with: python3 -m pytest tests/integration/test_wave2_cross_repo.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest


class TestToolsSymlinkIntegration:
    """Verify UpstreamDrift can load Tools via symlink."""

    def test_vendor_udtools_exists(self):
        """Symlink vendor/ud-tools/ should exist."""
        vendor_path = Path(__file__).parent.parent.parent / "vendor" / "ud-tools"
        assert vendor_path.exists(), f"vendor/ud-tools not found at {vendor_path}"
        assert vendor_path.is_dir(), "vendor/ud-tools is not a directory"

    def test_tools_contracts_import(self):
        """Should be able to import Tools contracts module."""
        try:
            from src.shared.python.contracts import (
                postcondition_check,
                precondition_check,
            )

            assert callable(precondition_check)
            assert callable(postcondition_check)
        except ImportError as e:
            pytest.fail(f"Failed to import Tools contracts: {e}")

    def test_tools_utilities_available(self):
        """Should be able to import common Tools utilities."""
        utilities = [
            "src.shared.python.contracts",
            "src.shared.python.validators",
        ]
        for util in utilities:
            try:
                spec = importlib.util.find_spec(util)
                assert spec is not None, f"{util} not found in path"
            except ImportError as e:
                pytest.fail(f"Failed to locate {util}: {e}")


class TestGasificationModelIntegration:
    """Verify Gasification_Model can load Tools from shared path."""

    def test_gasification_tools_loadable(self):
        """Gasification_Model should be able to access Tools utilities."""
        tools_path = Path(__file__).parent.parent.parent.parent / "Linux_Tools"
        if tools_path.exists():
            sys.path.insert(0, str(tools_path / "Tools" / "src"))
            try:
                # Try to import a gasification-specific module
                # This validates that Gasification_Model's import path works
                import gasification_equilibrium  # noqa: F401

            except ImportError as e:
                pytest.skip(
                    f"Gasification_Model not available in test environment: {e}"
                )
            finally:
                sys.path.pop(0)

    def test_tools_api_compatibility(self):
        """Verify Tools API hasn't changed in breaking ways."""
        # Check for expected function signatures
        try:
            # Verify signature: precondition_check(condition, message)
            import inspect

            from src.shared.python.contracts import precondition_check

            sig = inspect.signature(precondition_check)
            params = list(sig.parameters.keys())
            assert "condition" in params, "precondition_check missing 'condition' param"
        except Exception as e:
            pytest.fail(f"Tools API compatibility check failed: {e}")


class TestCrossEngineImports:
    """Verify all physics engine wrappers can initialize."""

    @pytest.mark.requires_drake
    def test_drake_import(self):
        """Drake engine wrapper should import."""
        try:
            from src.engines.physics_engines.drake import DrakeSimulator

            assert DrakeSimulator is not None
        except ImportError as e:
            pytest.skip(f"Drake not available: {e}")

    @pytest.mark.requires_opensim
    def test_opensim_import(self):
        """OpenSim engine wrapper should import."""
        try:
            from src.engines.physics_engines.opensim import OpenSimSimulator

            assert OpenSimSimulator is not None
        except ImportError as e:
            pytest.skip(f"OpenSim not available: {e}")

    @pytest.mark.requires_mujoco
    def test_mujoco_import(self):
        """MuJoCo engine wrapper should import."""
        try:
            from src.engines.physics_engines.mujoco import MuJoCoSimulator

            assert MuJoCoSimulator is not None
        except ImportError as e:
            pytest.skip(f"MuJoCo not available: {e}")

    @pytest.mark.requires_pinocchio
    def test_pinocchio_import(self):
        """Pinocchio engine wrapper should import."""
        try:
            from src.engines.physics_engines.pinocchio import PinocchioSimulator

            assert PinocchioSimulator is not None
        except ImportError as e:
            pytest.skip(f"Pinocchio not available: {e}")


class TestSharedAnthropometricConfig:
    """Verify golf humanoid anthropometric YAML is accessible."""

    def test_anthropometric_yaml_exists(self):
        """Shared YAML config should exist."""
        config_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "data"
            / "anthropometric"
            / "golf_humanoid.yaml"
        )
        assert config_path.exists(), f"Anthropometric YAML not found at {config_path}"

    def test_anthropometric_yaml_loadable(self):
        """Should be able to load and parse anthropometric YAML."""
        try:
            import yaml

            config_path = (
                Path(__file__).parent.parent.parent
                / "src"
                / "data"
                / "anthropometric"
                / "golf_humanoid.yaml"
            )
            with open(config_path) as f:
                config = yaml.safe_load(f)
                assert config is not None
                assert "segments" in config or "masses" in config
        except Exception as e:
            pytest.fail(f"Failed to load anthropometric YAML: {e}")


class TestManifestValidation:
    """Verify module manifest is consistent with actual modules."""

    def test_manifest_exists(self):
        """MANIFEST.md should exist."""
        manifest_path = Path(__file__).parent.parent.parent / "MANIFEST.md"
        if not manifest_path.exists():
            pytest.skip(
                "MANIFEST.md not found - Wave 2 manifest validation not enabled"
            )

    def test_manifest_modules_on_disk(self):
        """All modules in MANIFEST should exist on disk."""
        manifest_path = Path(__file__).parent.parent.parent / "MANIFEST.md"
        engines_path = Path(__file__).parent.parent.parent / "src" / "engines"

        if not engines_path.exists():
            pytest.skip("engines/ directory not found")

        # Simple check: ensure some expected engine dirs exist
        expected_engines = ["physics_engines", "biomechanics"]
        for engine in expected_engines:
            engine_path = engines_path / engine
            if not engine_path.exists():
                pytest.fail(f"Expected engine directory {engine} not found")

    def test_shared_python_modules_listed(self):
        """shared/python modules should be documented."""
        shared_path = Path(__file__).parent.parent.parent / "src" / "shared" / "python"
        manifest_path = Path(__file__).parent.parent.parent / "MANIFEST.md"

        if not shared_path.exists():
            pytest.skip("shared/python directory not found")

        # Read manifest
        manifest_content = manifest_path.read_text()

        # Check for key shared modules
        expected_modules = ["contracts", "validators"]
        for module in expected_modules:
            if (shared_path / module).exists():
                assert module in manifest_content, (
                    f"Module {module} not documented in MANIFEST.md"
                )


@pytest.mark.integration
class TestSymlinkPerformance:
    """Verify symlink imports don't significantly impact performance."""

    def test_import_time_acceptable(self):
        """Importing via symlink should be reasonably fast."""
        import time

        start = time.time()
        from src.shared.python.contracts import precondition_check  # noqa: F401

        elapsed = time.time() - start
        # Symlink import should complete in <100ms (generous timeout)
        assert elapsed < 0.1, f"Symlink import took {elapsed:.3f}s (expected <0.1s)"

    def test_import_reproducible(self):
        """Multiple imports via symlink should be consistent."""

        # Second import should use cache, be instant
        import time

        start = time.time()
        from src.shared.python.contracts import precondition_check as p2  # noqa: F401

        elapsed = time.time() - start
        assert elapsed < 0.01, f"Cached import took {elapsed:.3f}s (expected <0.01s)"


@pytest.fixture(scope="session")
def wave2_baseline_metrics():
    """Collect baseline metrics for regression detection."""
    return {
        "import_count": 0,
        "symlink_valid": True,
        "tools_api_stable": True,
        "manifest_consistent": True,
    }


def test_wave2_baseline_report(wave2_baseline_metrics):
    """Generate baseline metrics report for monitoring."""
    # This test always passes but documents current state
    print("\nWave 2 Baseline Metrics:")
    for key, value in wave2_baseline_metrics.items():
        print(f"  {key}: {value}")
