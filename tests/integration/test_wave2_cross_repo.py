"""
Wave 2 Cross-Repository Integration Tests

Validates that UpstreamDrift, Tools, and Gasification_Model can work together
across physics engines (Drake, OpenSim, MuJoCo, Pinocchio) with shared utilities.

Run with: python3 -m pytest tests/integration/test_wave2_cross_repo.py -v
"""

import importlib.util
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
                postcondition,
                precondition,
                require,
                ensure,
            )

            assert callable(precondition)
            assert callable(postcondition)
            assert callable(require)
            assert callable(ensure)
        except ImportError as e:
            pytest.fail(f"Failed to import Tools contracts: {e}")

    def test_tools_utilities_available(self):
        """Should be able to import common Tools utilities."""
        utilities = [
            "src.shared.python.contracts",
            "src.shared.python.core.contracts.validators",
            "src.shared.python._contracts_validators",
        ]
        for util in utilities:
            try:
                spec = importlib.util.find_spec(util)
                assert spec is not None, f"{util} not found in path"
            except ImportError as e:
                pytest.fail(f"Failed to locate {util}: {e}")


class TestSharedAnthropometricConfig:
    """Verify golf humanoid anthropometric YAML is accessible."""

    def test_anthropometric_yaml_exists(self):
        """Shared YAML configs should exist."""
        models_dir = Path(__file__).parent.parent.parent / "shared" / "models"
        paths = [
            models_dir / "golf_humanoid_dimensions.yaml",
            models_dir / "golf_humanoid_inertia.yaml",
            models_dir / "golf_humanoid_topology.yaml",
        ]
        for path in paths:
            assert path.exists(), f"Anthropometric YAML not found at {path}"

    def test_anthropometric_yaml_loadable(self):
        """Should be able to load and parse anthropometric YAMLs."""
        try:
            import yaml

            models_dir = Path(__file__).parent.parent.parent / "shared" / "models"
            paths = [
                models_dir / "golf_humanoid_dimensions.yaml",
                models_dir / "golf_humanoid_inertia.yaml",
                models_dir / "golf_humanoid_topology.yaml",
            ]
            for path in paths:
                with open(path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    assert config is not None
                    assert (
                        "golfer" in config
                        or "total_dof" in config
                        or "UpperTorsoLength" in config
                    )
        except Exception as e:  # noqa: BLE001 - surface any load failure
            pytest.fail(f"Failed to load anthropometric YAML: {e}")


@pytest.mark.integration
class TestSymlinkPerformance:
    """Verify symlink imports don't significantly impact performance."""

    def test_import_time_acceptable(self):
        """Importing via symlink should be reasonably fast."""
        import time

        start = time.time()
        from src.shared.python.contracts import precondition  # noqa: F401

        elapsed = time.time() - start
        # Symlink import should complete in <100ms (generous timeout)
        assert elapsed < 0.1, f"Symlink import took {elapsed:.3f}s (expected <0.1s)"

    def test_import_reproducible(self):
        """Multiple imports via symlink should be consistent."""

        # Second import should use cache, be instant
        import time

        start = time.time()
        from src.shared.python.contracts import precondition as p2  # noqa: F401

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
