"""Integration tests for Tools repository integration."""

from pathlib import Path

import numpy as np
import pytest

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


class TestToolsRepoIntegration:
    """Test integration with Tools repository packages."""

    def test_model_generation_import_contract(self) -> None:
        """Verify model_generation imports cleanly as a required contract."""
        import model_generation

        assert hasattr(model_generation, "quick_urdf")
        assert getattr(model_generation, "DEFAULT_HEIGHT_M", 0) > 0

    def test_signal_toolkit_compatibility(self) -> None:
        """Verify signal_toolkit is present and compatible as a required contract."""
        import signal_toolkit

        t = np.linspace(0, 1, 100)
        signal = signal_toolkit.SignalGenerator.sinusoid(
            t, amplitude=1.0, frequency=5.0
        )
        assert len(signal.values) == len(t)

    def test_humanoid_builder_compatibility(self) -> None:
        """Verify humanoid_character_builder is present and compatible as a required contract."""
        import humanoid_character_builder

        params = humanoid_character_builder.BodyParameters(height_m=1.75, mass_kg=70.0)
        assert params.height_m == 1.75
        assert params.mass_kg == 70.0


class TestCrossRepoImportPaths:
    """Test that import paths are correctly configured."""

    def test_pyproject_documents_tools_dependency(self) -> None:
        """Verify pyproject.toml documents Tools integration."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        content = pyproject_path.read_text()
        # Check for Tools repo reference
        assert (
            "Tools" in content or "tools" in content.lower()
        ), "Tools integration not documented in pyproject.toml"
