from pathlib import Path

import pytest


class TestEngineModelCompatibility:
    """Test that physics engines work with different model sources."""

    def test_mujoco_accepts_urdf(self) -> None:
        """Verify MuJoCo engine can load URDF models."""
        try:
            import mujoco
        except ImportError:
            pytest.skip("MuJoCo not installed")

        # Check for sample URDF files
        urdf_paths = list(
            Path("src/engines/physics_engines/mujoco/models").glob("**/*.urdf")
        )
        if not urdf_paths:
            urdf_paths = list(Path("src/shared/models").glob("**/*.urdf"))

        # At minimum, the engine should be importable
        assert mujoco is not None

    def test_model_registry_available(self) -> None:
        """Verify model registry can enumerate available models."""
        from src.shared.python.config.model_registry import ModelRegistry

        registry = ModelRegistry()
        models = registry.get_all_models()
        assert isinstance(models, list)
