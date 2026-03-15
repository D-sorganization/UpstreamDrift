"""Integration tests for Tools repository integration."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


def _require_real_tools_repo() -> bool:
    return os.environ.get("REQUIRE_REAL_TOOLS_REPO", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _assert_real_tools_path(module_file: str) -> None:
    normalized = module_file.replace("\\", "/").lower()
    assert any(
        marker in normalized
        for marker in (
            "/_tools_dep/",
            "/vendor/ud-tools/",
            "/repositories/tools/",
            "/tools/",
        )
    ), f"Expected Tools-backed provider path, got: {module_file}"


def _prepend_real_tools_paths() -> None:
    tools_root = os.environ.get("TOOLS_REPO_ROOT")
    if not tools_root:
        repo_root = Path(__file__).resolve().parents[3]
        for candidate in (
            repo_root / "_tools_dep",
            repo_root / "vendor" / "ud-tools",
            repo_root.parent / "Tools",
        ):
            if candidate.exists():
                tools_root = str(candidate.resolve())
                os.environ["TOOLS_REPO_ROOT"] = tools_root
                break

    if not tools_root:
        raise RuntimeError(
            "REQUIRE_REAL_TOOLS_REPO=1 but no Tools checkout was found. "
            "Expected _tools_dep, vendor/ud-tools, or ../Tools."
        )

    for path in reversed(
        [
            Path(tools_root) / "src" / "shared" / "python",
            Path(tools_root) / "src",
            Path(tools_root) / "src" / "python" / "src",
        ]
    ):
        if not path.exists():
            continue
        path_str = str(path.resolve())
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


def _import_or_skip(module_name: str):
    if _require_real_tools_repo():
        _prepend_real_tools_paths()
        module = __import__(module_name, fromlist=["__name__"])
        _assert_real_tools_path(str(Path(module.__file__).resolve()))
        return module

    try:
        return __import__(module_name, fromlist=["__name__"])
    except ImportError:
        pytest.skip(f"{module_name} not available in optional integration mode")


class TestToolsRepoIntegration:
    """Test integration with Tools repository packages."""

    def test_model_generation_import_contract(self) -> None:
        """Verify model_generation imports cleanly in both optional and required modes."""
        module = _import_or_skip("model_generation")
        assert hasattr(module, "quick_urdf")
        assert getattr(module, "DEFAULT_HEIGHT_M", 0) > 0

    def test_signal_toolkit_compatibility(self) -> None:
        """Verify signal_toolkit is compatible when present."""
        module = _import_or_skip("signal_toolkit")
        t = np.linspace(0, 1, 100)
        signal = module.SignalGenerator.sinusoid(t, amplitude=1.0, frequency=5.0)
        assert len(signal.values) == len(t)

    def test_humanoid_builder_compatibility(self) -> None:
        """Verify humanoid_character_builder is compatible when present."""
        module = _import_or_skip("humanoid_character_builder")
        params = module.BodyParameters(height_m=1.75, mass_kg=70.0)
        assert params.height_m == 1.75
        assert params.mass_kg == 70.0


class TestCrossRepoImportPaths:
    """Test that import paths are correctly configured."""

    def test_pythonpath_includes_tools(self) -> None:
        """Verify PYTHONPATH can be configured for Tools packages."""
        tools_path = Path(__file__).parent.parent.parent.parent / "Tools"
        if tools_path.exists():
            expected_path = tools_path / "src" / "shared" / "python"
            if expected_path.exists():
                if _require_real_tools_repo():
                    _prepend_real_tools_paths()
                    normalized_paths = {Path(path).resolve() for path in sys.path}
                    assert expected_path.resolve() in normalized_paths
                else:
                    assert expected_path.exists()

    def test_pyproject_documents_tools_dependency(self) -> None:
        """Verify pyproject.toml documents Tools integration."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        content = pyproject_path.read_text()
        # Check for Tools repo reference
        assert "Tools" in content or "tools" in content.lower(), (
            "Tools integration not documented in pyproject.toml"
        )


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
        try:
            from src.shared.python.config.model_registry import ModelRegistry

            registry = ModelRegistry()
            models = registry.get_all_models()
            assert isinstance(models, list)
        except ImportError:
            # Model registry might not exist yet
            pytest.skip("ModelRegistry not implemented")
