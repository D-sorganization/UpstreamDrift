"""Regression tests for engine path resolution.

Ensures engine_manager correctly resolves src/engines/ when suite_root
is the repo root (classic launcher) vs src/ (API server).
"""

from pathlib import Path
from unittest.mock import patch

from src.shared.python.engine_core.engine_manager import EngineManager
from src.shared.python.engine_core.engine_registry import EngineType


class TestEnginePathResolution:
    """Verify engines_root resolution logic in EngineManager."""

    def test_prefers_src_engines_when_available(self, tmp_path: Path) -> None:
        """When suite_root/src/engines exists, use it over suite_root/engines."""
        (tmp_path / "src" / "engines" / "physics_engines" / "mujoco").mkdir(
            parents=True,
        )
        (tmp_path / "engines" / "physics_engines").mkdir(parents=True)
        with patch(
            "src.shared.python.engine_core.engine_manager.get_src_root",
            return_value=tmp_path / "src",
        ):
            em = EngineManager(tmp_path)
        assert em.engines_root == tmp_path / "src" / "engines"

    def test_falls_back_to_direct_engines(self, tmp_path: Path) -> None:
        """When only suite_root/engines exists (no src/), use it directly."""
        (tmp_path / "engines" / "physics_engines" / "mujoco").mkdir(parents=True)
        with patch(
            "src.shared.python.engine_core.engine_manager.get_src_root",
            return_value=tmp_path,
        ):
            em = EngineManager(tmp_path)
        assert em.engines_root == tmp_path / "engines"

    def test_myosuite_path_correct(self, tmp_path: Path) -> None:
        """MYOSIM engine type maps to myosuite directory, not myosim."""
        (tmp_path / "engines" / "physics_engines" / "myosuite").mkdir(parents=True)
        with patch(
            "src.shared.python.engine_core.engine_manager.get_src_root",
            return_value=tmp_path,
        ):
            em = EngineManager(tmp_path)
        assert em.engine_paths[EngineType.MYOSIM].name == "myosuite"
