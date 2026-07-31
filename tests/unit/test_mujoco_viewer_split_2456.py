"""Contract tests for #2456: mujoco_viewer.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
VIEWER_DIR = REPO / "src/tools/model_explorer"
LOC_BUDGET = 700


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestMuJoCoViewerSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_backend_module_exists(self) -> None:
        assert (VIEWER_DIR / "_mujoco_viewer_backend.py").exists()


class TestMuJoCoViewerFileSizes:
    """Each file must be under 700 LOC after split."""

    @pytest.mark.unit
    def test_mujoco_viewer_split_2456_coordinator_loc(self) -> None:
        loc = _count_lines(VIEWER_DIR / "mujoco_viewer.py")
        assert loc <= LOC_BUDGET, f"mujoco_viewer.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_backend_loc(self) -> None:
        loc = _count_lines(VIEWER_DIR / "_mujoco_viewer_backend.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_mujoco_viewer_backend.py has {loc} LOC; budget {LOC_BUDGET}"


class TestMuJoCoViewerPublicAPI:
    """Public API must remain importable from mujoco_viewer (backward compat)."""

    @pytest.mark.unit
    def test_import_visualization_flags(self) -> None:
        from src.tools.model_explorer.mujoco_viewer import VisualizationFlags

        assert VisualizationFlags is not None

    @pytest.mark.unit
    def test_import_urdf_to_mjcf_converter(self) -> None:
        from src.tools.model_explorer.mujoco_viewer import URDFToMJCFConverter

        assert URDFToMJCFConverter is not None

    @pytest.mark.unit
    def test_import_mujoco_offscreen_renderer(self) -> None:
        from src.tools.model_explorer.mujoco_viewer import MuJoCoOffscreenRenderer

        assert MuJoCoOffscreenRenderer is not None

    @pytest.mark.unit
    def test_import_mujoco_viewer_widget(self) -> None:
        from src.tools.model_explorer.mujoco_viewer import MuJoCoViewerWidget

        assert MuJoCoViewerWidget is not None
