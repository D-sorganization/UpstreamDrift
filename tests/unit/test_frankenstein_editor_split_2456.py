"""Contract tests for #2456: frankenstein_editor.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
EDITOR_DIR = REPO / "src/tools/model_explorer"
LOC_BUDGET = 700


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestFrankensteinEditorSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_model_module_exists(self) -> None:
        assert (EDITOR_DIR / "_frankenstein_model.py").exists()

    @pytest.mark.unit
    def test_panels_module_exists(self) -> None:
        assert (EDITOR_DIR / "_frankenstein_panels.py").exists()


class TestFrankensteinEditorFileSizes:
    """Each file must be under 700 LOC after split."""

    @pytest.mark.unit
    def test_frankenstein_editor_split_2456_coordinator_loc(self) -> None:
        loc = _count_lines(EDITOR_DIR / "frankenstein_editor.py")
        assert loc <= LOC_BUDGET, (
            f"frankenstein_editor.py has {loc} LOC; budget {LOC_BUDGET}"
        )

    @pytest.mark.unit
    def test_model_loc(self) -> None:
        loc = _count_lines(EDITOR_DIR / "_frankenstein_model.py")
        assert loc <= LOC_BUDGET, (
            f"_frankenstein_model.py has {loc} LOC; budget {LOC_BUDGET}"
        )

    @pytest.mark.unit
    def test_panels_loc(self) -> None:
        loc = _count_lines(EDITOR_DIR / "_frankenstein_panels.py")
        assert loc <= LOC_BUDGET, (
            f"_frankenstein_panels.py has {loc} LOC; budget {LOC_BUDGET}"
        )


class TestFrankensteinEditorPublicAPI:
    """Public API must remain importable from frankenstein_editor (backward compat)."""

    @pytest.mark.unit
    def test_import_urdf_model(self) -> None:
        from src.tools.model_explorer.frankenstein_editor import URDFModel

        assert URDFModel is not None

    @pytest.mark.unit
    def test_import_model_panel(self) -> None:
        from src.tools.model_explorer.frankenstein_editor import ModelPanel

        assert ModelPanel is not None

    @pytest.mark.unit
    def test_import_frankenstein_editor(self) -> None:
        from src.tools.model_explorer.frankenstein_editor import FrankensteinEditor

        assert FrankensteinEditor is not None

    @pytest.mark.unit
    def test_import_steal_component_dialog(self) -> None:
        from src.tools.model_explorer.frankenstein_editor import StealComponentDialog

        assert StealComponentDialog is not None
