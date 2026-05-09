"""Contract tests for #2456: text_editor.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
EDITOR_DIR = REPO / "src/shared/python/model_generation/editor"
LOC_BUDGET = 700


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestTextEditorSplitStructure:
    """Split modules must exist after refactor."""

    @pytest.mark.unit
    def test_text_editor_split_2456_models_module_exists(self) -> None:
        assert (EDITOR_DIR / "_text_editor_models.py").exists()

    @pytest.mark.unit
    def test_validation_module_exists(self) -> None:
        assert (EDITOR_DIR / "_text_editor_validation.py").exists()


class TestTextEditorFileSizes:
    """Each file must be under 700 LOC after split."""

    @pytest.mark.unit
    def test_text_editor_split_2456_coordinator_loc(self) -> None:
        loc = _count_lines(EDITOR_DIR / "text_editor.py")
        assert loc <= LOC_BUDGET, f"text_editor.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_text_editor_split_2456_models_loc(self) -> None:
        loc = _count_lines(EDITOR_DIR / "_text_editor_models.py")
        assert loc <= LOC_BUDGET, (
            f"_text_editor_models.py has {loc} LOC; budget {LOC_BUDGET}"
        )

    @pytest.mark.unit
    def test_validation_loc(self) -> None:
        loc = _count_lines(EDITOR_DIR / "_text_editor_validation.py")
        assert loc <= LOC_BUDGET, (
            f"_text_editor_validation.py has {loc} LOC; budget {LOC_BUDGET}"
        )


class TestTextEditorPublicAPI:
    """Public API must remain importable from text_editor (backward compat)."""

    @pytest.mark.unit
    def test_import_validation_severity(self) -> None:
        from src.shared.python.model_generation.editor.text_editor import (
            ValidationSeverity,
        )

        assert ValidationSeverity is not None

    @pytest.mark.unit
    def test_import_validation_message(self) -> None:
        from src.shared.python.model_generation.editor.text_editor import (
            ValidationMessage,
        )

        assert ValidationMessage is not None

    @pytest.mark.unit
    def test_import_diff_hunk(self) -> None:
        from src.shared.python.model_generation.editor.text_editor import DiffHunk

        assert DiffHunk is not None

    @pytest.mark.unit
    def test_import_diff_result(self) -> None:
        from src.shared.python.model_generation.editor.text_editor import DiffResult

        assert DiffResult is not None

    @pytest.mark.unit
    def test_import_editor_version(self) -> None:
        from src.shared.python.model_generation.editor.text_editor import EditorVersion

        assert EditorVersion is not None

    @pytest.mark.unit
    def test_import_urdf_text_editor(self) -> None:
        from src.shared.python.model_generation.editor.text_editor import URDFTextEditor

        assert URDFTextEditor is not None
