"""Contract tests for #2456: frankenstein_editor.py split.

Tests run red before the split and green after.
"""

from __future__ import annotations

from pathlib import Path

import defusedxml.ElementTree as ET
import pytest

REPO = Path(__file__).parents[2]
EDITOR_DIR = REPO / "src/tools/model_explorer"
EDITOR_COORDINATOR = EDITOR_DIR / "frankenstein_editor" / "editor.py"
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
        loc = _count_lines(EDITOR_COORDINATOR)
        assert (
            loc <= LOC_BUDGET
        ), f"{EDITOR_COORDINATOR.name} has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_model_loc(self) -> None:
        loc = _count_lines(EDITOR_DIR / "_frankenstein_model.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_frankenstein_model.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_panels_loc(self) -> None:
        loc = _count_lines(EDITOR_DIR / "_frankenstein_panels.py")
        assert (
            loc <= LOC_BUDGET
        ), f"_frankenstein_panels.py has {loc} LOC; budget {LOC_BUDGET}"


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


class TestFrankensteinLegacyCompatibility:
    """Legacy split modules must forward to the canonical package."""

    @pytest.mark.unit
    def test_legacy_model_import_is_canonical_model(self) -> None:
        from src.tools.model_explorer._frankenstein_model import (
            URDFModel as LegacyURDFModel,
        )
        from src.tools.model_explorer.frankenstein_editor.model import URDFModel

        assert LegacyURDFModel is URDFModel

    @pytest.mark.unit
    def test_legacy_panel_imports_are_canonical_panel_classes(self) -> None:
        from src.tools.model_explorer._frankenstein_panels import (
            ModelPanel as LegacyModelPanel,
        )
        from src.tools.model_explorer._frankenstein_panels import (
            StealComponentDialog as LegacyStealComponentDialog,
        )
        from src.tools.model_explorer.frankenstein_editor.dialogs import (
            StealComponentDialog,
        )
        from src.tools.model_explorer.frankenstein_editor.panel import ModelPanel

        assert LegacyModelPanel is ModelPanel
        assert LegacyStealComponentDialog is StealComponentDialog

    @pytest.mark.unit
    def test_legacy_model_uses_canonical_validation_contract(self) -> None:
        from src.tools.model_explorer._frankenstein_model import URDFModel

        root = ET.fromstring('<robot name="shim_contract" />')
        model = URDFModel.from_element(root)

        assert hasattr(model, "validate_composition")
        assert model.to_xml(force=True).startswith("<?xml")
