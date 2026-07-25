"""Regression: the Setup Wizard tile must actually open something (issue #8067).

The launcher reported the toast ``Setup Wizard Launched`` while no window or
tab ever appeared. Two defects stacked up:

1. ``models.yaml`` points the tile at ``_embed_adapter.py``, but that module
   exposed no module-level ``get_dockable_ui``, which is what
   ``SpecialAppHandler.get_dockable_ui`` looks for. Embedding silently
   returned ``None`` and the launcher fell back to spawning the adapter as a
   script -- which only re-registers the adapter and exits.
2. Both the by-path load and the script run give the module no parent
   package, so its ``from .gui import ...`` raised
   ``ImportError: attempted relative import with no known parent package``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = REPO_ROOT / "src" / "config" / "models.yaml"
TOOL_ID = "config_setup_wizard"


def _tile_entry() -> dict:
    models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    for entry in models["models"]:
        if entry.get("id") == TOOL_ID:
            return entry
    pytest.fail(f"{TOOL_ID} tile missing from models.yaml")


def _load_tile_module_by_path():
    """Load the tile target the way ``SpecialAppHandler`` does."""
    script_path = REPO_ROOT / _tile_entry()["path"]
    assert script_path.exists(), f"tile path does not exist: {script_path}"

    spec = importlib.util.spec_from_file_location("embed_adapter", str(script_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tile_target_exposes_dockable_ui_factory() -> None:
    """The launcher's by-path load must find a ``get_dockable_ui`` callable."""
    module = _load_tile_module_by_path()

    assert callable(getattr(module, "get_dockable_ui", None))


def test_tile_target_builds_a_widget_without_parent_package(qapp) -> None:
    """Building the widget must not need the module's parent package."""
    module = _load_tile_module_by_path()

    widget = module.get_dockable_ui()
    try:
        assert widget is not None
        assert type(widget).__name__ == "ConfigSetupWizardWidget"
    finally:
        widget.deleteLater()


def test_tile_target_has_a_standalone_entry_point() -> None:
    """The subprocess fallback must have a ``__main__`` body to run.

    Without one the spawned process exits immediately and the launcher's
    "Launched" toast describes nothing.
    """
    source = (REPO_ROOT / _tile_entry()["path"]).read_text(encoding="utf-8")

    assert '__name__ == "__main__"' in source
    assert "_run_standalone" in source


def test_adapter_create_main_widget_uses_absolute_gui_import() -> None:
    """Pin the absolute import: a relative one breaks both launch paths."""
    source = (REPO_ROOT / _tile_entry()["path"]).read_text(encoding="utf-8")

    assert "from src.tools.config_setup_wizard.gui import" in source
    assert "from .gui import" not in source
