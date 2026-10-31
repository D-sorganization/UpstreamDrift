"""Desktop tile for the anatomical event table (HRG-E4 final slice, #13175)."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config.launcher_manifest_loader import LauncherManifest
from src.shared.python.launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    EmbeddableTool,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    ROOT
    / "ui/src/components/analysis/anatomicalEventTable/fixtures/eventTableView.json"
)
TILE = "anatomical_event_table"
_ADAPTER_MODULE = "src.tools.anatomical_event_table._embed_adapter"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def app():
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6.QtWidgets not loadable: {exc}")
    inst = QApplication.instance() or QApplication([])
    if isinstance(inst, MagicMock):
        pytest.skip("real PyQt6 required (found MagicMock stub)")
    return inst


def _cells(table) -> list[list[str]]:
    return [
        [table.item(r, c).text() for c in range(table.columnCount())]
        for r in range(table.rowCount())
    ]


def test_importing_adapter_registers_without_loading_pyqt6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    had_qt = "PyQt6" in sys.modules
    import src.tools.anatomical_event_table as pkg

    monkeypatch.delitem(sys.modules, _ADAPTER_MODULE, raising=False)
    if hasattr(pkg, "_embed_adapter"):
        monkeypatch.delattr(pkg, "_embed_adapter")
    monkeypatch.delitem(EMBEDDABLE_TOOL_REGISTRY, TILE, raising=False)
    importlib.import_module(_ADAPTER_MODULE)

    assert isinstance(EMBEDDABLE_TOOL_REGISTRY[TILE], EmbeddableTool)
    if not had_qt:
        assert "PyQt6" not in sys.modules


def test_adapter_cleanup_is_idempotent(app) -> None:
    from src.tools.anatomical_event_table._embed_adapter import (
        AnatomicalEventTableEmbedAdapter,
    )

    adapter = AnatomicalEventTableEmbedAdapter()
    assert adapter.tool_id == TILE
    assert adapter.is_dirty() is False
    assert adapter.create_main_widget(None) is adapter.create_main_widget(None)
    adapter.cleanup()
    adapter.cleanup()


def test_opened_export_shows_the_golden_cells(app, golden, tmp_path) -> None:
    from src.tools.anatomical_event_table.gui import AnatomicalEventTablePanel

    path = tmp_path / "events.json"
    path.write_text(json.dumps(golden["document"]), encoding="utf-8")
    panel = AnatomicalEventTablePanel()
    assert panel.table_widget.table.rowCount() == 0
    panel.open_document(path)
    assert _cells(panel.table_widget.table) == golden["view"]["rows"]
    assert panel.source_label.text() == str(path)
    assert panel.error_label.text() == ""


@pytest.mark.parametrize("text", ["not json", "[1, 2]", json.dumps({"schema": "nope"})])
def test_bad_file_reports_an_error_and_keeps_the_table(
    app, golden, tmp_path, text
) -> None:
    from src.tools.anatomical_event_table.gui import AnatomicalEventTablePanel

    good = tmp_path / "good.json"
    good.write_text(json.dumps(golden["document"]), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(text, encoding="utf-8")
    panel = AnatomicalEventTablePanel()
    panel.open_document(good)
    panel.open_document(bad)
    assert _cells(panel.table_widget.table) == golden["view"]["rows"]
    assert panel.source_label.text() == str(good)
    assert panel.error_label.text() != ""


def test_parity_entry_is_hosted_by_the_tile_on_both_surfaces() -> None:
    registry = json.loads(
        (ROOT / "src/config/feature_parity.json").read_text(encoding="utf-8")
    )
    entry = registry["features"]["biomechanics.anatomical_event_table"]
    assert entry["status"] == "parity"
    assert entry["tiles"] == [TILE]
    tiles = LauncherManifest.load(
        ROOT / "src/config/launcher_manifest.json", include_provider_tiles=False
    ).tiles
    tile = next(t for t in tiles if t.id == TILE)
    assert tile.web.route == "/tools/analysis"
    entry_points = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    embeddable = entry_points["project"]["entry-points"][
        "upstream_drift.embeddable_tools"
    ]
    assert embeddable[TILE] == _ADAPTER_MODULE


def test_panel_passes_the_gui_thread_blocking_ratchet() -> None:
    from scripts.ci.check_gui_thread_blocking_ratchet import find_unmigrated_files

    gui = ROOT / "src/tools/anatomical_event_table/gui.py"
    assert gui not in find_unmigrated_files()
