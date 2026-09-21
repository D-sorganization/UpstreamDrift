"""Unit tests for embedded-host workspace layout and dock geometry persistence (#8899).

Tests state_snapshot and restore_state round-trips (tab order, active tab,
dock areas, and dock geometry), schema extension in LayoutManager, and
launcher persistence wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from src.launchers.embedded_host import EmbeddedHostWidget
from src.launchers.launcher_constants import ViewMode
from src.launchers.launcher_layout_manager import LayoutManager
from src.launchers.launcher_layout_persistence import (
    load_layout_state,
    save_layout_state,
)
from src.shared.python.config.model_registry import ModelConfig as ModelSpec
from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    register_embeddable_tool,
    unregister_embeddable_tool,
)

pytestmark = [pytest.mark.unit]


class _StubTool:
    """Minimal EmbeddableTool for persistence testing."""

    def __init__(
        self,
        tool_id: str,
        *,
        supports_embedded: bool = True,
        prefers_dock: bool = False,
    ) -> None:
        self.tool_id = tool_id
        self._caps = EmbedCapabilities(
            supports_embedded=supports_embedded,
            prefers_dock=prefers_dock,
        )

    def embed_capabilities(self) -> EmbedCapabilities:
        return self._caps

    def create_main_widget(self, parent: Any) -> QWidget:
        label = QLabel(self.tool_id, parent)
        label.setObjectName(f"stub::{self.tool_id}")
        return label

    def cleanup(self) -> None:
        pass

    def is_dirty(self) -> bool:
        return False


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def host(qapp: QApplication) -> Any:
    widget = EmbeddedHostWidget()
    yield widget
    widget.close()
    widget.deleteLater()


@pytest.fixture
def clean_registry() -> Any:
    registered: list[str] = []
    yield registered
    for tool_id in registered:
        unregister_embeddable_tool(tool_id)


def _reg(tool: _StubTool, registered: list[str]) -> None:
    register_embeddable_tool(tool)
    registered.append(tool.tool_id)


def test_embedded_host_state_snapshot_structure(
    host: EmbeddedHostWidget, clean_registry: list[str]
) -> None:
    """state_snapshot should include tabs, docks, active_tab, and dock_geometry."""
    t1 = _StubTool("tool_a")
    t2 = _StubTool("tool_b", prefers_dock=True)
    _reg(t1, clean_registry)
    _reg(t2, clean_registry)

    host.open_tab("tool_a")
    host.open_dock("tool_b", area=Qt.DockWidgetArea.LeftDockWidgetArea)

    snapshot = host.state_snapshot()
    assert snapshot["tabs"] == ["tool_a"]
    assert "tool_b" in snapshot["docks"]
    assert snapshot["docks"]["tool_b"] == int(
        Qt.DockWidgetArea.LeftDockWidgetArea.value
    )
    assert snapshot["active_tab"] == 0
    assert "dock_geometry" in snapshot
    assert isinstance(snapshot["dock_geometry"], str)


def test_embedded_host_restore_state_preserves_tab_order_and_active_tab(
    host: EmbeddedHostWidget, clean_registry: list[str]
) -> None:
    """restore_state should preserve exact tab ordering and active tab index."""
    tools = [_StubTool(f"tool_{i}") for i in range(4)]
    for t in tools:
        _reg(t, clean_registry)

    for t in tools:
        host.open_tab(t.tool_id)

    host.focus_tab("tool_2")
    assert host.open_tool_ids() == ["tool_0", "tool_1", "tool_2", "tool_3"]
    assert host.active_tool_id() == "tool_2"

    snapshot = host.state_snapshot()
    assert snapshot["tabs"] == ["tool_0", "tool_1", "tool_2", "tool_3"]
    assert snapshot["active_tab"] == 2

    for t in tools:
        host.close_tab(t.tool_id)
    assert host.active_tool_ids() == set()

    host.restore_state(snapshot)

    assert host.open_tool_ids() == ["tool_0", "tool_1", "tool_2", "tool_3"]
    assert host.active_tool_id() == "tool_2"


def test_embedded_host_restore_state_preserves_dock_areas(
    host: EmbeddedHostWidget, clean_registry: list[str]
) -> None:
    """restore_state should reopen docks in their saved dock areas."""
    d1 = _StubTool("dock_left", prefers_dock=True)
    d2 = _StubTool("dock_right", prefers_dock=True)
    _reg(d1, clean_registry)
    _reg(d2, clean_registry)

    host.open_dock("dock_left", area=Qt.DockWidgetArea.LeftDockWidgetArea)
    host.open_dock("dock_right", area=Qt.DockWidgetArea.RightDockWidgetArea)

    snapshot = host.state_snapshot()
    assert snapshot["docks"]["dock_left"] == int(
        Qt.DockWidgetArea.LeftDockWidgetArea.value
    )
    assert snapshot["docks"]["dock_right"] == int(
        Qt.DockWidgetArea.RightDockWidgetArea.value
    )

    host.close_dock("dock_left")
    host.close_dock("dock_right")
    assert host.active_tool_ids() == set()

    host.restore_state(snapshot)
    assert "dock_left" in host.active_tool_ids()
    assert "dock_right" in host.active_tool_ids()


def test_embedded_host_save_and_restore_state_qt_api(host: EmbeddedHostWidget) -> None:
    """EmbeddedHostWidget forwards saveState and restoreState to its internal QMainWindow."""
    state = host.saveState()
    assert isinstance(state, QByteArray)
    assert not state.isEmpty()

    result = host.restoreState(state)
    assert isinstance(result, bool)


def test_restore_state_skips_unregistered_tools(
    host: EmbeddedHostWidget,
) -> None:
    """restore_state logs and skips missing tools without raising."""
    host.restore_state(
        {
            "tabs": ["non_existent_tab_tool"],
            "docks": {
                "non_existent_dock_tool": int(
                    Qt.DockWidgetArea.RightDockWidgetArea.value
                )
            },
            "active_tab": 0,
            "dock_geometry": "deadbeef",
        }
    )
    assert host.active_tool_ids() == set()


def test_layout_manager_save_and_load_preserves_workspace_and_dock_state(
    tmp_path: Path,
) -> None:
    """LayoutManager persists workspace and dock_state in JSON layout configuration."""
    cfg = tmp_path / "layout_test.json"
    available_models = {
        "m1": ModelSpec(
            id="m1", name="Model 1", description="", type="managed", path="p1"
        ),
    }
    lm = LayoutManager(
        config_file=cfg,
        available_models=available_models,
        get_model_func=lambda mid: available_models.get(mid),
        create_card_func=lambda *a, **k: MagicMock(),
        create_header_func=lambda *a: MagicMock(),
    )
    lm.model_order = ["m1"]

    workspace_data = {
        "tabs": ["tool_a", "tool_b"],
        "docks": {"tool_c": 2},
        "active_tab": 1,
        "dock_geometry": "1234abcd",
    }
    window_state = {
        "selected_model": "m1",
        "geometry": {"x": 10, "y": 20, "width": 800, "height": 600},
        "options": {},
        "workspace": workspace_data,
        "dock_state": "aabbccdd",
    }

    lm.save_layout(window_state)

    raw_json = json.loads(cfg.read_text(encoding="utf-8"))
    assert raw_json["workspace"] == workspace_data
    assert raw_json["dock_state"] == "aabbccdd"

    lm2 = LayoutManager(
        config_file=cfg,
        available_models=available_models,
        get_model_func=lambda mid: available_models.get(mid),
        create_card_func=lambda *a, **k: MagicMock(),
        create_header_func=lambda *a: MagicMock(),
    )
    loaded = lm2.load_layout()
    assert loaded is not None
    assert loaded["workspace"] == workspace_data
    assert loaded["dock_state"] == "aabbccdd"
    assert lm2.workspace == workspace_data
    assert lm2.dock_state == "aabbccdd"


def test_save_layout_state_captures_embedded_host_snapshot(
    tmp_path: Path,
) -> None:
    """save_layout_state queries embedded_host.state_snapshot and dock state."""
    cfg = tmp_path / "launcher_layout.json"
    available_models = {
        "m1": ModelSpec(
            id="m1", name="Model 1", description="", type="managed", path="p1"
        ),
    }
    lm = LayoutManager(
        config_file=cfg,
        available_models=available_models,
        get_model_func=lambda mid: available_models.get(mid),
        create_card_func=lambda *a, **k: MagicMock(),
        create_header_func=lambda *a: MagicMock(),
    )
    lm.model_order = ["m1"]

    mock_host = MagicMock()
    mock_host.state_snapshot.return_value = {
        "tabs": ["t1"],
        "docks": {},
        "active_tab": 0,
        "dock_geometry": "0011",
    }
    mock_host.host_window.saveState.return_value = QByteArray(b"qt_state")

    launcher = MagicMock()
    launcher.selected_model = "m1"
    launcher.x.return_value = 100
    launcher.y.return_value = 100
    launcher.width.return_value = 1280
    launcher.height.return_value = 800
    launcher.chk_live.isChecked.return_value = True
    launcher.chk_gpu.isChecked.return_value = False
    launcher.chk_docker.isChecked.return_value = True
    launcher.chk_wsl.isChecked.return_value = False
    launcher.layout_manager = lm
    launcher.embedded_host = mock_host

    save_layout_state(launcher)

    mock_host.state_snapshot.assert_called_once()
    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["workspace"]["tabs"] == ["t1"]
    assert "dock_state" in saved


def test_load_layout_state_restores_embedded_host_workspace(
    tmp_path: Path,
) -> None:
    """load_layout_state calls restore_state with the persisted workspace dict."""
    cfg = tmp_path / "launcher_layout.json"
    available_models = {
        "m1": ModelSpec(
            id="m1", name="Model 1", description="", type="managed", path="p1"
        ),
    }
    lm = LayoutManager(
        config_file=cfg,
        available_models=available_models,
        get_model_func=lambda mid: available_models.get(mid),
        create_card_func=lambda *a, **k: MagicMock(),
        create_header_func=lambda *a: MagicMock(),
    )
    lm.model_order = ["m1"]

    workspace_payload = {
        "tabs": ["t1", "t2"],
        "docks": {"d1": 1},
        "active_tab": 1,
    }
    payload = {
        "model_order": ["m1"],
        "workspace": workspace_payload,
        "dock_state": b"dock_bytes".hex(),
    }
    cfg.write_text(json.dumps(payload), encoding="utf-8")

    mock_host = MagicMock()
    launcher = MagicMock()
    launcher.layout_manager = lm
    launcher.embedded_host = mock_host
    launcher.model_cards = {}
    launcher._viewmode_actions = {ViewMode.LIST_LARGE: MagicMock()}

    load_layout_state(launcher)

    mock_host.restore_state.assert_called_once_with(workspace_payload)
    mock_host.host_window.restoreState.assert_called_once()


def test_persistence_handles_none_embedded_host(tmp_path: Path) -> None:
    """save_layout_state and load_layout_state handle embedded_host=None without error."""
    cfg = tmp_path / "launcher_layout.json"
    available_models = {
        "m1": ModelSpec(
            id="m1", name="Model 1", description="", type="managed", path="p1"
        ),
    }
    lm = LayoutManager(
        config_file=cfg,
        available_models=available_models,
        get_model_func=lambda mid: available_models.get(mid),
        create_card_func=lambda *a, **k: MagicMock(),
        create_header_func=lambda *a: MagicMock(),
    )
    lm.model_order = ["m1"]

    launcher = MagicMock()
    launcher.selected_model = "m1"
    launcher.x.return_value = 0
    launcher.y.return_value = 0
    launcher.width.return_value = 1000
    launcher.height.return_value = 700
    launcher.chk_live.isChecked.return_value = True
    launcher.chk_gpu.isChecked.return_value = False
    launcher.chk_docker.isChecked.return_value = False
    launcher.chk_wsl.isChecked.return_value = False
    launcher.layout_manager = lm
    launcher.embedded_host = None
    launcher.model_cards = {}
    launcher._viewmode_actions = {ViewMode.LIST_LARGE: MagicMock()}

    save_layout_state(launcher)
    load_layout_state(launcher)

    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert "workspace" not in saved
