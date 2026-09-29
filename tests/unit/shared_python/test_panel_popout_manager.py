"""Tests for the sidebar panel pop-out manager (issue #5380).

Verifies that sidebar panels can be detached into a floating QDialog,
re-docked, and that pop-out state is persisted across sessions.

Design: TDD/DbC/LOD/DRY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.gui_launcher.panel_popout_manager import (
    PanelNotRegisteredError,
    PanelPopoutManager,
    PopoutState,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakePanel:
    """Minimal stand-in for a sidebar panel widget."""

    def __init__(self, name: str = "FakePanel") -> None:
        self.name = name
        self._parent: Any = None

    def parent(self) -> Any:
        return self._parent

    def setParent(self, parent: Any) -> None:
        self._parent = parent


# ---------------------------------------------------------------------------
# PopoutState dataclass
# ---------------------------------------------------------------------------


class TestPopoutState:
    """PopoutState encapsulates serialisable panel geometry."""

    def test_to_dict_and_from_dict_round_trip(self) -> None:
        state = PopoutState(
            panel_id="tools",
            is_floating=True,
            x=100,
            y=200,
            width=400,
            height=600,
        )
        d = state.to_dict()
        restored = PopoutState.from_dict(d)

        assert restored.panel_id == "tools"
        assert restored.is_floating is True
        assert restored.x == 100
        assert restored.y == 200
        assert restored.width == 400
        assert restored.height == 600

    def test_to_dict_contains_expected_keys(self) -> None:
        state = PopoutState(panel_id="sim", is_floating=False)
        d = state.to_dict()
        assert "panel_id" in d
        assert "is_floating" in d
        assert "x" in d
        assert "y" in d
        assert "width" in d
        assert "height" in d

    def test_from_dict_with_minimal_keys_defaults_gracefully(self) -> None:
        state = PopoutState.from_dict({"panel_id": "mini", "is_floating": False})
        assert state.panel_id == "mini"
        assert state.is_floating is False
        assert state.x == 0
        assert state.y == 0

    def test_from_dict_rejects_missing_panel_id(self) -> None:
        with pytest.raises((KeyError, ValueError)):
            PopoutState.from_dict({"is_floating": True})


# ---------------------------------------------------------------------------
# Registration / DbC preconditions
# ---------------------------------------------------------------------------


class TestPanelRegistration:
    """Panels must be registered before popping out."""

    def test_register_accepts_valid_panel(self) -> None:
        mgr = PanelPopoutManager()
        panel = FakePanel()
        mgr.register_panel("tools", panel)
        assert mgr.is_registered("tools")

    def test_is_registered_returns_false_for_unknown_id(self) -> None:
        mgr = PanelPopoutManager()
        assert mgr.is_registered("nonexistent") is False

    def test_register_rejects_empty_panel_id(self) -> None:
        mgr = PanelPopoutManager()
        with pytest.raises(ValueError, match="panel_id"):
            mgr.register_panel("", FakePanel())

    def test_register_rejects_none_panel(self) -> None:
        mgr = PanelPopoutManager()
        with pytest.raises((TypeError, ValueError)):
            mgr.register_panel("tools", None)  # type: ignore[arg-type]

    def test_popout_unregistered_panel_raises(self) -> None:
        mgr = PanelPopoutManager()
        with pytest.raises(PanelNotRegisteredError, match="not_here"):
            mgr.popout("not_here")

    def test_redock_unregistered_panel_raises(self) -> None:
        mgr = PanelPopoutManager()
        with pytest.raises(PanelNotRegisteredError, match="not_here"):
            mgr.redock("not_here")

    def test_register_twice_replaces_entry(self) -> None:
        mgr = PanelPopoutManager()
        panel_a = FakePanel("A")
        panel_b = FakePanel("B")
        mgr.register_panel("tools", panel_a)
        mgr.register_panel("tools", panel_b)
        assert mgr.get_panel("tools") is panel_b

    def test_unregister_panel(self) -> None:
        mgr = PanelPopoutManager()
        mgr.register_panel("tools", FakePanel())
        mgr.unregister_panel("tools")
        assert mgr.is_registered("tools") is False

    def test_unregister_unknown_is_noop(self) -> None:
        mgr = PanelPopoutManager()
        # Should not raise
        mgr.unregister_panel("nonexistent")


# ---------------------------------------------------------------------------
# Pop-out / re-dock (headless, Qt-free)
# ---------------------------------------------------------------------------


class TestPopoutRedock:
    """pop-out and re-dock lifecycle (headless, mocked Qt)."""

    def test_popout_marks_panel_floating(self) -> None:
        mgr = PanelPopoutManager()
        panel = FakePanel()
        mgr.register_panel("tools", panel)

        with (
            patch(
                "src.shared.python.gui_launcher.panel_popout_manager"
                "._create_float_dialog",
                return_value=MagicMock(),
            ),
        ):
            mgr.popout("tools")

        assert mgr.is_floating("tools")

    def test_redock_clears_floating_flag(self) -> None:
        mgr = PanelPopoutManager()
        panel = FakePanel()
        mgr.register_panel("tools", panel)

        mock_dialog = MagicMock()
        with patch(
            "src.shared.python.gui_launcher.panel_popout_manager._create_float_dialog",
            return_value=mock_dialog,
        ):
            mgr.popout("tools")

        mgr.redock("tools")
        assert mgr.is_floating("tools") is False

    def test_is_floating_false_before_popout(self) -> None:
        mgr = PanelPopoutManager()
        mgr.register_panel("tools", FakePanel())
        assert mgr.is_floating("tools") is False

    def test_is_floating_raises_for_unregistered(self) -> None:
        mgr = PanelPopoutManager()
        with pytest.raises(PanelNotRegisteredError):
            mgr.is_floating("ghost")

    def test_popout_returns_dialog_object(self) -> None:
        mgr = PanelPopoutManager()
        mgr.register_panel("tools", FakePanel())
        mock_dialog = MagicMock()
        with patch(
            "src.shared.python.gui_launcher.panel_popout_manager._create_float_dialog",
            return_value=mock_dialog,
        ):
            result = mgr.popout("tools")
        assert result is mock_dialog

    def test_popout_idempotent_returns_same_dialog(self) -> None:
        """Calling popout twice on an already-floating panel is a no-op."""
        mgr = PanelPopoutManager()
        mgr.register_panel("tools", FakePanel())
        mock_dialog = MagicMock()
        with patch(
            "src.shared.python.gui_launcher.panel_popout_manager._create_float_dialog",
            return_value=mock_dialog,
        ):
            first = mgr.popout("tools")
            second = mgr.popout("tools")
        assert first is second

    def test_redock_is_noop_when_not_floating(self) -> None:
        """redock on a docked panel must not raise."""
        mgr = PanelPopoutManager()
        mgr.register_panel("tools", FakePanel())
        # Should not raise
        mgr.redock("tools")


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """Pop-out state is saved to / loaded from a JSON file."""

    def test_save_state_creates_file(self, tmp_path: Path) -> None:
        mgr = PanelPopoutManager(state_file=tmp_path / "popout_state.json")
        mgr.register_panel("analysis", FakePanel())

        mock_dialog = MagicMock()
        mock_dialog.x.return_value = 50
        mock_dialog.y.return_value = 60
        mock_dialog.width.return_value = 300
        mock_dialog.height.return_value = 400
        with patch(
            "src.shared.python.gui_launcher.panel_popout_manager._create_float_dialog",
            return_value=mock_dialog,
        ):
            mgr.popout("analysis")

        mgr.save_state()
        state_file = tmp_path / "popout_state.json"
        assert state_file.exists()

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        state_file = tmp_path / "popout_state.json"
        mgr = PanelPopoutManager(state_file=state_file)
        mgr.register_panel("analysis", FakePanel())

        mock_dialog = MagicMock()
        mock_dialog.x.return_value = 10
        mock_dialog.y.return_value = 20
        mock_dialog.width.return_value = 320
        mock_dialog.height.return_value = 480
        with patch(
            "src.shared.python.gui_launcher.panel_popout_manager._create_float_dialog",
            return_value=mock_dialog,
        ):
            mgr.popout("analysis")

        mgr.save_state()

        # Restore in a new manager with same state file
        mgr2 = PanelPopoutManager(state_file=state_file)
        mgr2.register_panel("analysis", FakePanel())
        restored = mgr2.load_state()

        assert "analysis" in restored
        assert restored["analysis"].is_floating is True

    def test_load_state_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        mgr = PanelPopoutManager(state_file=tmp_path / "nonexistent.json")
        result = mgr.load_state()
        assert result == {}

    def test_load_state_returns_empty_on_corrupt_json(self, tmp_path: Path) -> None:
        state_file = tmp_path / "corrupt.json"
        state_file.write_text("NOT JSON", encoding="utf-8")
        mgr = PanelPopoutManager(state_file=state_file)
        result = mgr.load_state()
        assert result == {}

    def test_save_state_with_no_state_file_is_noop(self) -> None:
        """If no state_file configured, save_state silently no-ops."""
        mgr = PanelPopoutManager()
        mgr.register_panel("tools", FakePanel())
        # Should not raise
        mgr.save_state()

    def test_apply_state_repops_panels_that_were_floating(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        # Write a state that marks "sim" as floating
        state_data = {
            "sim": {
                "panel_id": "sim",
                "is_floating": True,
                "x": 100,
                "y": 150,
                "width": 250,
                "height": 300,
            }
        }
        state_file.write_text(json.dumps(state_data), encoding="utf-8")

        mgr = PanelPopoutManager(state_file=state_file)
        mgr.register_panel("sim", FakePanel())

        mock_dialog = MagicMock()
        with patch(
            "src.shared.python.gui_launcher.panel_popout_manager._create_float_dialog",
            return_value=mock_dialog,
        ):
            mgr.apply_saved_state()

        assert mgr.is_floating("sim")
