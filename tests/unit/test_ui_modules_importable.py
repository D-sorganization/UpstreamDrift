"""Importability tests for ui widget and overlay modules (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.ui.loading_button import LoadingButton, LoadingSpinner
from src.shared.python.ui.overlay import OverlayWidget
from src.shared.python.ui.recent_models import RecentModelItem, RecentModelsPanel
from src.shared.python.ui.shortcuts_overlay import DEFAULT_SHORTCUTS
from src.shared.python.ui.toast import Toast, ToastManager, ToastType


class TestLoadingButtonModuleImportable:
    def test_loading_button_importable(self) -> None:
        assert LoadingButton is not None

    def test_loading_spinner_importable(self) -> None:
        assert LoadingSpinner is not None


class TestToastModuleImportable:
    def test_toast_importable(self) -> None:
        assert Toast is not None

    def test_toast_manager_importable(self) -> None:
        assert ToastManager is not None

    def test_toast_type_importable(self) -> None:
        assert ToastType is not None

    def test_toast_type_has_members(self) -> None:
        assert len(list(ToastType)) > 0


class TestOverlayModuleImportable:
    def test_overlay_widget_importable(self) -> None:
        assert OverlayWidget is not None


class TestRecentModelsImportable:
    def test_recent_model_item_importable(self) -> None:
        assert RecentModelItem is not None

    def test_recent_models_panel_importable(self) -> None:
        assert RecentModelsPanel is not None


class TestShortcutsOverlayImportable:
    def test_default_shortcuts_importable(self) -> None:
        assert DEFAULT_SHORTCUTS is not None

    def test_default_shortcuts_nonempty(self) -> None:
        assert len(DEFAULT_SHORTCUTS) > 0
