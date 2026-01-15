"""User preferences dialog for Golf Modeling Suite.

Provides a centralized settings interface for user preferences including:
- Appearance (theme, font size)
- Behavior (startup options, notifications)
- Performance (GPU acceleration, cache settings)

Usage:
    from shared.python.ui.preferences_dialog import PreferencesDialog

    dialog = PreferencesDialog(parent_window)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        # Preferences were saved
        pass
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow

# Import theme if available
try:
    from shared.python.theme import Colors, Sizes, Weights, get_qfont  # noqa: F401

    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default preferences file location
PREFS_DIR = Path.home() / ".golf_modeling_suite"
PREFS_FILE = PREFS_DIR / "preferences.json"


@dataclass
class UserPreferences:
    """User preferences data structure."""

    # Appearance
    theme: str = "dark"  # "dark", "light", "system"
    font_size: int = 10
    show_tooltips: bool = True
    compact_mode: bool = False

    # Startup
    show_splash_screen: bool = True
    restore_last_session: bool = False
    check_updates_on_startup: bool = True
    auto_detect_engines: bool = True

    # Notifications
    show_notifications: bool = True
    notification_duration: int = 4  # seconds
    play_sounds: bool = False

    # Performance
    enable_gpu_acceleration: bool = True
    max_recent_models: int = 10
    cache_model_previews: bool = True
    preload_engines: bool = False

    # Paths
    default_export_path: str = ""
    recent_models: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "UserPreferences":
        """Load preferences from file."""
        if PREFS_FILE.exists():
            try:
                data = json.loads(PREFS_FILE.read_text())
                return cls(
                    **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                )
            except Exception as e:
                logger.warning(f"Failed to load preferences: {e}")
        return cls()

    def save(self) -> None:
        """Save preferences to file."""
        try:
            PREFS_DIR.mkdir(parents=True, exist_ok=True)
            PREFS_FILE.write_text(json.dumps(asdict(self), indent=2))
            logger.info("Preferences saved")
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")


class PreferencesDialog(QDialog):
    """User preferences dialog with tabbed sections."""

    def __init__(self, parent: "QMainWindow | None" = None) -> None:
        """Create the preferences dialog.

        Args:
            parent: Parent window
        """
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(550, 450)
        self.resize(600, 500)

        # Load current preferences
        self.prefs = UserPreferences.load()
        self._original_prefs = UserPreferences.load()

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_appearance_tab(), "Appearance")
        tabs.addTab(self._create_startup_tab(), "Startup")
        tabs.addTab(self._create_notifications_tab(), "Notifications")
        tabs.addTab(self._create_performance_tab(), "Performance")
        layout.addWidget(tabs)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        if apply_btn := buttons.button(QDialogButtonBox.StandardButton.Apply):
            apply_btn.clicked.connect(self._on_apply)
        if restore_btn := buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ):
            restore_btn.clicked.connect(self._on_restore_defaults)

        layout.addWidget(buttons)

    def _create_appearance_tab(self) -> QWidget:
        """Create the appearance settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Theme group
        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout(theme_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        self.theme_combo.setCurrentText(self.prefs.theme.capitalize())
        theme_layout.addRow("Color theme:", self.theme_combo)

        layout.addWidget(theme_group)

        # Font group
        font_group = QGroupBox("Font")
        font_layout = QFormLayout(font_group)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        self.font_size_spin.setValue(self.prefs.font_size)
        self.font_size_spin.setSuffix(" pt")
        font_layout.addRow("Base font size:", self.font_size_spin)

        layout.addWidget(font_group)

        # UI Options group
        ui_group = QGroupBox("User Interface")
        ui_layout = QVBoxLayout(ui_group)

        self.tooltips_check = QCheckBox("Show tooltips")
        self.tooltips_check.setChecked(self.prefs.show_tooltips)
        ui_layout.addWidget(self.tooltips_check)

        self.compact_check = QCheckBox("Compact mode (smaller spacing)")
        self.compact_check.setChecked(self.prefs.compact_mode)
        ui_layout.addWidget(self.compact_check)

        layout.addWidget(ui_group)

        layout.addStretch()
        return tab

    def _create_startup_tab(self) -> QWidget:
        """Create the startup settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Startup behavior group
        startup_group = QGroupBox("Startup Behavior")
        startup_layout = QVBoxLayout(startup_group)

        self.splash_check = QCheckBox("Show splash screen on startup")
        self.splash_check.setChecked(self.prefs.show_splash_screen)
        startup_layout.addWidget(self.splash_check)

        self.restore_session_check = QCheckBox("Restore last session on startup")
        self.restore_session_check.setChecked(self.prefs.restore_last_session)
        startup_layout.addWidget(self.restore_session_check)

        self.updates_check = QCheckBox("Check for updates on startup")
        self.updates_check.setChecked(self.prefs.check_updates_on_startup)
        startup_layout.addWidget(self.updates_check)

        self.auto_detect_check = QCheckBox("Auto-detect physics engines")
        self.auto_detect_check.setChecked(self.prefs.auto_detect_engines)
        startup_layout.addWidget(self.auto_detect_check)

        layout.addWidget(startup_group)

        layout.addStretch()
        return tab

    def _create_notifications_tab(self) -> QWidget:
        """Create the notifications settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Notification settings group
        notif_group = QGroupBox("Toast Notifications")
        notif_layout = QVBoxLayout(notif_group)

        self.notif_check = QCheckBox("Show notifications")
        self.notif_check.setChecked(self.prefs.show_notifications)
        notif_layout.addWidget(self.notif_check)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Display duration:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 10)
        self.duration_spin.setValue(self.prefs.notification_duration)
        self.duration_spin.setSuffix(" seconds")
        duration_layout.addWidget(self.duration_spin)
        duration_layout.addStretch()
        notif_layout.addLayout(duration_layout)

        self.sounds_check = QCheckBox("Play notification sounds")
        self.sounds_check.setChecked(self.prefs.play_sounds)
        notif_layout.addWidget(self.sounds_check)

        layout.addWidget(notif_group)

        layout.addStretch()
        return tab

    def _create_performance_tab(self) -> QWidget:
        """Create the performance settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Hardware group
        hw_group = QGroupBox("Hardware Acceleration")
        hw_layout = QVBoxLayout(hw_group)

        self.gpu_check = QCheckBox("Enable GPU acceleration (requires restart)")
        self.gpu_check.setChecked(self.prefs.enable_gpu_acceleration)
        hw_layout.addWidget(self.gpu_check)

        self.preload_check = QCheckBox("Preload physics engines at startup")
        self.preload_check.setChecked(self.prefs.preload_engines)
        hw_layout.addWidget(self.preload_check)

        layout.addWidget(hw_group)

        # Cache group
        cache_group = QGroupBox("Cache Settings")
        cache_layout = QVBoxLayout(cache_group)

        recent_layout = QHBoxLayout()
        recent_layout.addWidget(QLabel("Maximum recent models:"))
        self.recent_spin = QSpinBox()
        self.recent_spin.setRange(5, 50)
        self.recent_spin.setValue(self.prefs.max_recent_models)
        recent_layout.addWidget(self.recent_spin)
        recent_layout.addStretch()
        cache_layout.addLayout(recent_layout)

        self.cache_previews_check = QCheckBox("Cache model preview images")
        self.cache_previews_check.setChecked(self.prefs.cache_model_previews)
        cache_layout.addWidget(self.cache_previews_check)

        layout.addWidget(cache_group)

        layout.addStretch()
        return tab

    def _collect_preferences(self) -> UserPreferences:
        """Collect current UI values into preferences object."""
        return UserPreferences(
            theme=self.theme_combo.currentText().lower(),
            font_size=self.font_size_spin.value(),
            show_tooltips=self.tooltips_check.isChecked(),
            compact_mode=self.compact_check.isChecked(),
            show_splash_screen=self.splash_check.isChecked(),
            restore_last_session=self.restore_session_check.isChecked(),
            check_updates_on_startup=self.updates_check.isChecked(),
            auto_detect_engines=self.auto_detect_check.isChecked(),
            show_notifications=self.notif_check.isChecked(),
            notification_duration=self.duration_spin.value(),
            play_sounds=self.sounds_check.isChecked(),
            enable_gpu_acceleration=self.gpu_check.isChecked(),
            max_recent_models=self.recent_spin.value(),
            cache_model_previews=self.cache_previews_check.isChecked(),
            preload_engines=self.preload_check.isChecked(),
            default_export_path=self.prefs.default_export_path,
            recent_models=self.prefs.recent_models,
        )

    def _on_apply(self) -> None:
        """Apply current settings without closing."""
        self.prefs = self._collect_preferences()
        self.prefs.save()

    def _on_accept(self) -> None:
        """Accept and save settings."""
        self._on_apply()
        self.accept()

    def _on_restore_defaults(self) -> None:
        """Restore default settings."""
        defaults = UserPreferences()

        # Update UI
        self.theme_combo.setCurrentText(defaults.theme.capitalize())
        self.font_size_spin.setValue(defaults.font_size)
        self.tooltips_check.setChecked(defaults.show_tooltips)
        self.compact_check.setChecked(defaults.compact_mode)
        self.splash_check.setChecked(defaults.show_splash_screen)
        self.restore_session_check.setChecked(defaults.restore_last_session)
        self.updates_check.setChecked(defaults.check_updates_on_startup)
        self.auto_detect_check.setChecked(defaults.auto_detect_engines)
        self.notif_check.setChecked(defaults.show_notifications)
        self.duration_spin.setValue(defaults.notification_duration)
        self.sounds_check.setChecked(defaults.play_sounds)
        self.gpu_check.setChecked(defaults.enable_gpu_acceleration)
        self.recent_spin.setValue(defaults.max_recent_models)
        self.cache_previews_check.setChecked(defaults.cache_model_previews)
        self.preload_check.setChecked(defaults.preload_engines)

    def get_preferences(self) -> UserPreferences:
        """Get the current preferences."""
        return self.prefs


__all__ = ["PreferencesDialog", "UserPreferences", "PREFS_FILE"]
