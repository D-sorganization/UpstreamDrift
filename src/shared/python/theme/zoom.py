# ruff: noqa: E501
"""Application-level zoom support for fleet PyQt6 desktop applications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from PyQt6.QtCore import QEvent, QObject, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication


class ZoomSettings(Protocol):
    """Minimal QSettings-compatible persistence protocol."""

    def value(
        self,
        key: str,
        defaultValue: object | None = None,
        **kwargs: object,
    ) -> object:
        """Read a setting value."""

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        """Persist a setting value."""


@dataclass(frozen=True)
class ZoomConfig:
    """Configuration contract for application UI zoom."""

    minimum_percent: int = 50
    maximum_percent: int = 200
    default_percent: int = 100
    step_percent: int = 10
    settings_key: str = "percent"
    settings_org: str = "D-sorganization"
    settings_app: str = "FleetShared"

    def __post_init__(self) -> None:
        """Validate zoom bounds."""
        _require_positive("minimum_percent", self.minimum_percent)
        _require_positive("maximum_percent", self.maximum_percent)
        _require_positive("step_percent", self.step_percent)
        if self.minimum_percent > self.maximum_percent:
            raise ValueError("minimum_percent must not exceed maximum_percent")
        if not self.minimum_percent <= self.default_percent <= self.maximum_percent:
            raise ValueError("default_percent must be within configured bounds")


@dataclass(frozen=True)
class ZoomTokenSet:
    """Scaled UI dimensions for QSS and responsive sizing helpers."""

    font_px: int
    label_font_px: int
    padding_px: int
    spacing_px: int
    icon_px: int
    minimum_control_px: int

    @classmethod
    def from_percent(cls, zoom_percent: int) -> ZoomTokenSet:
        """Create tokens scaled from the fleet baseline dimensions."""
        return cls(
            font_px=scale_px(12, zoom_percent),
            label_font_px=scale_px(11, zoom_percent),
            padding_px=scale_px(8, zoom_percent),
            spacing_px=scale_px(6, zoom_percent),
            icon_px=scale_px(16, zoom_percent),
            minimum_control_px=scale_px(80, zoom_percent),
        )


class ApplicationZoomController(QObject):
    """QApplication event filter that applies Chrome-style UI zoom."""

    zoomChanged = pyqtSignal(int)

    def __init__(
        self,
        app: QApplication,
        config: ZoomConfig | None = None,
        settings: ZoomSettings | None = None,
    ) -> None:
        """Create a zoom controller for ``app``."""
        super().__init__()
        self._app = app
        self._config = config or ZoomConfig()
        self._settings = settings or QSettings(
            self._config.settings_org,
            self._config.settings_app,
        )
        self._base_font = QFont(app.font())
        self._base_size = _point_size(self._base_font)
        self._zoom_percent = self._load_percent()
        self._apply_font()

    @property
    def zoom_percent(self) -> int:
        """Current application zoom percentage."""
        return self._zoom_percent

    @property
    def tokens(self) -> ZoomTokenSet:
        """Scaled design tokens for the current zoom percentage."""
        return ZoomTokenSet.from_percent(self._zoom_percent)

    @property
    def base_point_size(self) -> float:
        """Base application font point size before zoom scaling."""
        return self._base_size

    def install(self) -> None:
        """Install this controller as the QApplication event filter."""
        self._app.installEventFilter(self)

    def uninstall(self) -> None:
        """Remove this controller from QApplication event filters."""
        self._app.removeEventFilter(self)

    def set_zoom_percent(self, percent: int) -> None:
        """Set zoom, clamped to configured bounds, and persist it."""
        clamped = self._clamp(percent)
        if clamped == self._zoom_percent:
            return
        self._zoom_percent = clamped
        self._settings.setValue(self._config.settings_key, clamped)
        self._apply_font()
        self.zoomChanged.emit(clamped)

    def zoom_in(self) -> None:
        """Increase application zoom by one configured step."""
        self.set_zoom_percent(self._zoom_percent + self._config.step_percent)

    def zoom_out(self) -> None:
        """Decrease application zoom by one configured step."""
        self.set_zoom_percent(self._zoom_percent - self._config.step_percent)

    def reset_zoom(self) -> None:
        """Reset application zoom to the configured default."""
        self.set_zoom_percent(self._config.default_percent)

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:  # noqa: N802
        """Handle Ctrl+wheel and Ctrl+shortcut app zoom events."""
        if event is None:
            return False
        if event.type() == QEvent.Type.Wheel:
            return self._handle_wheel(cast(QWheelEvent, event))
        if event.type() == QEvent.Type.KeyPress:
            return self._handle_key(cast(QKeyEvent, event))
        return bool(super().eventFilter(obj, event))

    def _handle_wheel(self, event: QWheelEvent) -> bool:
        if not _has_control(event.modifiers()):
            return False
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()
        return True

    def _handle_key(self, event: QKeyEvent) -> bool:
        if not _has_control(event.modifiers()):
            return False
        if event.key() in _ZOOM_IN_KEYS:
            self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key.Key_0:
            self.reset_zoom()
        else:
            return False
        event.accept()
        return True

    def _apply_font(self) -> None:
        font = QFont(self._base_font)
        font.setPointSizeF(self._base_size * self._zoom_percent / 100.0)
        self._app.setFont(font)

    def _clamp(self, percent: int) -> int:
        return max(
            self._config.minimum_percent,
            min(self._config.maximum_percent, percent),
        )

    def _load_percent(self) -> int:
        value = self._settings.value(
            self._config.settings_key,
            self._config.default_percent,
            type=int,
        )
        return self._clamp(_coerce_percent(value, self._config.default_percent))


_ZOOM_IN_KEYS = {Qt.Key.Key_Plus, Qt.Key.Key_Equal}


def install_application_zoom(
    app: QApplication,
    config: ZoomConfig | None = None,
    settings: ZoomSettings | None = None,
) -> ApplicationZoomController:
    """Create, install, and return an application zoom controller."""
    controller = ApplicationZoomController(app, config, settings)
    controller.install()
    return controller


def scale_px(value: int, zoom_percent: int) -> int:
    """Scale a pixel value by ``zoom_percent`` with contract validation."""
    _require_positive("zoom_percent", zoom_percent)
    _require_positive("value", value)
    return max(1, round(value * zoom_percent / 100.0))


def _point_size(font: QFont) -> float:
    point_size = font.pointSizeF()
    if point_size > 0:
        return float(point_size)
    return float(font.pointSize() or 10)


def _coerce_percent(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return default


def _has_control(modifiers: Qt.KeyboardModifier) -> bool:
    return bool(modifiers & Qt.KeyboardModifier.ControlModifier)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
