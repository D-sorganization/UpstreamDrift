"""Variant overlay selector for the player (#9797).

A row of checkable variants; the player asks :meth:`tracks_for` which
:class:`Track` objects to draw on the current view. Tracks are projected
once per (view, variants) and cached, so scrubbing stays cheap.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from src.motion_capture.reconstruct.overlay3d import Track
from src.shared.python.core.process_safety import narrow_catch

from .overlay_render import OverlaySpec
from .session import SessionMedia


class VariantOverlayBox(QWidget):
    """Checkboxes, one per variant that has something to draw."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(QLabel("Model overlay"))
        self._checks: dict[str, QCheckBox] = {}
        self._session: Path | None = None
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[Track, ...]] = {}
        self.error: str | None = None

    def load(self, media: SessionMedia | None) -> None:
        for check in self._checks.values():
            self._layout.removeWidget(check)
            check.deleteLater()
        self._checks = {}
        self._cache = {}
        self._session = media.root if media else None
        if media is None:
            return
        for variant in media.variants:
            if not (variant.has_reconstruction or variant.has_model_fit):
                continue
            check = QCheckBox(variant.label)
            check.toggled.connect(lambda _checked: self.changed.emit())
            self._layout.addWidget(check)
            self._checks[variant.name] = check
        self._layout.addStretch(1)

    def selected(self) -> tuple[str, ...]:
        return tuple(n for n, c in self._checks.items() if c.isChecked())

    def select(self, names: tuple[str, ...]) -> None:
        for name, check in self._checks.items():
            check.setChecked(name in names)

    def tracks_for(self, view: str, names: tuple[str, ...] = ()) -> tuple[Track, ...]:
        """Projected tracks on ``view`` (cached), for ``names`` or the ticks.

        ``names`` lets a layout tile name its own variants (#9814); empty
        falls back to whatever the operator has checked here, which is what
        the single-view player has always drawn.
        """
        names = names or self.selected()
        if self._session is None or not names:
            return ()
        key = (view, names)
        if key not in self._cache:
            self.error = None
            self._cache[key] = ()
            with narrow_catch(ValueError, OSError, KeyError, log_message="overlay"):
                self._cache[key] = OverlaySpec.build(self._session, view, names).tracks
            if not self._cache[key]:
                self.error = (
                    f"nothing to draw for {', '.join(n or 'default' for n in names)}"
                )
        return self._cache[key]
