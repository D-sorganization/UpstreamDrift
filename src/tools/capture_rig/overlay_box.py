"""Variant overlay selector for the player (#9797).

A row of checkable variants; the player asks :meth:`tracks_for` which
:class:`Track` objects to draw on the current view. Tracks are projected
once per (view, variants) and cached, so scrubbing stays cheap.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSlider, QWidget

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

        self.joint_torques = QCheckBox("Joint torques")
        self.forces = QCheckBox("Forces")
        self.torques = QCheckBox("Torques")
        self.legend = QCheckBox("Legend")
        self.forces.setChecked(True)
        self.torques.setChecked(True)
        self.legend.setChecked(False)

        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(1, 500)
        self.scale_slider.setValue(100)
        self.scale_slider.setToolTip("Scale")
        self.scale = self.scale_slider

        for widget in (
            self.joint_torques,
            self.forces,
            self.torques,
            self.legend,
        ):
            widget.toggled.connect(lambda _checked: self.changed.emit())
        self.scale_slider.valueChanged.connect(lambda _val: self.changed.emit())

        self._layout.addWidget(self.joint_torques)
        self._layout.addWidget(self.forces)
        self._layout.addWidget(self.torques)
        self._layout.addWidget(self.legend)
        self._layout.addWidget(QLabel("Scale"))
        self._layout.addWidget(self.scale_slider)
        self._layout.addStretch(1)

    @property
    def force_scale(self) -> float:
        return float(self.scale_slider.value()) / 100.0

    @property
    def joint_torques_enabled(self) -> bool:
        return self.joint_torques.isChecked()

    @property
    def forces_enabled(self) -> bool:
        return self.forces.isChecked()

    @property
    def torques_enabled(self) -> bool:
        return self.torques.isChecked()

    @property
    def legend_enabled(self) -> bool:
        return self.legend.isChecked()

    def load(self, media: SessionMedia | None) -> None:
        for check in self._checks.values():
            self._layout.removeWidget(check)
            check.deleteLater()
        self._checks = {}
        self._cache = {}
        self._session = media.root if media else None
        if media is None:
            return
        insert_idx = 1
        for variant in media.variants:
            if not (variant.has_reconstruction or variant.has_model_fit):
                continue
            check = QCheckBox(variant.label)
            check.toggled.connect(lambda _checked: self.changed.emit())
            self._layout.insertWidget(insert_idx, check)
            self._checks[variant.name] = check
            insert_idx += 1

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
