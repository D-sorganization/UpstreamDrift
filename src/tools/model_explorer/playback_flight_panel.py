"""Shot card beside the live Results Playback: the thin Qt layer (#15150).

The rows, their numbers and the card options come from :mod:`shot_card_rows`
(Qt-free, the exporter's computation); this module only draws them, with the
display-unit selectors, and docks the card. PyQt6 is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.tools.model_explorer.shot_card_rows import (
    ShotCardSettings,
    ShotSummary,
    load_card_settings,
    shot_rows,
)
from src.tools.native_viewer_export.display_units import (
    DisplayUnits,
    load_display_units,
)


@dataclass
class ShotCard:
    """The thin Qt card: unit selectors over the rows of one :class:`ShotSummary`.

    The widget's ``shot_rows`` property holds the ``(label, value)`` pairs on
    screen, for tests and screenshots.
    """

    summary: ShotSummary
    settings: ShotCardSettings = field(default_factory=load_card_settings)
    units: DisplayUnits = field(default_factory=load_display_units)
    widget: Any = field(init=False)

    def __post_init__(self) -> None:
        from PyQt6.QtWidgets import QComboBox, QFormLayout, QVBoxLayout, QWidget

        from src.shared.python.theme.sidekick_tokens import get_current_sidekick_tokens
        from src.tools.native_viewer_export import dialog_core

        self._accent = get_current_sidekick_tokens()["sidekick.color.accent"]
        self.widget = QWidget()
        self.widget.setMinimumWidth(self.settings.min_width_px)
        layout, selectors = QVBoxLayout(self.widget), QFormLayout()
        options = (
            dialog_core.distance_unit_options(),
            dialog_core.speed_unit_options(),
        )
        current = (self.units.distance_unit, self.units.speed_unit)
        self._combos = [QComboBox(), QComboBox()]
        for label, combo, rows, unit in zip(
            ("Distance", "Speed"), self._combos, options, current, strict=True
        ):
            for value, text in rows:
                combo.addItem(text, value)
            combo.setCurrentIndex(max(0, combo.findData(unit)))
            combo.currentIndexChanged.connect(self._units_changed)
            selectors.addRow(label, combo)
        self._form = QFormLayout()
        layout.addLayout(selectors)
        layout.addLayout(self._form)
        layout.addStretch(1)
        self._render()

    def unit_choices(self) -> tuple[tuple[tuple[str, str], ...], ...]:
        """``(value, label)`` rows of the distance and speed selectors."""
        return tuple(
            tuple((c.itemData(i), c.itemText(i)) for i in range(c.count()))
            for c in self._combos
        )

    def set_units(self, distance_unit: str, speed_unit: str) -> None:
        """Select units (``ValueError`` for an unknown one) and redraw."""
        decimals = self.units.distance_decimals
        self.units = DisplayUnits(distance_unit, speed_unit, decimals)
        for combo, unit in zip(self._combos, (distance_unit, speed_unit), strict=True):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(unit))
            combo.blockSignals(False)
        self._render()

    def set_summary(self, summary: ShotSummary) -> None:
        """Show another run's shot."""
        self.summary = summary
        self._render()

    def shown_rows(self) -> tuple[tuple[str, str], ...]:
        """``(label, value)`` pairs on screen."""
        return tuple(self.widget.property("shot_rows"))

    def _units_changed(self) -> None:
        self.set_units(self._combos[0].currentData(), self._combos[1].currentData())

    def _render(self) -> None:
        from PyQt6.QtWidgets import QLabel

        while self._form.rowCount():
            self._form.removeRow(0)
        rows, section = shot_rows(self.summary, self.units, self.settings), None
        for row in rows:
            if row.section != section:
                section, head = row.section, QLabel(row.section)
                head.setStyleSheet(f"font-weight: bold; color: {self._accent};")
                self._form.addRow(head)
            value = QLabel(row.value)
            value.setWordWrap(True)
            self._form.addRow(row.label, value)
        self.widget.setProperty("shot_rows", [(r.label, r.value) for r in rows])


def attach_shot_dock(window: Any, dock: Any, summary: ShotSummary) -> Any:
    """``dock`` showing ``summary``; a new one docks right of the playback.

    It is added side by side with the right-edge docks (the playback's), not
    with ``splitDockWidget``: the playback dock is tabbed, and Qt would then
    add a tab instead of a neighbour.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDockWidget

    if dock is None:
        dock = QDockWidget(load_card_settings().title, window)
        dock.setObjectName("ShotCardDock")
        area = Qt.DockWidgetArea.RightDockWidgetArea
        window.addDockWidget(area, dock, Qt.Orientation.Horizontal)
    card = getattr(dock, "shot_card", None)
    if isinstance(card, ShotCard):
        card.set_summary(summary)
    else:
        dock.shot_card = card = ShotCard(summary)  # the dock keeps it alive
        dock.setWidget(card.widget)
    dock.show()
    return dock
