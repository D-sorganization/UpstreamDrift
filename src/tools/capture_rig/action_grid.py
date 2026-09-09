"""Action buttons grouped by workflow step, with section labels (#9816).

:func:`group_actions` is the pure grouping: each action sits under the
*last* step of :data:`workflow.STEPS` that lists it (Record belongs to the
take, not to the calibration that also uses it); actions no step lists
(preview, load, stop) form a trailing *Session* group. :class:`ActionGrid`
lays the groups out in two columns, filled top to bottom, so the grid reads
in step order and stays compact. The buttons are created and owned by the
tile; the grid only places them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QPushButton, QWidget

from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics
from src.shared.python.theme.typography import Sizes, Weights, get_qfont

from . import styling, workflow

SESSION_GROUP = "Session"
COLUMNS = 2

Groups = tuple[tuple[str, tuple[str, ...]], ...]


def group_actions(
    steps: Sequence[workflow.Step],
    actions: Sequence[str],
    *,
    trailing_title: str = SESSION_GROUP,
) -> Groups:
    """``((title, actions), ...)`` in step order, then ``trailing_title``.

    Precondition: ``actions`` has no duplicates. Postcondition: every action
    appears exactly once, at the last step listing it; empty groups are
    dropped and the order of actions within a group follows ``actions``.
    """
    require(len(set(actions)) == len(actions), "actions must be unique", actions)
    home: dict[str, str] = {}
    for step in steps:
        for action in step.actions:
            if action in actions:
                home[action] = step.title
    titles = [s.title for s in steps] + [trailing_title]
    groups = {title: [] for title in titles}  # type: dict[str, list[str]]
    for action in actions:
        groups[home.get(action, trailing_title)].append(action)
    return tuple((t, tuple(groups[t])) for t in titles if groups[t])


class ActionGrid(QWidget):
    """Section label + that step's buttons per row, two step-columns wide."""

    def __init__(
        self,
        buttons: Mapping[str, QPushButton],
        steps: Sequence[workflow.Step] = workflow.STEPS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        require(bool(buttons), "at least one button is required")
        self.groups: Groups = group_actions(steps, tuple(buttons))
        self.section_labels: dict[str, QLabel] = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(LayoutMetrics.SPACING_SM)
        grid.setVerticalSpacing(LayoutMetrics.SPACING_SM)
        widest = max(len(actions) for _, actions in self.groups)
        rows = math.ceil(len(self.groups) / COLUMNS)
        for index, (title, actions) in enumerate(self.groups):
            row, block = index % rows, (index // rows) * (widest + 1)
            label = QLabel(title)
            label.setFont(get_qfont(Sizes.SM, Weights.SEMIBOLD))
            label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.section_labels[title] = label
            grid.addWidget(label, row, block)
            for offset, action in enumerate(actions, start=1):
                grid.addWidget(buttons[action], row, block + offset)
        for column in range(COLUMNS):
            grid.setColumnStretch(column * (widest + 1), 0)
            for offset in range(1, widest + 1):
                grid.setColumnStretch(column * (widest + 1) + offset, 1)
        self.restyle()

    @staticmethod
    def section_style() -> str:
        return styling.section_label_style()

    def restyle(self) -> None:
        style = self.section_style()
        for label in self.section_labels.values():
            label.setStyleSheet(style)
