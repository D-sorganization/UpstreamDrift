"""Action buttons grouped by workflow step, with section labels (#9816).

:func:`group_actions` is the pure grouping: each action sits under the
*last* step of :data:`workflow.STEPS` that lists it (Record belongs to the
take, not to the calibration that also uses it); actions no step lists
(preview, load, stop) form a trailing *Session* group.

:class:`ActionGrid` places those groups in a :class:`~.flow_layout.FlowLayout`
so they wrap onto as many rows as the pane is wide enough for (#9844). Each
section label opens a fresh row, so it always sits immediately above or
beside the buttons it names, and the grid's minimum width is one button
rather than the sum of every column. The buttons are created and owned by
the tile; the grid only places them, tooltips and all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QSizePolicy, QWidget

from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics
from src.shared.python.theme.typography import Sizes, Weights, get_qfont

from . import styling, workflow
from .flow_layout import FlowLayout

SESSION_GROUP = "Session"
NARROW_WIDTH = 320  # the width the grid must still fit inside (#9844)

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
    """A wrapping row of step sections: a label, then that step's buttons."""

    def __init__(
        self,
        buttons: Mapping[str, QPushButton],
        steps: Sequence[workflow.Step] = workflow.STEPS,
        parent: QWidget | None = None,
    ) -> None:
        """Precondition: ``buttons`` is not empty.

        Postcondition: every button in ``buttons`` is placed exactly once,
        each group's label opens the row its first button starts on, and
        ``minimumSizeHint().width()`` is at most :data:`NARROW_WIDTH`.
        """
        super().__init__(parent)
        require(bool(buttons), "at least one button is required")
        self.groups: Groups = group_actions(steps, tuple(buttons))
        self.section_labels: dict[str, QLabel] = {}
        flow = FlowLayout(self, spacing=LayoutMetrics.SPACING_SM)
        row_height = max(b.sizeHint().height() for b in buttons.values())
        for title, actions in self.groups:
            flow.add_widget(self._label(title, row_height), starts_line=True)
            for action in actions:
                flow.add_widget(buttons[action])
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        self.setSizePolicy(policy)
        self.restyle()

    def _label(self, title: str, row_height: int) -> QLabel:
        """The section label for ``title``, wrapping rather than widening."""
        label = QLabel(title)
        label.setFont(get_qfont(Sizes.SM, Weights.SEMIBOLD))
        label.setWordWrap(True)
        label.setMinimumHeight(row_height)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.section_labels[title] = label
        return label

    @staticmethod
    def section_style() -> str:
        return styling.section_label_style()

    def restyle(self) -> None:
        style = self.section_style()
        for label in self.section_labels.values():
            label.setStyleSheet(style)
