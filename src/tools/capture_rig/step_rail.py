"""The step rail: where you are, and the one thing to press next (#9845).

The tile shows a step list, a guidance pane and twenty equal-looking
buttons; none of them answers "what do I press now?". :class:`StepRail`
does, in a strip narrow enough for a side dock: every step of
:data:`workflow.STEPS` as one elided row coloured by its
:class:`~.workflow.Status`, the current step bold, and under it the one
**primary** action to press with that step's remaining actions beside it.

The rail owns no state of its own and knows nothing about the tile: it is
fed :func:`workflow.evaluate` states and emits :attr:`StepRail.action_triggered`
with an action key, which the tile passes to the ``trigger()`` it already
has. Blocked and skipped rows quote :func:`workflow.action_hints`, so the
wording of "why not" has exactly one source. Colours come from
:mod:`.styling`, spacing from :class:`LayoutMetrics`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics, QResizeEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics
from src.shared.python.theme.typography import Sizes, Weights, get_qfont

from . import styling, workflow
from .workflow import Status, Step, StepState

#: The rail lives in a side dock: it must never widen the tile past this.
MAX_RAIL_WIDTH = 260

STATUS_KIND: dict[Status, str] = {
    Status.DONE: "done",
    Status.READY: "ready",  # a ready step that is not the current one
    Status.BLOCKED: "blocked",
    Status.SKIPPED: "skipped",
}
STATUS_GLYPH: dict[Status, str] = {
    Status.DONE: "✓",
    Status.READY: "▶",
    Status.BLOCKED: "○",
    Status.SKIPPED: "–",
}
CURRENT_KIND = "current"
HEADING = "What to do next"
HEADING_HINT = "Where you are in the capture workflow, and the next action."


def action_label(action: str, labels: Mapping[str, str] | None = None) -> str:
    """The button text for ``action``: the caller's, else read from the key.

    Precondition: ``action`` is not empty. Postcondition: the result is not
    empty, so no button is nameless.
    """
    require(bool(action), "an action key is required", action)
    if labels is not None and action in labels:
        return labels[action]
    return action.replace("_", " ").capitalize()


def blocked_note(step: Step, hints: Mapping[str, str]) -> str:
    """Why ``step`` cannot be worked on, in :func:`workflow.action_hints` words.

    Postcondition: the result is a substring of one of ``hints`` (never new
    wording), and is empty when nothing withholds the step.
    """
    marker = f"step '{step.title}' is"
    for action in step.actions:
        hint = hints.get(action, "")
        found = hint.find(marker)
        if found >= 0:
            return hint[found + len(marker) :].strip()
    return ""


class RailLabel(QLabel):
    """A label that elides its text rather than widening the rail.

    Invariant: :attr:`full_text` is what the label means; ``text()`` is what
    fits, and the tooltip always carries the whole of it.
    """

    def __init__(self, text: str, *, small: bool = False) -> None:
        super().__init__(text)
        self.full_text = text
        size = Sizes.SM if small else Sizes.MD
        weight = Weights.NORMAL if small else Weights.MEDIUM
        self.setFont(get_qfont(size, weight))
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        """Zero width: the rail's floor comes from its buttons, not its prose."""
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, a0: QResizeEvent | None) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(a0)
        metrics = QFontMetrics(self.font())
        fitted = metrics.elidedText(
            self.full_text, Qt.TextElideMode.ElideRight, self.width()
        )
        if fitted != self.text():
            self.setText(fitted)


class StepRail(QWidget):
    """The workflow as a narrow rail with one obvious next action.

    Invariant: after :meth:`set_states` there is a row per step, and at most
    one primary button — the current step's first action the caller allows.
    """

    action_triggered = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._labels = dict(labels) if labels is not None else None
        self._states: tuple[StepState, ...] = ()
        self._hints: dict[str, str] = {}
        self.rows: dict[str, RailLabel] = {}
        self.notes: dict[str, RailLabel] = {}
        self.kinds: dict[str, str] = {}
        self.primary: QPushButton | None = None
        self.primary_action: str | None = None
        self.secondary: tuple[QPushButton, ...] = ()
        self.secondary_actions: tuple[str, ...] = ()
        self.heading = RailLabel(HEADING)
        self.heading.setToolTip(HEADING_HINT)
        self._body = QVBoxLayout(self)
        self._body.setContentsMargins(
            LayoutMetrics.SPACING_MD,
            LayoutMetrics.SPACING_SM,
            LayoutMetrics.SPACING_MD,
            LayoutMetrics.SPACING_SM,
        )
        self._body.setSpacing(LayoutMetrics.SPACING_SM)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    # -- state ---------------------------------------------------------------
    def set_states(
        self,
        states: Sequence[StepState],
        *,
        enabled: frozenset[str] | None = None,
        hints: Mapping[str, str] | None = None,
    ) -> None:
        """Repaint the rail for ``states``.

        Precondition: ``states`` is not empty. ``enabled`` defaults to
        :func:`workflow.enabled_actions` and ``hints`` to
        :func:`workflow.action_hints`, so a caller that has already computed
        them (the tile has) passes them instead of paying twice.
        Postcondition: :attr:`rows` has one row per state and
        :attr:`primary_action` is an action of the current step, or ``None``
        when no step is current.
        """
        require(bool(states), "the workflow has no steps", states)
        self._states = tuple(states)
        allowed = (
            enabled if enabled is not None else workflow.enabled_actions(self._states)
        )
        self._hints = (
            dict(hints) if hints is not None else workflow.action_hints(self._states)
        )
        self._rebuild(allowed)

    @property
    def current_key(self) -> str | None:
        """The key of the step the operator is on, or ``None``."""
        state = workflow.current(self._states)
        return None if state is None else state.step.key

    def kind_of(self, key: str) -> str:
        """The row kind (see :data:`styling.STEP_KINDS`) drawn for step ``key``."""
        require(key in self.kinds, "unknown step", key)
        return self.kinds[key]

    def hint_of(self, action: str) -> str:
        """The tooltip text this rail shows for ``action``."""
        return self._hints.get(action, "")

    def restyle(self) -> None:
        """Re-read the palette (the theme-change hook)."""
        self.heading.setStyleSheet(styling.section_label_style())
        for key, row in self.rows.items():
            row.setStyleSheet(styling.step_row_style(self.kinds[key]))
        for note in self.notes.values():
            note.setStyleSheet(styling.section_label_style())
        if self.primary is not None:
            self.primary.setStyleSheet(styling.rail_action_style(primary=True))
        for button in self.secondary:
            button.setStyleSheet(styling.rail_action_style(primary=False))

    # -- rendering -----------------------------------------------------------
    def _rebuild(self, allowed: frozenset[str]) -> None:
        self._clear()
        self._body.addWidget(self.heading)
        current = workflow.current(self._states)
        for state in self._states:
            is_current = current is not None and state.step.key == current.step.key
            self._add_row(
                state, CURRENT_KIND if is_current else STATUS_KIND[state.status]
            )
            if is_current:
                self._add_actions(state.step, allowed)
        self._body.addStretch(1)
        self.restyle()

    def _clear(self) -> None:
        self.rows, self.notes, self.kinds = {}, {}, {}
        self.primary, self.primary_action = None, None
        self.secondary, self.secondary_actions = (), ()
        while (item := self._body.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None and widget is not self.heading:
                widget.setParent(None)
                widget.deleteLater()

    def _add_row(self, state: StepState, kind: str) -> None:
        step = state.step
        row = RailLabel(f"{STATUS_GLYPH[state.status]}  {step.title}")
        reason = f" ({state.reason})" if state.reason else ""
        row.setToolTip(f"{row.full_text}\n{step.purpose}\n{state.status.value}{reason}")
        self.rows[step.key] = row
        self.kinds[step.key] = kind
        self._body.addWidget(row)
        note = blocked_note(step, self._hints)
        if note:
            label = RailLabel(note, small=True)
            label.setToolTip(note)
            self.notes[step.key] = label
            self._body.addWidget(label)

    def _add_actions(self, step: Step, allowed: frozenset[str]) -> None:
        actions = tuple(step.actions)
        first = next((a for a in actions if a in allowed), actions[0])
        self.primary_action = first
        self.primary = self._button(first, allowed, primary=True)
        self._body.addWidget(self.primary)
        rest = tuple(a for a in actions if a != first)
        self.secondary_actions = rest
        buttons = tuple(self._button(a, allowed, primary=False) for a in rest)
        self.secondary = buttons
        if not buttons:
            return
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(LayoutMetrics.SPACING_SM)
        for button in buttons:
            row.addWidget(button)
        self._body.addLayout(row)

    def _button(
        self, action: str, allowed: frozenset[str], *, primary: bool
    ) -> QPushButton:
        button = QPushButton(action_label(action, self._labels))
        button.setEnabled(action in allowed)
        button.setToolTip(self.hint_of(action))
        button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        if primary:
            button.setMinimumHeight(LayoutMetrics.TRANSPORT_BUTTON_HEIGHT)
            button.setFont(get_qfont(Sizes.MD, Weights.SEMIBOLD))
        button.clicked.connect(lambda _=False, a=action: self.action_triggered.emit(a))
        return button
