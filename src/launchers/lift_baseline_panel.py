"""Canonical PyQt6 view of the LIFT-1 cross-engine five-lift baseline.

``LiftBaselinePanel`` renders the engine-free view built by
``src.shared.python.lifting.baseline_view.lift_view`` (LIFT-8, #11748): a lift
selector, a per-engine summary table, a pose/pair-metric parity table, a
per-engine phase table and a gaps list, plus the receipt's provenance.

This module is ENGINE-FREE and HEADLESS: it never imports MuJoCo, Drake,
OpenSim or Pinocchio, and it must never create a ``QApplication`` or show a
window itself -- it is a plain ``QWidget`` meant to be embedded (see
``src.launchers.exercise_dashboard``) and driven by tests through an offscreen
``qapp`` fixture.

"Unavailable is never zero": every scalar comes from ``baseline_view`` already
wrapped as ``{"value": float | None, "reason": str | None}``; a ``None`` value
always renders as the literal string ``"unavailable"`` with the reason as its
tooltip, never as ``"0"`` or a blank cell.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.lifting.baseline_view import (
    available_lifts,
    lift_view,
    load_baseline,
)
from src.shared.python.lifting.pack_audit.report import ENGINE_TITLES, TITLES

UNAVAILABLE = "unavailable"
PROVENANCE_NOTE = "Derived, provisional; not approved."

_ENGINE_COLUMNS: tuple[str, ...] = (
    "Engine",
    "Pack commit",
    "Bodies/nq",
    "Total mass",
    "Bar height",
    "Hand-mid height",
    "Smoke (loaded/stepped)",
    "Phases",
)
_PAIR_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("segments_max_m", "Segments"),
    ("hands_max_m", "Hands"),
    ("feet_max_m", "Feet"),
    ("bar_centre_max_m", "Bar centre"),
    ("com_max_m", "COM"),
    ("lifter_com_max_m", "Lifter COM"),
)
_PHASE_COLUMNS: tuple[str, ...] = ("Phase", "Fraction", "L hand-bar", "R hand-bar")

_STATUS_COLORS: dict[str, QColor] = {
    "pass": QColor(198, 239, 206),
    "fail": QColor(255, 199, 206),
    "unavailable": QColor(230, 230, 230),
}


# ---------------------------------------------------------------------------
# Pure formatting/row-building helpers (no Qt objects; directly testable).
# ---------------------------------------------------------------------------


def _format_scalar_field(
    field: dict[str, Any], *, unit: str = "m"
) -> tuple[str, str | None]:
    """Render a ``{"value", "reason"}`` scalar field as ``(text, tooltip)``.

    ``value is None`` always renders as :data:`UNAVAILABLE`, with the field's
    reason (if any) as the tooltip -- never a bare "0".
    """
    value = field.get("value")
    if value is None:
        return UNAVAILABLE, field.get("reason")
    if unit == "kg":
        return f"{value:.2f} kg", None
    if unit == "mm":
        return f"{value * 1000:.1f} mm", None
    if unit == "plain":
        return f"{value:.2f}", None
    return f"{value:.3f} m", None


def _short_commit(commit: str | None) -> str:
    """First 7 characters of a pack commit hash, or :data:`UNAVAILABLE`."""
    if not commit:
        return UNAVAILABLE
    return commit[:7]


def _smoke_text(value: bool | None) -> str:
    if value is None:
        return UNAVAILABLE
    return "yes" if value else "no"


def _pair_title(pair: str) -> str:
    """``"mujoco|opensim"`` -> ``"MuJoCo vs OpenSim"``."""
    return " vs ".join(ENGINE_TITLES.get(part, part) for part in pair.split("|"))


def build_engine_row(engine: dict[str, Any]) -> list[tuple[str, str | None]]:
    """``(text, tooltip)`` cells for one engine's row in the engine table."""
    pack = engine["pack"]
    structure = engine["structure"]
    smoke = engine["smoke"]
    n_bodies = structure.get("n_bodies")
    nq = structure.get("nq")
    bodies_nq = (
        f"{n_bodies if n_bodies is not None else UNAVAILABLE}/"
        f"{nq if nq is not None else UNAVAILABLE}"
    )
    mass_cell = _format_scalar_field(engine["total_mass_kg"], unit="kg")
    bar_cell = _format_scalar_field(engine["bar_above_sole_m"])
    hand_cell = _format_scalar_field(engine["hand_mid_above_sole_m"])
    smoke_cell = (
        f"{_smoke_text(smoke.get('loaded'))}/{_smoke_text(smoke.get('stepped'))}",
        None,
    )
    return [
        (ENGINE_TITLES.get(engine["engine"], engine["engine"]), None),
        (_short_commit(pack.get("commit")), pack.get("commit")),
        (bodies_nq, None),
        mass_cell,
        bar_cell,
        hand_cell,
        smoke_cell,
        (str(len(engine["phases"])), None),
    ]


def build_pair_metric_cells(
    pair_metrics: dict[str, dict[str, Any]],
) -> list[tuple[str, str | None, str]]:
    """``(text, tooltip, status)`` cells, in :data:`_PAIR_METRIC_COLUMNS` order."""
    cells: list[tuple[str, str | None, str]] = []
    for key, _title in _PAIR_METRIC_COLUMNS:
        metric = pair_metrics.get(key, {})
        value = metric.get("value")
        status = metric.get("status", "unavailable")
        tolerance_m = metric.get("tolerance_m")
        if value is None:
            text = UNAVAILABLE
            tip = metric.get("reason") or "value unavailable"
        else:
            text = f"{value * 1000:.1f} mm"
            tip = (
                f"tolerance {tolerance_m * 1000:.1f} mm"
                if tolerance_m is not None
                else None
            )
        cells.append((text, tip, status))
    return cells


def build_phase_row(phase: dict[str, Any]) -> list[tuple[str, str | None]]:
    """``(text, tooltip)`` cells for one phase's row in the phase table."""
    name = phase.get("name") or UNAVAILABLE
    fraction_cell = _format_scalar_field(phase["fraction"], unit="plain")
    hand_bar = phase["hand_bar_axis_distance_m"]
    left_cell = _format_scalar_field(hand_bar["l"], unit="mm")
    right_cell = _format_scalar_field(hand_bar["r"], unit="mm")
    return [(name, None), fraction_cell, left_cell, right_cell]


def _build_table(columns: tuple[str, ...], *, object_name: str) -> QTableWidget:
    table = QTableWidget()
    table.setObjectName(object_name)
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    return table


def _set_cell(
    table: QTableWidget, row: int, col: int, text: str, tooltip: str | None
) -> None:
    item = QTableWidgetItem(text)
    if tooltip:
        item.setToolTip(tooltip)
    table.setItem(row, col, item)


class LiftBaselinePanel(QWidget):
    """Read-only cross-engine view of one lift from the LIFT-1 baseline receipt."""

    def __init__(
        self,
        receipt: dict[str, Any] | None = None,
        initial_lift: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the panel, loading *receipt* (or the committed one) for *initial_lift*.

        Args:
            receipt: A pre-loaded baseline receipt (mainly for tests). When
                ``None``, :func:`load_baseline` loads the committed receipt.
            initial_lift: The lift to show first. Falls back to the first
                available lift when ``None`` or not present in the receipt.
            parent: Optional Qt parent widget.

        Raises:
            TypeError: If *receipt* is not ``None``/a dict, or *initial_lift*
                is not ``None``/a string.

        Postcondition:
            The widget never raises for a missing or invalid receipt; a
            ``FileNotFoundError``/``ValueError`` from loading it instead shows
            a message label (``error_message()`` becomes non-``None``) and
            all tables remain empty.
        """
        super().__init__(parent)
        if receipt is not None and not isinstance(receipt, dict):
            raise TypeError(
                f"receipt must be a dict or None, got {type(receipt).__name__}"
            )
        if initial_lift is not None and not isinstance(initial_lift, str):
            raise TypeError(
                f"initial_lift must be a str or None, got {type(initial_lift).__name__}"
            )

        self._receipt: dict[str, Any] | None = None
        self._lifts: list[str] = []
        self._lift: str | None = None
        self._view: dict[str, Any] | None = None
        self._error: str | None = None

        self._setup_ui()

        if receipt is None:
            try:
                receipt = load_baseline()
            except (FileNotFoundError, ValueError) as exc:
                self._show_error(str(exc))
                return
        try:
            lifts = available_lifts(receipt)
        except TypeError as exc:
            self._show_error(str(exc))
            return
        if not lifts:
            self._show_error("Baseline receipt has no lifts recorded.")
            return

        self._receipt = receipt
        self._lifts = lifts
        for lift in lifts:
            self._lift_selector.addItem(
                TITLES.get(lift, lift.replace("_", " ").title()), lift
            )

        start_lift = initial_lift if initial_lift in lifts else lifts[0]
        self._lift_selector.setCurrentIndex(lifts.index(start_lift))
        self._load_lift(start_lift)

    def _setup_ui(self) -> None:
        """Construct child widgets, layouts, and signal connections."""
        layout = QVBoxLayout(self)

        self._message_label = QLabel()
        self._message_label.setObjectName("lift-baseline-message")
        self._message_label.setWordWrap(True)
        self._message_label.setVisible(False)
        layout.addWidget(self._message_label)

        self._provenance_label = QLabel()
        self._provenance_label.setObjectName("lift-baseline-provenance")
        layout.addWidget(self._provenance_label)

        self._lift_row = QWidget()
        lift_row_layout = QHBoxLayout(self._lift_row)
        lift_row_layout.addWidget(QLabel("Lift:"))
        self._lift_selector = QComboBox()
        self._lift_selector.setObjectName("lift-selector")
        lift_row_layout.addWidget(self._lift_selector)
        layout.addWidget(self._lift_row)

        self._engine_table = _build_table(_ENGINE_COLUMNS, object_name="engine-table")
        layout.addWidget(self._engine_table)

        self._pose_row = QWidget()
        pose_row_layout = QHBoxLayout(self._pose_row)
        pose_row_layout.addWidget(QLabel("Pose:"))
        self._pose_selector = QComboBox()
        self._pose_selector.setObjectName("pose-selector")
        pose_row_layout.addWidget(self._pose_selector)
        layout.addWidget(self._pose_row)

        self._pair_table = _build_table(
            tuple(title for _key, title in _PAIR_METRIC_COLUMNS),
            object_name="pair-table",
        )
        layout.addWidget(self._pair_table)

        self._phase_row = QWidget()
        phase_row_layout = QHBoxLayout(self._phase_row)
        phase_row_layout.addWidget(QLabel("Phase engine:"))
        self._phase_engine_selector = QComboBox()
        self._phase_engine_selector.setObjectName("phase-engine-selector")
        phase_row_layout.addWidget(self._phase_engine_selector)
        layout.addWidget(self._phase_row)

        self._phase_table = _build_table(_PHASE_COLUMNS, object_name="phase-table")
        layout.addWidget(self._phase_table)

        self._gaps_label = QLabel("Gaps:")
        layout.addWidget(self._gaps_label)
        self._gaps_list = QListWidget()
        self._gaps_list.setObjectName("gaps-list")
        layout.addWidget(self._gaps_list)

        note = QLabel(PROVENANCE_NOTE)
        note.setObjectName("lift-baseline-note")
        layout.addWidget(note)

        self._data_widgets: tuple[QWidget, ...] = (
            self._lift_row,
            self._engine_table,
            self._pose_row,
            self._pair_table,
            self._phase_row,
            self._phase_table,
            self._gaps_label,
            self._gaps_list,
        )

        self._lift_selector.currentIndexChanged.connect(self._on_lift_changed)
        self._pose_selector.currentIndexChanged.connect(self._on_pose_changed)
        self._phase_engine_selector.currentIndexChanged.connect(
            self._on_phase_engine_changed
        )

    # -- error state ---------------------------------------------------

    def _show_error(self, message: str) -> None:
        self._error = message
        self._message_label.setText(message)
        self._message_label.setVisible(True)
        for widget in self._data_widgets:
            widget.setVisible(False)

    def error_message(self) -> str | None:
        """The load-error message, or ``None`` when a receipt is loaded."""
        return self._error

    def message_visible(self) -> bool:
        """Whether the error-message label is currently shown.

        Uses ``isVisibleTo(self)`` rather than ``isVisible()`` so this is
        correct even when the panel itself is never shown (e.g. embedded in
        a dock that is never shown, or under a headless test).
        """
        return self._message_label.isVisibleTo(self)

    def provenance_text(self) -> str:
        """The rendered schema/generated-timestamp provenance line."""
        return self._provenance_label.text()

    # -- lift/pose/phase-engine selection --------------------------------

    def current_lift(self) -> str | None:
        """The lift currently rendered, or ``None`` before any receipt loads."""
        return self._lift

    def lift_count(self) -> int:
        """Number of lifts offered by the lift selector."""
        return self._lift_selector.count()

    def pose_count(self) -> int:
        """Number of poses offered by the pose selector for the current lift."""
        return self._pose_selector.count()

    def phase_engine_count(self) -> int:
        """Number of engines offered by the phase-engine selector."""
        return self._phase_engine_selector.count()

    def select_lift(self, lift: str) -> None:
        """Programmatically switch to *lift* (as a user picking the combo would).

        Raises:
            ValueError: If *lift* is not one of the lifts available.
        """
        index = self._lift_selector.findData(lift)
        if index < 0:
            raise ValueError(
                f"unknown lift {lift!r}; available lifts are {self._lifts}"
            )
        self._lift_selector.setCurrentIndex(index)

    def select_pose(self, pose: str) -> None:
        """Programmatically switch the pair-metric table to *pose*.

        Raises:
            ValueError: If *pose* is not offered for the current lift.
        """
        index = self._pose_selector.findText(pose)
        if index < 0:
            raise ValueError(f"unknown pose {pose!r} for lift {self._lift!r}")
        self._pose_selector.setCurrentIndex(index)

    def select_phase_engine(self, engine: str) -> None:
        """Programmatically switch the phase table to *engine*.

        Raises:
            ValueError: If *engine* has no result for the current lift.
        """
        index = self._phase_engine_selector.findData(engine)
        if index < 0:
            raise ValueError(f"unknown engine {engine!r} for lift {self._lift!r}")
        self._phase_engine_selector.setCurrentIndex(index)

    def _on_lift_changed(self, index: int) -> None:
        lift = self._lift_selector.itemData(index)
        if lift is not None:
            self._load_lift(lift)

    def _on_pose_changed(self, _index: int) -> None:
        self._render_pair_table()

    def _on_phase_engine_changed(self, _index: int) -> None:
        self._render_phase_table()

    # -- rendering --------------------------------------------------------

    def _load_lift(self, lift: str) -> None:
        self._lift = lift
        self._view = lift_view(self._receipt, lift)
        self._render_provenance()
        self._render_engine_table()
        self._populate_pose_selector()
        self._populate_phase_engine_selector()
        self._render_gaps()

    def _render_provenance(self) -> None:
        receipt = self._receipt or {}
        schema = receipt.get("schema", UNAVAILABLE)
        generated = receipt.get("generated_utc", UNAVAILABLE)
        self._provenance_label.setText(f"Schema: {schema}  |  Generated: {generated}")

    def _render_engine_table(self) -> None:
        engines = self._view["engines"] if self._view else []
        table = self._engine_table
        table.setRowCount(len(engines))
        for row, engine in enumerate(engines):
            for col, (text, tooltip) in enumerate(build_engine_row(engine)):
                _set_cell(table, row, col, text, tooltip)

    def _populate_pose_selector(self) -> None:
        poses = sorted((self._view or {}).get("comparisons", {}).get("poses", {}))
        self._pose_selector.blockSignals(True)
        self._pose_selector.clear()
        self._pose_selector.addItems(poses)
        self._pose_selector.blockSignals(False)
        self._render_pair_table()

    def _render_pair_table(self) -> None:
        table = self._pair_table
        comparisons = (self._view or {}).get("comparisons", {})
        pairs = comparisons.get("poses", {}).get(self._pose_selector.currentText(), {})
        pair_names = sorted(pairs)
        table.setRowCount(len(pair_names))
        table.setVerticalHeaderLabels([_pair_title(pair) for pair in pair_names])
        for row, pair in enumerate(pair_names):
            for col, (text, tooltip, status) in enumerate(
                build_pair_metric_cells(pairs[pair])
            ):
                item = QTableWidgetItem(text)
                if tooltip:
                    item.setToolTip(tooltip)
                item.setBackground(
                    _STATUS_COLORS.get(status, _STATUS_COLORS["unavailable"])
                )
                table.setItem(row, col, item)

    def _populate_phase_engine_selector(self) -> None:
        engines = [engine["engine"] for engine in (self._view or {}).get("engines", [])]
        self._phase_engine_selector.blockSignals(True)
        self._phase_engine_selector.clear()
        for engine in engines:
            self._phase_engine_selector.addItem(
                ENGINE_TITLES.get(engine, engine), engine
            )
        self._phase_engine_selector.blockSignals(False)
        self._render_phase_table()

    def _render_phase_table(self) -> None:
        table = self._phase_table
        engine = self._phase_engine_selector.currentData()
        engines_by_name = {
            e["engine"]: e for e in (self._view or {}).get("engines", [])
        }
        phases = engines_by_name.get(engine, {}).get("phases", []) if engine else []
        table.setRowCount(len(phases))
        for row, phase in enumerate(phases):
            for col, (text, tooltip) in enumerate(build_phase_row(phase)):
                _set_cell(table, row, col, text, tooltip)

    def _render_gaps(self) -> None:
        self._gaps_list.clear()
        for gap in (self._view or {}).get("gaps", []):
            engines_text = ", ".join(
                ENGINE_TITLES.get(engine, engine) for engine in gap.get("engines", [])
            )
            item = QListWidgetItem(f"{gap.get('title', '')} ({engines_text})")
            evidence = "\n".join(gap.get("evidence", []))
            if evidence:
                item.setToolTip(evidence)
            self._gaps_list.addItem(item)

    # -- read-only accessors for tests (Law of Demeter: no widget chains) --

    def _table(self, table_name: str) -> QTableWidget:
        tables = {
            "engines": self._engine_table,
            "pairs": self._pair_table,
            "phases": self._phase_table,
        }
        if table_name not in tables:
            raise ValueError(
                f"unknown table {table_name!r}; valid names are {list(tables)}"
            )
        return tables[table_name]

    def engine_row_count(self) -> int:
        """Rows currently rendered in the per-engine table."""
        return self._engine_table.rowCount()

    def pair_row_count(self) -> int:
        """Rows currently rendered in the pair-metric table."""
        return self._pair_table.rowCount()

    def phase_row_count(self) -> int:
        """Rows currently rendered in the phase table."""
        return self._phase_table.rowCount()

    def gap_count(self) -> int:
        """Number of gap entries currently listed."""
        return self._gaps_list.count()

    def cell_text(self, table_name: str, row: int, col: int) -> str:
        """Text of one cell. *table_name* is one of ``"engines"``, ``"pairs"``, ``"phases"``."""
        item = self._table(table_name).item(row, col)
        return item.text() if item is not None else ""

    def cell_tooltip(self, table_name: str, row: int, col: int) -> str | None:
        """Tooltip of one cell, or ``None`` when the cell has none."""
        item = self._table(table_name).item(row, col)
        return item.toolTip() if item is not None else None

    def gap_text(self, row: int) -> str:
        """Text of one gaps-list row."""
        item = self._gaps_list.item(row)
        return item.text() if item is not None else ""

    def gap_tooltip(self, row: int) -> str | None:
        """Tooltip (evidence) of one gaps-list row, or ``None`` when it has none."""
        item = self._gaps_list.item(row)
        return item.toolTip() if item is not None else None
