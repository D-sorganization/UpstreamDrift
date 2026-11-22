"""Shot card of the live Results Playback: impact factors and ball flight (#15150).

The numbers are the native-viewer exporter's, never recomputed: the strike is
prepared by ``core.prepare_shot``, the factors come from
``impact_factors.factors_for_export`` and the flight (Rust ``upstream_physics``
wheel) from ``flight_tracer.attach_flight``, so the card agrees with a clip
exported with ``--impact-factors --ball-flight``. :func:`summary_for_path` and
:func:`shot_rows` are Qt-free; :class:`ShotCard` imports PyQt6 lazily.
Options: ``playback_flight_panel_settings.json`` (see its ``sources``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from src.tools.native_viewer_export.display_units import (
    DisplayUnits,
    distance_text,
    load_display_units,
    speed_text,
)
from src.tools.native_viewer_export.flight_tracer import NO_CONTACT_STATUS

SETTINGS_PATH = Path(__file__).with_name("playback_flight_panel_settings.json")
_FACTOR_KEYS = {
    "club_speed_mph": "clubhead_speed",
    "face_to_path_deg": "face_to_path_deg",
    "attack_angle_deg": "attack_angle_deg",
}
#: (section index, label, summary field, format) of every numeric row
_ROWS = (
    (0, "Club Speed", "club_speed_mph", "speed"),
    (0, "Face to Path", "face_to_path_deg", "signed"),
    (0, "Attack Angle", "attack_angle_deg", "signed"),
    (1, "Ball Speed", "ball_speed_mph", "speed"),
    (1, "Launch Angle", "launch_deg", "angle"),
    (1, "Spin Rate", "spin_rpm", "spin"),
    (1, "Spin Axis", "spin_axis_deg", "sided_angle"),
    (2, "Carry", "carry_m", "distance"),
    (2, "Total", "total_m", "distance"),
    (2, "Apex", "apex_m", "distance"),
    (2, "Side", "side_m", "sided_distance"),
)
_IMPACT_ROWS = 3  # rows shown whatever happened to the ball


@dataclass(frozen=True)
class ShotCardSettings:
    """Card options (meanings: the settings JSON's ``sources``)."""

    title: str
    sections: tuple[str, str, str]
    blank_text: str
    flight_unavailable_text: str
    needs_bundle_text: str
    default_club: str
    club_labels: Mapping[str, str]
    angle_decimals: int
    spin_decimals: int
    side_left_text: str
    side_right_text: str
    min_width_px: int

    def __post_init__(self) -> None:
        if len(self.sections) != 3 or not isinstance(self.club_labels, Mapping):
            raise ValueError("sections needs 3 names and club_labels a mapping")
        if min(self.angle_decimals, self.spin_decimals, self.min_width_px) < 0:
            raise ValueError("decimals and min_width_px must not be negative")


def load_card_settings(path: Path | None = None) -> ShotCardSettings:
    """Settings from ``path`` (default: the file beside this module)."""
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    try:
        return ShotCardSettings(**{**document, "sections": tuple(document["sections"])})
    except (TypeError, KeyError) as exc:
        raise ValueError(f"shot card settings: {exc}") from exc


@dataclass(frozen=True)
class ShotSummary:
    """The exporter's shot in SI and mph (``None``: not computed).

    ``flight_reason`` says why there is no flight (``None``: flight shown);
    ``contact`` is the exporter's contact verdict; ``side_m`` is + = right.
    """

    status: str
    contact: bool = True
    club_speed_mph: float | None = None
    face_to_path_deg: float | None = None
    attack_angle_deg: float | None = None
    ball_speed_mph: float | None = None
    launch_deg: float | None = None
    spin_rpm: float | None = None
    spin_axis_deg: float | None = None
    carry_m: float | None = None
    total_m: float | None = None
    apex_m: float | None = None
    side_m: float | None = None
    flight_reason: str | None = None


@dataclass(frozen=True)
class ShotRow:
    """One ``label: value`` line of the card in its ``section``."""

    section: str
    label: str
    value: str


def club_label(spec_bytes: bytes, settings: ShotCardSettings) -> str:
    """Club label of the flight and roll models for the bundle's club name."""
    try:
        club = json.loads(spec_bytes).get("club")
    except (ValueError, AttributeError):
        club = None
    name = str(club.get("name", "")) if isinstance(club, Mapping) else ""
    return str(settings.club_labels.get(name, name or settings.default_club))


def summary_from_shot(panel: Any, flight: Any) -> ShotSummary:
    """Summary of an exporter factors ``panel`` and ``FlightPlayback``.

    Either may be ``None``; a flight without a path gives its status as the
    ``flight_reason``.
    """
    none = "not computed"
    card = None if panel is None else panel.card
    rows = {} if card is None else {row.key: row.value for row in card.rows}
    verdict = None if panel is None else panel.verdict
    base = ShotSummary(
        none if panel is None else str(panel.status),
        verdict is None or bool(verdict.contact),
        **{name: rows.get(key) for name, key in _FACTOR_KEYS.items()},
    )
    if flight is None or flight.path_m is None:
        return replace(base, flight_reason=none if flight is None else flight.status)
    aimed = flight.aim_lateral_m
    return replace(
        base,
        ball_speed_mph=flight.ball_speed_mph,
        launch_deg=flight.launch_deg,
        spin_rpm=flight.spin_rpm,
        spin_axis_deg=flight.spin_axis_deg,
        carry_m=flight.carry_m,
        total_m=None if flight.rollout is None else flight.rollout.total_m,
        apex_m=flight.apex_m,
        side_m=flight.lateral_m if aimed is None else aimed,
    )


def evaluate_swing(swing: Any) -> ShotSummary:
    """The exporter's shot of ``swing`` (a ``core.SwingInput``).

    As ``runner.run_export``: the impact is detected when the bundle records
    none, then ``core.export_swing``'s strike, factors and flight.
    """
    from src.tools.native_viewer_export import core, flight_tracer
    from src.tools.native_viewer_export import impact_factors as factors
    from src.tools.native_viewer_export.overlay import detect_impact_time_s

    export = core.ExportSettings(
        impact_factors=factors.load_factor_settings(),
        ball_flight=flight_tracer.load_tracer_settings(None, load_display_units()),
        speed_unit="mph",  # the summary is unit-free; the rows convert
    )
    if not swing.has_impact_provenance:
        try:
            export = replace(export, impact_time_s=detect_impact_time_s(swing))
        except (core.BackendUnavailable, ValueError, KeyError, RuntimeError):
            pass  # as the exporter: the panel then says "no impact detected"
    try:
        swing, export, _ = core.prepare_shot(swing, export)
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        return ShotSummary(f"unavailable: {exc}", flight_reason=str(exc))
    panel, export = factors.factors_for_export(swing, export)
    shown = core.ClipOverlays(factors=panel)
    shown = flight_tracer.attach_flight(shown, swing, export)
    return summary_from_shot(shown.factors, shown.flight)


def summary_for_path(path: str | Path) -> ShotSummary:
    """The shot of the run at ``path``; a non-bundle run says it needs one."""
    from src.shared.python.motion_matching.same_input.bundle import InputBundle
    from src.tools.native_viewer_export.core import SwingInput

    cfg = load_card_settings()
    try:
        bundle = InputBundle.load(Path(path))
    except (KeyError, ValueError, OSError):
        return ShotSummary(cfg.needs_bundle_text, flight_reason=cfg.needs_bundle_text)
    club = club_label(bundle.spec_bytes, cfg)
    q, v, engine = bundle.reference_q, bundle.reference_v, bundle.reference_engine
    return evaluate_swing(SwingInput(bundle, q, Path(path).stem, club, engine, v=v))


def _text(value: float, kind: str, units: DisplayUnits, cfg: ShotCardSettings) -> str:
    deg = cfg.angle_decimals
    if kind == "speed":
        return speed_text(value, units)
    if kind == "distance":
        return distance_text(value, units)
    if kind == "spin":
        return f"{value:.{cfg.spin_decimals}f} rpm"
    if kind in ("angle", "signed"):
        return f"{value:{'+' if kind == 'signed' else ''}.{deg}f} deg"
    side = cfg.side_right_text if value >= 0.0 else cfg.side_left_text
    plain = "distance" if kind == "sided_distance" else "angle"
    return f"{_text(abs(value), plain, units, cfg)} {side}"


def shot_rows(
    summary: ShotSummary, units: DisplayUnits, settings: ShotCardSettings
) -> tuple[ShotRow, ...]:
    """Card rows of ``summary`` in ``units``: status, impact, launch, flight.

    Without contact the launch and flight rows give way to the exporter's
    "no contact: no flight", without a flight to the reason; a value not
    computed shows ``blank_text``, never 0. ``TypeError`` for a wrong type.
    """
    if not isinstance(summary, ShotSummary):
        raise TypeError("summary must be a ShotSummary")
    if not isinstance(units, DisplayUnits):
        raise TypeError("units must be DisplayUnits")
    names = settings.sections
    rows = [ShotRow(names[0], "Status", summary.status)]
    for section, label, name, kind in _ROWS:
        value = getattr(summary, name)
        ok = value is not None and math.isfinite(value)
        text = _text(value, kind, units, settings) if ok else settings.blank_text
        rows.append(ShotRow(names[section], label, text))
    if not summary.contact:
        why = NO_CONTACT_STATUS
    elif summary.flight_reason is not None:
        why = f"{settings.flight_unavailable_text}: {summary.flight_reason}"
    else:
        return tuple(rows)
    return (*rows[: 1 + _IMPACT_ROWS], ShotRow(names[1], "Ball Flight", why))


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
