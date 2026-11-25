"""On-screen impact factors panel of native exports (#15150).

A fixed-size panel drawn on every frame of a clip: the launch-monitor rows of
the shared impact card (:mod:`impact_card`, computed by ``impact_parameters``
from the displayed clubhead series), blank or live before impact and frozen at
their impact values from the impact frame to the end of the clip. Options are
read from ``impact_factors_settings.json`` (see its ``sources``). Without a
known impact the panel says so and shows no numbers; a face that passes the
ball without touching it is labelled "closest approach, no contact".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    ImpactParameters,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.panel_model import (
    ImpactCard,
    build_impact_card,
)
from src.tools.native_viewer_export import compositor, contact_verdict, impact_card

if TYPE_CHECKING:
    from src.tools.native_viewer_export.ball import BallPlacement
    from src.tools.native_viewer_export.core import ExportSettings, SwingInput

SETTINGS_PATH = Path(__file__).with_name("impact_factors_settings.json")
TITLE = "Impact Factors"
BEFORE, AT_IMPACT, NO_IMPACT = "before impact", "at impact", "no impact detected"
NO_CONTACT = "closest approach, no contact"
NOT_DETECTED = "closest approach (impact not detected)"
POSITIONS = ("right", "left")
GRID_ANCHORS = ("bottom_right", "bottom_left", "top_right", "top_left")
PRE_IMPACT_MODES = ("blank", "live")
Lines = tuple[tuple[str, str], ...]
club_series = impact_card._club_series  # the card's own face series


def _row_labels() -> dict[str, str]:
    """Key -> label of every row the shared card model builds."""
    card = build_impact_card(ImpactParameters({}, 0, 0.0, "", 0.0, 0.0, 0.0))
    return {row.key: row.label for row in card.rows}


ROW_LABELS = _row_labels()


@dataclass(frozen=True)
class FactorSettings:
    """Validated ``impact_factors_settings.json`` (``TypeError``/``ValueError``)."""

    pre_impact: str
    blank_text: str
    rows: tuple[str, ...]
    position: str
    opacity: float
    font_height_divisor: int
    min_font_px: int
    widest_value: str
    closest_approach_fallback: bool
    grid_anchor: str = "bottom_right"
    grid_font_tile_fraction: float = 0.056
    grid_min_font_px: int = 16

    def __post_init__(self) -> None:
        if self.pre_impact not in PRE_IMPACT_MODES:
            raise ValueError(f"pre_impact must be one of {PRE_IMPACT_MODES}")
        if not isinstance(self.closest_approach_fallback, bool):
            raise TypeError("closest_approach_fallback must be true or false")
        if isinstance(self.rows, str):
            raise TypeError("rows must be a list of row keys")
        object.__setattr__(self, "rows", tuple(self.rows))
        if not self.rows or not set(self.rows) <= set(ROW_LABELS):
            raise ValueError(f"rows must be keys of {sorted(ROW_LABELS)}")
        if self.position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must lie in [0, 1]")
        if min(self.font_height_divisor, self.min_font_px) < 1:
            raise ValueError("font_height_divisor and min_font_px must be positive")
        if self.grid_anchor not in GRID_ANCHORS:
            raise ValueError(f"grid_anchor must be one of {GRID_ANCHORS}")
        fraction = self.grid_font_tile_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, int | float):
            raise TypeError("grid_font_tile_fraction must be a number")
        if not (math.isfinite(fraction) and 0.0 < fraction <= 1.0):
            raise ValueError("grid_font_tile_fraction must lie in (0, 1]")
        value = self.grid_min_font_px
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("grid_min_font_px must be an integer")
        if value < 1:
            raise ValueError("grid_min_font_px must be positive")

    @property
    def grid_layout(self) -> compositor.GridLayout:
        """The panel placement in a composed 2x2 grid frame."""
        return compositor.GridLayout(
            self.grid_anchor, self.grid_font_tile_fraction, self.grid_min_font_px
        )

    @property
    def blank_rows(self) -> Lines:
        """Every row with the blank text in place of a value."""
        return tuple((ROW_LABELS[key], self.blank_text) for key in self.rows)


def load_factor_settings(path: Path | None = None) -> FactorSettings:
    """Read a settings file (default the one beside this module)."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    raw.pop("sources", None)
    return FactorSettings(**raw)


@dataclass(frozen=True)
class FactorsPanel:
    """The panel of one swing: what it shows at each time and how it is drawn.

    ``impact_time_s`` is ``None`` when no impact is known (``status`` then
    says why and no number is shown). Postcondition: the line count and
    :meth:`bbox` do not depend on the time shown.
    """

    settings: FactorSettings
    impact_time_s: float | None
    status: str
    frozen: Lines
    series: ClubheadSeries | None = None  # source of the live rows
    card: ImpactCard | None = field(default=None, compare=False, repr=False)
    verdict: contact_verdict.ContactVerdict | None = field(
        default=None, compare=False, repr=False
    )  # the one no-contact decision, read by the flight too
    _live: dict[int, Lines] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.settings, FactorSettings):
            raise TypeError("settings must be a FactorSettings")
        if self.impact_time_s is not None and not math.isfinite(self.impact_time_s):
            raise ValueError("impact_time_s must be finite or None")

    def with_model_smash(self, ball_speed_mph: float | None) -> FactorsPanel:
        """Panel whose smash row is that of the flight model's ball speed.

        The same row as the impact card's (:func:`impact_card.with_model_smash`);
        ``None``, or a panel without its card, returns ``self``.
        """
        if self.card is None:
            return self
        card = impact_card.with_model_smash(self.card, ball_speed_mph)
        frozen = _rows(card, self.settings)
        return replace(self, card=card, frozen=frozen)

    def lines_at(self, time_s: float) -> list[tuple[str, str]]:
        """``(label, value)`` lines at swing time ``time_s``: status, then rows."""
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self.impact_time_s is None:
            return [*_status_lines(self.status), *self.settings.blank_rows]
        if time_s >= self.impact_time_s:
            return [*_status_lines(self.status), *self.frozen]
        return [*_status_lines(BEFORE), *self._live_rows(time_s)]

    def _live_rows(self, time_s: float) -> Lines:
        """Rows of the source sample nearest ``time_s`` (blank if not computable)."""
        if self.settings.pre_impact != "live" or self.series is None:
            return self.settings.blank_rows
        index = int(np.argmin(np.abs(self.series.times_s - time_s)))
        if index not in self._live:
            try:
                params = extract_impact_parameters(
                    self.series,
                    impact_card.address_target_frame(self.series),
                    impact_index=index,
                    use_tools=False,
                )
                rows = _rows(build_impact_card(params), self.settings)
            except ValueError:  # e.g. club head slower than the extractor minimum
                rows = self.settings.blank_rows
            self._live[index] = rows
        return self._live[index]

    def _widest(self) -> list[tuple[str, str, bool]]:
        """Lines at their widest values: the panel size never depends on time."""
        reserve = (self.settings.blank_text, self.settings.widest_value)
        widest = [
            (lab, val, False)
            for lab, val in _status_lines(max(BEFORE, self.status, key=len))
        ]
        widest += [
            (lab, max(val, *reserve, key=len), False) for lab, val in self.frozen
        ]
        return widest

    def _box(
        self, shape: tuple[int, ...], grid: bool = False
    ) -> tuple[tuple[int, int, int, int], int]:
        cfg = self.settings
        if grid:
            return self._grid_box(shape)
        size = max(cfg.min_font_px, int(shape[0]) // cfg.font_height_divisor)
        return compositor.text_panel_box(
            shape, TITLE, self._widest(), cfg.position, size
        )

    def _grid_box(
        self, shape: tuple[int, ...]
    ) -> tuple[tuple[int, int, int, int], int]:
        """Box in the ``grid_anchor`` tile corner of a composed 2x2 frame."""
        layout = self.settings.grid_layout
        return compositor.grid_text_panel_box(shape, TITLE, self._widest(), layout)

    def bbox(
        self, shape: tuple[int, ...], grid: bool = False
    ) -> tuple[int, int, int, int]:
        """Pixel box ``(x0, y0, x1, y1)`` of the panel in a frame of ``shape``.

        ``grid`` places it in the ``grid_anchor`` tile of a composed 2x2 frame.
        """
        return self._box(shape, grid)[0]

    def draw(
        self, img: compositor.Image8, time_s: float, grid: bool = False
    ) -> compositor.Image8:
        """Copy of ``img`` with the panel at ``time_s``; only :meth:`bbox` changes."""
        box, size = self._box(img.shape, grid)
        blank = self.settings.blank_text
        lines = [(lab, val, val == blank) for lab, val in self.lines_at(time_s)]
        return compositor.draw_text_panel(
            img, box, size, TITLE, lines, self.settings.opacity
        )

    def draw_all(
        self, outputs: dict[str, compositor.Image8], time_s: float, grid_view: str
    ) -> dict[str, compositor.Image8]:
        """``outputs`` with the panel on each; the ``grid_view`` one gets the grid layout."""
        return {
            name: self.draw(frame, time_s, grid=name == grid_view)
            for name, frame in outputs.items()
        }


def draw_card_all(
    outputs: dict[str, compositor.Image8], card: Any, grid_view: str
) -> dict[str, compositor.Image8]:
    """``outputs`` with the impact card on each; the ``grid_view`` one gets the
    grid placement of ``impact_factors_settings.json``."""
    layout = load_factor_settings().grid_layout
    return {
        name: compositor.draw_impact_card(
            frame, card, grid=layout if name == grid_view else None
        )
        for name, frame in outputs.items()
    }


def _status_lines(status: str) -> Lines:
    """``status`` wrapped over lines (never cut), the first one labelled."""
    wrapped = compositor._wrap_reason(status)
    return tuple(("Status" if i == 0 else "", text) for i, text in enumerate(wrapped))


def _rows(card: ImpactCard, settings: FactorSettings) -> Lines:
    by_key = {row.key: row for row in card.rows}
    text = compositor._card_value_text  # one value format for both panels
    return tuple((ROW_LABELS[key], text(by_key[key])) for key in settings.rows)


def no_impact_panel(settings: FactorSettings, status: str = NO_IMPACT) -> FactorsPanel:
    """Panel that shows ``status`` and blank rows on every frame."""
    return FactorsPanel(settings, None, status, settings.blank_rows)


def closest_approach(
    series: ClubheadSeries, ball: BallPlacement
) -> tuple[float, float]:
    """``(time_s, gap_m)`` of the face centre's closest pass of the ball centre.

    The face-centre path is piecewise linear; only segments after the top of
    the backswing count (at address the face rests beside the ball).
    """
    from src.shared.python.model_appearance.club_face import top_of_backswing_index

    t, head = series.times_s, np.asarray(series.face_center_m, dtype=float)
    try:
        top = top_of_backswing_index(t, head)
    except ValueError:
        top = 0
    centre = np.asarray(ball.position_m, dtype=float)
    start, step = head[top:-1], np.diff(head[top:], axis=0)
    length2 = np.maximum(np.einsum("ij,ij->i", step, step), 1e-18)
    s = np.clip(np.einsum("ij,ij->i", centre - start, step) / length2, 0.0, 1.0)
    gap = np.linalg.norm(start + s[:, None] * step - centre, axis=1)
    j = int(np.argmin(gap))
    k = top + j
    return float(t[k] + s[j] * (t[k + 1] - t[k])), float(gap[j])


def panel_from_series(
    series: ClubheadSeries,
    impact_time_s: float,
    ball: BallPlacement | None,
    settings: FactorSettings,
    *,
    detected: bool = True,
    speed_unit: str | None = None,
) -> FactorsPanel:
    """Panel of ``series`` with the impact card rows frozen at ``impact_time_s``.

    With a ``ball`` whose contact point lies off the face the status reads
    "closest approach, no contact" with the miss distance (face centre to ball
    centre); ``detected=False`` marks an instant that is only the closest
    approach. Raises ``TypeError``/``ValueError`` for bad settings, a
    non-finite time or a series the impact card cannot be computed from.
    """
    if not isinstance(settings, FactorSettings):
        raise TypeError("settings must be a FactorSettings")
    card = impact_card.card_from_series(series, impact_time_s, ball, speed_unit)
    verdict = contact_verdict.verdict_from_card(card, ball)
    status = AT_IMPACT if detected else NOT_DETECTED
    if ball is not None and not verdict.contact:
        miss_mm = closest_approach(series, ball)[1] * 1000.0
        status = f"{NO_CONTACT} (miss {miss_mm:.0f} mm)"
    rows = _rows(card, settings)
    return FactorsPanel(settings, impact_time_s, status, rows, series, card, verdict)


def factors_for_export(
    swing: SwingInput, settings: ExportSettings
) -> tuple[FactorsPanel | None, ExportSettings]:
    """The export's panel (``None`` when off) and the settings to export with.

    Without a known impact time (none set, none in the bundle) the panel says
    "no impact detected", unless the closest-approach fallback is on: then that
    instant becomes the impact time of the returned settings and the panel is
    labelled so. Any failure gives a panel stating the reason, without numbers.
    """
    config: Any = settings.impact_factors
    if config is None or not settings.hud:
        return None, settings
    known = settings.impact_time_s is not None or swing.has_impact_provenance
    try:
        series = club_series(swing)
        if not known:
            if not config.closest_approach_fallback or swing.ball is None:
                return no_impact_panel(config), settings
            when = closest_approach(series, swing.ball)[0]
            settings = replace(settings, impact_time_s=when)
        impact = settings.impact_time_s
        if impact is None:
            impact = swing.impact_time_s
        panel = panel_from_series(
            series,
            impact,
            swing.ball,
            config,
            detected=known,
            speed_unit=settings.speed_unit,
        )
    except impact_card._EXPECTED_ERRORS as exc:
        reason = f"unavailable: {exc}"
        return no_impact_panel(config, reason), settings
    return panel, settings
