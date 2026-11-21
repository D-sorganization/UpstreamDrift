"""On-screen impact factors panel of native exports (#15150).

A fixed-size panel drawn on every frame of a clip: the launch-monitor rows of
the shared impact card (:mod:`impact_card`, computed by ``impact_parameters``
from the displayed clubhead series), blank before impact and frozen at their
impact values from the impact frame to the end of the clip. Options are read
from ``impact_factors_settings.json`` (see its ``sources``). Without a known
impact the panel says so and shows no numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.impact_parameters import ClubheadSeries, ImpactParameters
from src.shared.python.impact_parameters.panel_model import (
    ImpactCard,
    build_impact_card,
)
from src.tools.native_viewer_export import compositor, impact_card

if TYPE_CHECKING:
    from src.tools.native_viewer_export.ball import BallPlacement
    from src.tools.native_viewer_export.core import ExportSettings, SwingInput

SETTINGS_PATH = Path(__file__).with_name("impact_factors_settings.json")
TITLE = "Impact Factors"
BEFORE, AT_IMPACT, NO_IMPACT = "before impact", "at impact", "no impact detected"
POSITIONS = ("right", "left")
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

    blank_text: str
    rows: tuple[str, ...]
    position: str
    opacity: float
    font_height_divisor: int
    min_font_px: int
    widest_value: str

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
        if not isinstance(self.settings, FactorSettings):
            raise TypeError("settings must be a FactorSettings")
        if self.impact_time_s is not None and not math.isfinite(self.impact_time_s):
            raise ValueError("impact_time_s must be finite or None")

    def lines_at(self, time_s: float) -> list[tuple[str, str]]:
        """``(label, value)`` lines at swing time ``time_s``: status, then rows."""
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self.impact_time_s is None:
            return [("Status", self.status), *self.settings.blank_rows]
        if time_s >= self.impact_time_s:
            return [("Status", self.status), *self.frozen]
        return [("Status", BEFORE), *self.settings.blank_rows]

    def _box(self, shape: tuple[int, ...]) -> tuple[tuple[int, int, int, int], int]:
        cfg = self.settings
        size = max(cfg.min_font_px, int(shape[0]) // cfg.font_height_divisor)
        reserve = (cfg.blank_text, cfg.widest_value)
        widest = [("Status", max(BEFORE, self.status, key=len), False)]
        widest += [
            (lab, max(val, *reserve, key=len), False) for lab, val in self.frozen
        ]
        return compositor.text_panel_box(shape, TITLE, widest, cfg.position, size)

    def bbox(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Pixel box ``(x0, y0, x1, y1)`` of the panel in a frame of ``shape``."""
        return self._box(shape)[0]

    def draw(self, img: compositor.Image8, time_s: float) -> compositor.Image8:
        """Copy of ``img`` with the panel at ``time_s``; only :meth:`bbox` changes."""
        box, size = self._box(img.shape)
        blank = self.settings.blank_text
        lines = [(lab, val, val == blank) for lab, val in self.lines_at(time_s)]
        return compositor.draw_text_panel(
            img, box, size, TITLE, lines, self.settings.opacity
        )


def _rows(card: ImpactCard, settings: FactorSettings) -> Lines:
    by_key = {row.key: row for row in card.rows}
    text = compositor._card_value_text  # one value format for both panels
    return tuple((ROW_LABELS[key], text(by_key[key])) for key in settings.rows)


def no_impact_panel(settings: FactorSettings, status: str = NO_IMPACT) -> FactorsPanel:
    """Panel that shows ``status`` and blank rows on every frame."""
    return FactorsPanel(settings, None, status, settings.blank_rows)


def panel_from_series(
    series: ClubheadSeries,
    impact_time_s: float,
    ball: BallPlacement | None,
    settings: FactorSettings,
) -> FactorsPanel:
    """Panel of ``series`` with the impact card rows frozen at ``impact_time_s``.

    Raises ``TypeError``/``ValueError`` for bad settings, a non-finite time or
    a series the impact card cannot be computed from.
    """
    if not isinstance(settings, FactorSettings):
        raise TypeError("settings must be a FactorSettings")
    card = impact_card.card_from_series(series, impact_time_s, ball)
    return FactorsPanel(settings, impact_time_s, AT_IMPACT, _rows(card, settings))


def factors_for_export(
    swing: SwingInput, settings: ExportSettings
) -> FactorsPanel | None:
    """The export's panel, or ``None`` when the option or the HUD is off.

    Without a known impact time (none set, none in the bundle) the panel says
    "no impact detected"; any failure gives a panel stating the reason. Neither
    shows a number.
    """
    config: Any = settings.impact_factors
    if config is None or not settings.hud:
        return None
    impact = settings.impact_time_s
    if impact is None:
        if not swing.has_impact_provenance:
            return no_impact_panel(config)
        impact = swing.impact_time_s
    try:
        return panel_from_series(club_series(swing), impact, swing.ball, config)
    except impact_card._EXPECTED_ERRORS as exc:
        return no_impact_panel(config, compositor._short(f"unavailable: {exc}"))
