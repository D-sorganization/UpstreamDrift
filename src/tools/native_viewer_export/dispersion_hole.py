"""Dispersion on the hole: every shot of a dispersion run on one hole (#15150).

All sample traces with their carry and rest points on the hole plan and side
profile, the nominal shot drawn over them like the export's shot (thin
translucent samples, the bold shot on top: ``dispersion_traces_settings.json``),
the mean landing point, a spread ellipse and the mean +/- standard
deviation of carry and side. The hole and the shot drawing are
``golf_environment`` (``shots_on_hole`` and ``hole_figure``); this module feeds
them the dispersion shots and adds the spread overlays. The hole is aimed along
the address line: a dispersion has no single finish line. Options and every
number: ``dispersion_hole_settings.json`` (see ``sources``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.native_viewer_export.dispersion_traces import (
    DispersionTraceSettings,
    HoleLineStyle,
    draw_on_hole,
    load_trace_settings,
)
from src.tools.native_viewer_export.display_units import (
    DisplayUnits,
    load_display_units,
)
from src.tools.native_viewer_export.shot_dispersion import (
    INPUTS,
    DispersionResult,
    DispersionSettings,
    SampleOutcome,
    landing_ellipse,
)

SETTINGS_PATH = Path(__file__).with_name("dispersion_hole_settings.json")
_ALPHAS = ("ellipse_fill_alpha", "ellipse_edge_alpha")
_RGBS = ("mean_rgb", "ellipse_rgb")
_POSITIVE = ("mean_marker_size", "ellipse_linewidth")


@dataclass(frozen=True)
class DispersionHoleSettings:
    """Validated options of the dispersion hole still (``style`` is the JSON).

    ``traces`` gives the export's nominal-versus-sample line styling.
    """

    preset: str
    style: Mapping[str, Any]
    traces: DispersionTraceSettings = field(default_factory=load_trace_settings)

    def __post_init__(self) -> None:
        if not self.preset:
            raise ValueError("preset must name a hole preset")
        for key in _ALPHAS:
            if not 0.0 <= float(self.style[key]) <= 1.0:
                raise ValueError(f"{key} must be in 0..1")
        for key in _RGBS:
            if len(self.style[key]) != 3 or not all(
                0 <= c <= 1 for c in self.style[key]
            ):
                raise ValueError(f"{key} must be three channels in 0..1")
        for key in _POSITIVE:
            if not (math.isfinite(self.style[key]) and self.style[key] > 0.0):
                raise ValueError(f"{key} must be positive and finite")
        missing = set(INPUTS) - set(self.style["input_names"])
        if missing:
            raise ValueError(f"input_names lack {sorted(missing)}")


def load_dispersion_hole_settings(
    path: Path | None = None, preset: str | None = None
) -> DispersionHoleSettings:
    """Settings from ``path`` (default: beside this module).

    ``preset`` replaces the file's hole preset and must be one of
    ``shots_settings.json``. Raises ``ValueError`` for a bad or unknown entry.
    """
    from src.tools.golf_environment.shots import load_shots_settings

    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    raw.pop("sources", None)
    name = preset or raw.pop("preset")
    raw.pop("preset", None)
    known = load_shots_settings()["hole"]["presets"]
    if name not in known:
        raise ValueError(f"preset must be one of {sorted(known)}, not {name}")
    return DispersionHoleSettings(name, raw)


def dispersion_hole_shots(
    result: DispersionResult, hole: Any, cfg: DispersionHoleSettings
) -> list[Any]:
    """One ``HoleShot`` per available shot, on the tee of ``hole``.

    Preconditions: at least one available shot (``ValueError``). Postcondition:
    each path starts on the tee and keeps the flight model's carry and side;
    the roll of a shot that has one is added at its rest point.
    """
    shots = result.available
    if not shots:
        raise ValueError("no shot has a flight to draw")
    return _hole_shots(shots, hole, cfg.traces.hole_sample_style().colour)


def _hole_shots(
    shots: list[SampleOutcome], hole: Any, colour: tuple[float, float, float, float]
) -> list[Any]:
    """``shots`` (each with a flight) as ``HoleShot``s of ``colour`` on ``hole``."""
    from src.tools.golf_environment.shots import TargetFlight, shots_on_hole

    flights = []
    for o in shots:
        local = np.column_stack([o.trace[:, 0], -o.trace[:, 1], o.trace[:, 2]])
        flights.append(
            TargetFlight(
                local,
                float(o.metrics["carry_m"] or 0.0),
                float(o.metrics["side_m"] or 0.0),
            )
        )
    labels = [f"Shot {i + 1}" for i in range(len(shots))]
    placed = shots_on_hole(flights, hole, labels, [colour] * len(shots))
    out = []
    for shot, o in zip(placed, shots, strict=True):
        if o.roll_m is not None and o.roll_offset_m is not None:
            shot = replace(
                shot,
                roll_m=o.roll_m,
                total_m=shot.carry_m + o.roll_m,
                roll_offset_m=o.roll_offset_m,
            )
        out.append(shot)
    return out


def title_note(result: DispersionResult, cfg: DispersionHoleSettings) -> str:
    """Second title line: replay-level dispersion, varied inputs and the aim."""
    names = ", ".join(cfg.style["input_names"][k] for k in INPUTS)
    return cfg.style["title_format"].format(n=len(result.available), inputs=names)


def stats_line(
    result: DispersionResult, cfg: DispersionHoleSettings, units: DisplayUnits
) -> str:
    """Mean +/- standard deviation of carry and side in display units."""
    return spread_text(result, cfg.style["legend_format"]["stats"], units)


def spread_text(result: DispersionResult, fmt: str, units: DisplayUnits) -> str:
    """``fmt`` filled with ``n`` and mean +/- sd of carry and side in ``units``.

    Raises ``ValueError`` when no available shot has a carry and a side.
    """
    summary = result.summary()
    carry, side = summary["carry_m"], summary["side_m"]
    if carry is None or side is None:
        raise ValueError("no shot has a carry and a side")
    k = units.distance_scale
    return fmt.format(
        n=int(carry["n"]),
        carry_mean=k * carry["mean"],
        carry_sd=k * carry["std"],
        side_mean=k * side["mean"],
        side_sd=k * side["std"],
        unit=units.distance_unit,
    )


def _plan_points(shots: list[Any], hole: Any, scale: float) -> np.ndarray:
    """(n, 2) (down-range, left) carry points of ``shots`` in display units."""
    from src.tools.golf_environment.shots import hole_axes

    tee, forward, left = hole_axes(hole)
    rel = np.array([np.asarray(s.path_m, dtype=float)[-1] - tee for s in shots])
    return scale * np.column_stack([rel @ forward, rel @ left])


def _overlay_spread(
    axes: tuple[Any, Any],
    points: np.ndarray,
    sigma_cfg: DispersionSettings,
    cfg: DispersionHoleSettings,
) -> list[Any]:
    """Mean marker and spread ellipse; the legend handles and labels."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    plan, side = axes
    s, text = cfg.style, cfg.style["legend_format"]
    sample = cfg.traces.hole_sample_style()
    mean = points.mean(axis=0)
    marker = {"marker": s["mean_marker"], "ms": s["mean_marker_size"], "ls": ""}
    marker.update(mfc=s["mean_rgb"], mec="white", zorder=8)
    plan.plot(mean[0], mean[1], gid="mean_landing", **marker)
    side.plot(mean[0], 0.0, clip_on=False, **marker)
    handles: list[Any] = [
        Line2D([0], [0], color=sample.colour, lw=sample.width_pt),
        Line2D([0], [0], color="none", **{k: marker[k] for k in marker if k != "ls"}),
    ]
    labels = [text["traces"].format(n=len(points)), text["mean"]]
    fit = landing_ellipse(
        points, sigma_cfg.ellipse_sigma, sigma_cfg.min_points_for_ellipse
    )
    if fit is not None:
        centre, width, height, angle = fit
        patch = Ellipse(
            (float(centre[0]), float(centre[1])), width, height, angle=angle, zorder=4
        )
        patch.set(gid="spread_ellipse", lw=s["ellipse_linewidth"])
        patch.set_facecolor((*s["ellipse_rgb"], s["ellipse_fill_alpha"]))
        patch.set_edgecolor((*s["ellipse_rgb"], s["ellipse_edge_alpha"]))
        plan.add_patch(patch)
        handles.append(patch)
        labels.append(text["ellipse"].format(sigma=sigma_cfg.ellipse_sigma))
    return [handles, labels]


def _draw_nominal(
    axes: tuple[Any, Any],
    nominal: SampleOutcome,
    hole: Any,
    cfg: DispersionHoleSettings,
    width_pt: float,
    scale: float,
) -> Any:
    """The nominal shot bold on top in the export's hole-view colour; its handle."""
    from matplotlib.lines import Line2D

    from src.tools.native_viewer_export.hole_view import load_hole_view_settings

    colour = load_hole_view_settings().colour_rgba
    (shot,) = _hole_shots([nominal], hole, colour)
    style = HoleLineStyle(colour, width_pt, cfg.traces.hole_nominal_zorder)
    draw_on_hole(axes, [shot.path_m], hole, scale, style, "nominal_shot")
    return Line2D([], [], color=colour, lw=width_pt)


def build_dispersion_hole_figure(
    result: DispersionResult,
    sigma_cfg: DispersionSettings,
    cfg: DispersionHoleSettings,
    units: DisplayUnits | None = None,
    nominal: SampleOutcome | None = None,
) -> tuple[Any, Any, Any]:
    """The figure and its (plan, profile) axes; the caller saves and closes it.

    Preconditions: at least one available shot (``ValueError``). Postcondition:
    ``golf_environment.hole_figure`` draws the hole and every trace; this adds
    the mean landing point, the ``ellipse_sigma`` ellipse of ``sigma_cfg`` and a
    legend with mean +/- sd of carry and side. The samples are thin
    (``hole_sample_width_pt``); ``nominal`` (the base setup's shot, with a
    flight, else ``ValueError``) is drawn on top at the hole figure's width.
    """
    if nominal is not None and not nominal.available:
        raise ValueError("the nominal shot has no flight to draw")
    if nominal is not None and not result.available:
        return _nominal_only_figure(nominal, cfg, units or load_display_units())
    from matplotlib.lines import Line2D

    from src.tools.golf_environment.hole_figure import build_hole_figure
    from src.tools.golf_environment.shots import hole_from_settings, load_shots_settings

    units = units or load_display_units()
    settings = load_shots_settings()
    bold = settings["figure"]["line_width"]  # the nominal keeps the shot width
    thin = {"line_width": cfg.traces.hole_sample_width_pt}
    settings["figure"] = {**settings["figure"], **cfg.style["figure"], **thin}
    hole = hole_from_settings(settings, cfg.preset)
    shots = dispersion_hole_shots(result, hole, cfg)
    fig, plan, side = build_hole_figure(
        shots, hole, settings, units, False, title_note(result, cfg)
    )
    plan.get_legend().remove()
    points = _plan_points(shots, hole, units.distance_scale)
    handles, labels = _overlay_spread((plan, side), points, sigma_cfg, cfg)
    if nominal is not None:
        scale = units.distance_scale
        handles.append(_draw_nominal((plan, side), nominal, hole, cfg, bold, scale))
        labels.append(cfg.style["legend_format"]["nominal"])
    labels.append(stats_line(result, cfg, units))
    handles.append(Line2D([], [], ls=""))
    plan.legend(handles, labels, loc="upper left", framealpha=0.9, ncol=1).set_zorder(
        10
    )
    return fig, plan, side


def _nominal_only_figure(
    nominal: SampleOutcome, cfg: DispersionHoleSettings, units: DisplayUnits
) -> tuple[Any, Any, Any]:
    """The hole still of the nominal shot alone (no sample has a flight).

    Drawn like the export's shot (its colour and width); the title says that
    there is no dispersion, and there is no mean marker or ellipse.
    """
    from src.tools.golf_environment.hole_figure import build_hole_figure
    from src.tools.golf_environment.shots import hole_from_settings, load_shots_settings
    from src.tools.native_viewer_export.hole_view import load_hole_view_settings

    settings = load_shots_settings()
    settings["figure"] = {**settings["figure"], **cfg.style["figure"]}
    hole = hole_from_settings(settings, cfg.preset)
    colour = load_hole_view_settings().colour_rgba
    shots = _hole_shots([nominal], hole, colour)
    note = cfg.style["title_format_nominal_only"]
    return build_hole_figure(shots, hole, settings, units, False, note)


def write_dispersion_hole(
    result: DispersionResult,
    sigma_cfg: DispersionSettings,
    path: Path,
    cfg: DispersionHoleSettings | None = None,
    units: DisplayUnits | None = None,
    nominal: SampleOutcome | None = None,
) -> Path:
    """Write the dispersion-on-the-hole PNG to ``path`` and return it.

    Raises ``ValueError`` when no shot (or a given ``nominal``) has a flight;
    with no sample flight but a ``nominal`` one, the nominal shot alone is drawn.
    Postcondition: ``path`` exists; distances are in ``units`` (default:
    display-units settings); ``nominal`` is emphasised over the samples.
    """
    from matplotlib import pyplot as plt

    fig, _, _ = build_dispersion_hole_figure(
        result, sigma_cfg, cfg or load_dispersion_hole_settings(), units, nominal
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
