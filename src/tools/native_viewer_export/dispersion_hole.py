"""Dispersion on the hole: every shot of a dispersion run on one hole (#15150).

All sample traces with their carry and rest points on the hole plan and side
profile, the mean landing point, a spread ellipse and the mean +/- standard
deviation of carry and side. The hole and the shot drawing are
``golf_environment`` (``shots_on_hole`` and ``hole_figure``); this module feeds
them the dispersion shots and adds the spread overlays. The hole is aimed along
the address line: a dispersion has no single finish line. Options and every
number: ``dispersion_hole_settings.json`` (see ``sources``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.native_viewer_export.display_units import (
    DisplayUnits,
    load_display_units,
)
from src.tools.native_viewer_export.shot_dispersion import (
    INPUTS,
    DispersionResult,
    DispersionSettings,
    landing_ellipse,
)

SETTINGS_PATH = Path(__file__).with_name("dispersion_hole_settings.json")
_ALPHAS = ("trace_alpha", "ellipse_fill_alpha", "ellipse_edge_alpha")
_RGBS = ("trace_rgb", "mean_rgb", "ellipse_rgb")
_POSITIVE = ("mean_marker_size", "ellipse_linewidth")


@dataclass(frozen=True)
class DispersionHoleSettings:
    """Validated options of the dispersion hole still (``style`` is the JSON)."""

    preset: str
    style: Mapping[str, Any]

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
    from src.tools.golf_environment.shots import TargetFlight, shots_on_hole

    shots = result.available
    if not shots:
        raise ValueError("no shot has a flight to draw")
    style = cfg.style
    colour = (*style["trace_rgb"], style["trace_alpha"])
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
    summary = result.summary()
    carry, side = summary["carry_m"], summary["side_m"]
    if carry is None or side is None:
        raise ValueError("no shot has a carry and a side")
    k = units.distance_scale
    return cfg.style["legend_format"]["stats"].format(
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
    mean = points.mean(axis=0)
    marker = {"marker": s["mean_marker"], "ms": s["mean_marker_size"], "ls": ""}
    marker.update(mfc=s["mean_rgb"], mec="white", zorder=8)
    plan.plot(mean[0], mean[1], gid="mean_landing", **marker)
    side.plot(mean[0], 0.0, clip_on=False, **marker)
    handles: list[Any] = [
        Line2D([0], [0], color=(*s["trace_rgb"], s["trace_alpha"]), lw=2.4),
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


def build_dispersion_hole_figure(
    result: DispersionResult,
    sigma_cfg: DispersionSettings,
    cfg: DispersionHoleSettings,
    units: DisplayUnits | None = None,
) -> tuple[Any, Any, Any]:
    """The figure and its (plan, profile) axes; the caller saves and closes it.

    Preconditions: at least one available shot (``ValueError``). Postcondition:
    ``golf_environment.hole_figure`` draws the hole and every trace; this adds
    the mean landing point, the ``ellipse_sigma`` ellipse of ``sigma_cfg`` and a
    legend with mean +/- sd of carry and side.
    """
    from matplotlib.lines import Line2D

    from src.tools.golf_environment.hole_figure import build_hole_figure
    from src.tools.golf_environment.shots import hole_from_settings, load_shots_settings

    units = units or load_display_units()
    settings = load_shots_settings()
    settings["figure"] = {**settings["figure"], **cfg.style["figure"]}
    hole = hole_from_settings(settings, cfg.preset)
    shots = dispersion_hole_shots(result, hole, cfg)
    fig, plan, side = build_hole_figure(
        shots, hole, settings, units, False, title_note(result, cfg)
    )
    plan.get_legend().remove()
    points = _plan_points(shots, hole, units.distance_scale)
    handles, labels = _overlay_spread((plan, side), points, sigma_cfg, cfg)
    labels.append(stats_line(result, cfg, units))
    handles.append(Line2D([], [], ls=""))
    plan.legend(handles, labels, loc="upper left", framealpha=0.9, ncol=1).set_zorder(
        10
    )
    return fig, plan, side


def write_dispersion_hole(
    result: DispersionResult,
    sigma_cfg: DispersionSettings,
    path: Path,
    cfg: DispersionHoleSettings | None = None,
    units: DisplayUnits | None = None,
) -> Path:
    """Write the dispersion-on-the-hole PNG to ``path`` and return it.

    Raises ``ValueError`` when no shot has a flight. Postcondition: ``path``
    exists; distances are in ``units`` (default: display-units settings).
    """
    from matplotlib import pyplot as plt

    fig, _, _ = build_dispersion_hole_figure(
        result, sigma_cfg, cfg or load_dispersion_hole_settings(), units
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
