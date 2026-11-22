"""Shot dispersion from small random variations of the shot setup (#15150).

Replays the one fitted swing ``n_shots`` times, each with a small random
club roll in the hands and ball position (the two replay-time changes of
:mod:`shot_setup`), and collects per shot the impact factors and the ball
flight, with no rendering.

Reuse and its limits. The seed and trial count are validated by
``perturbation.PerturbationConfig`` and the mean/standard deviation come from
``perturbation.statistics.compute_metric_statistics`` (population std,
``ddof=0``). The perturbation package's samplers perturb polynomial torque
coefficients of a trial adapter, which has no counterpart here (the varied
inputs are two replay-time scalars), so the samples are drawn with
``numpy.random.default_rng(seed)`` Gaussians; ``outcome_dispersion`` is for
wrapped angle error against a target and is not needed for the plain spreads.

The standard deviations in ``shot_dispersion_settings.json`` are first
illustrative values, not measurements.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.perturbation.config import PerturbationConfig
from src.shared.python.perturbation.statistics import compute_metric_statistics
from src.tools.native_viewer_export import shot_setup

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(__file__).with_name("shot_dispersion_settings.json")
METRICS = (
    "carry_m",
    "side_m",
    "face_angle_deg",
    "club_path_deg",
    "launch_deg",
    "spin_rpm",
)
INPUTS = ("club_roll_deg", "ball_shift_m")
SENSITIVITY_METRICS = ("side_m", "carry_m")
_EXPECTED = (ValueError, RuntimeError, KeyError, OSError)


@dataclass(frozen=True)
class DispersionSettings:
    """Values of ``shot_dispersion_settings.json`` (validated on construction)."""

    n_shots: int
    seed: int
    club_roll_sd_deg: float
    ball_position_sd_m: float
    base_club_roll_deg: float = 0.0
    base_ball_shift_m: float = 0.0
    ellipse_sigma: float = 1.0
    min_points_for_ellipse: int = 3
    plot: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        PerturbationConfig(n_trials=self.n_shots, seed=self.seed)
        for name in ("club_roll_sd_deg", "ball_position_sd_m"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and not negative")
        if not (math.isfinite(self.ellipse_sigma) and self.ellipse_sigma > 0.0):
            raise ValueError("ellipse_sigma must be positive and finite")
        if self.min_points_for_ellipse < 3:
            raise ValueError("min_points_for_ellipse must be at least 3")
        shot_setup.validate_shot_setup(self.base_club_roll_deg, self.base_ball_shift_m)


@dataclass(frozen=True)
class ShotSetup:
    """One shot's replay-time setup (club roll in degrees, ball shift in metres)."""

    club_roll_deg: float
    ball_shift_m: float


@dataclass(frozen=True)
class SampleOutcome:
    """Result of one shot: metrics, or the ``reason`` there is no flight.

    ``metrics`` values may be ``None`` (reason in ``notes``). ``trace_target_m``
    is the flight path relative to the ball as (forward, right, up) columns;
    ``path_m`` and ``times_s`` are the world-frame path and its times for a
    later video overlay.
    """

    setup: ShotSetup
    reason: str | None = None
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    trace_target_m: np.ndarray | None = None
    notes: Mapping[str, str] = field(default_factory=dict)
    path_m: np.ndarray | None = None
    times_s: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return self.reason is None and self.trace_target_m is not None

    @property
    def trace(self) -> np.ndarray:
        """The (n, 3) trace; ``ValueError`` for a shot without a flight."""
        if self.trace_target_m is None:
            raise ValueError("shot has no flight trace")
        return self.trace_target_m


Evaluator = Callable[[ShotSetup], SampleOutcome]


def load_dispersion_settings(path: Path | None = None) -> DispersionSettings:
    """Read the settings file; ``KeyError``/``ValueError`` when missing or bad."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    raw.pop("sources", None)
    return DispersionSettings(**raw)


def sample_setups(settings: DispersionSettings) -> list[ShotSetup]:
    """``n_shots`` Gaussian setups around the base, clipped to the shot limits.

    Postcondition: deterministic for a seed; every value passes
    ``shot_setup.validate_shot_setup``. Zero standard deviations give n copies
    of the base.
    """
    limits = shot_setup.load_shot_setup_settings()
    rng = np.random.default_rng(settings.seed)
    draws = rng.standard_normal((settings.n_shots, 2))
    rolls = settings.base_club_roll_deg + settings.club_roll_sd_deg * draws[:, 0]
    shifts = settings.base_ball_shift_m + settings.ball_position_sd_m * draws[:, 1]
    rolls = np.clip(rolls, -limits.max_abs_roll_deg, limits.max_abs_roll_deg)
    bound = limits.max_abs_ball_shift_m
    shifts = np.clip(shifts, -bound, bound)
    return [ShotSetup(float(r), float(s)) for r, s in zip(rolls, shifts, strict=True)]


def _row_values(card: Any) -> dict[str, float | None]:
    if isinstance(card, str) or not getattr(card, "available", False):
        return {}
    return {row.key: row.value for row in card.rows}


def _target_trace(flight: Any) -> np.ndarray:
    """Path relative to the ball in (forward, right, up) columns."""
    rel = np.asarray(flight.path_m) - np.asarray(flight.path_m)[0]
    target, left, up = np.asarray(flight.axes_m)
    return np.column_stack([rel @ target, -(rel @ left), rel @ up])


def _outcome(setup: ShotSetup, flight: Any, card: Any) -> SampleOutcome:
    if flight is None or flight.path_m is None or flight.axes_m is None:
        why = "no flight" if flight is None else str(flight.status)
        return SampleOutcome(setup, reason=why)
    values = _row_values(card)
    metrics = {
        "carry_m": flight.carry_m,
        "side_m": flight.lateral_m,
        "face_angle_deg": values.get("face_angle_deg"),
        "club_path_deg": values.get("club_path_deg"),
        "launch_deg": flight.launch_deg,
        "spin_rpm": flight.spin_rpm,
    }
    notes = {}
    if not values:
        notes["impact_card"] = card if isinstance(card, str) else "unavailable"
    trace = _target_trace(flight)
    return SampleOutcome(
        setup, None, metrics, trace, notes, flight.path_m, flight.times_s
    )


def make_evaluator(swing: Any, export_settings: Any) -> Evaluator:
    """Evaluator replaying ``swing`` per setup with no rendering.

    Order as in ``core.export_swing``: club roll, ball, contact frame, impact
    card, flight. ``export_settings.ball_flight`` defaults to the packaged
    tracer settings. Failures become an outcome with a reason, not a raise.
    """
    from src.tools.native_viewer_export import contact_frame, core, flight_tracer
    from src.tools.native_viewer_export.ball import attach_ball
    from src.tools.native_viewer_export.impact_card import impact_card_for

    base = export_settings
    if base.ball_flight is None:
        base = replace(base, ball_flight=flight_tracer.load_tracer_settings())

    def evaluate(setup: ShotSetup) -> SampleOutcome:
        cfg = replace(
            base,
            club_roll_in_hands_deg=setup.club_roll_deg,
            ball_position_shift_m=setup.ball_shift_m,
        )
        try:
            rolled = shot_setup.apply_club_roll(swing, setup.club_roll_deg)
            placed, _ = attach_ball(rolled, cfg)
            if cfg.impact_time_s is not None or placed.has_impact_provenance:
                placed, contact = contact_frame.refine(
                    placed, core.impact_time_s(placed, cfg)
                )
                if contact is not None:
                    cfg = replace(cfg, impact_time_s=contact)
            card = impact_card_for(placed, core.impact_time_s(placed, cfg), placed.ball)
            flight = flight_tracer.flight_for_export(placed, cfg)
        except _EXPECTED as exc:
            logger.warning("shot %s unavailable: %s", setup, exc)
            return SampleOutcome(setup, reason=f"unavailable: {exc}")
        return _outcome(setup, flight, card)

    return evaluate


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    stat = compute_metric_statistics(np.asarray(values, dtype=float))
    return {"mean": float(stat.mean), "std": float(stat.std), "n": len(values)}


def slopes(
    columns: Mapping[str, Sequence[float]], y: Sequence[float]
) -> dict[str, float | None]:
    """Joint least-squares slopes of ``y`` on every varying input column.

    Fitting the inputs together keeps a chance correlation between them from
    leaking into each other's slope. A constant column (or fewer points than
    unknowns) gives ``None``. Raises ``ValueError`` on unequal lengths.
    """
    ys = np.asarray(y, dtype=float)
    cols = {k: np.asarray(v, dtype=float) for k, v in columns.items()}
    if any(len(v) != len(ys) for v in cols.values()):
        raise ValueError("every column must have as many values as y")
    used = [k for k, v in cols.items() if len(v) > 1 and float(np.ptp(v)) > 0.0]
    out: dict[str, float | None] = dict.fromkeys(cols)
    if not used or len(ys) <= len(used):
        return out
    design = np.column_stack([*(cols[k] for k in used), np.ones_like(ys)])
    fit = np.linalg.lstsq(design, ys, rcond=None)[0]
    out.update({k: float(c) for k, c in zip(used, fit[:-1], strict=True)})
    return out


@dataclass(frozen=True)
class DispersionResult:
    """All shots of one dispersion run."""

    outcomes: tuple[SampleOutcome, ...]

    @property
    def available(self) -> list[SampleOutcome]:
        return [o for o in self.outcomes if o.available]

    def _column(self, metric: str, rows: Sequence[SampleOutcome]) -> list[float]:
        found = (o.metrics.get(metric) for o in rows)
        return [float(v) for v in found if v is not None]

    def summary(self) -> dict[str, dict[str, float] | None]:
        """Mean, population std and count of each metric over available shots."""
        rows = self.available
        return {m: _stats(self._column(m, rows)) for m in METRICS}

    def sensitivity(self) -> dict[str, dict[str, float | None]]:
        """Least-squares slope of side and carry against each varied input."""
        out: dict[str, dict[str, float | None]] = {}
        for metric in SENSITIVITY_METRICS:
            rows = [o for o in self.available if o.metrics.get(metric) is not None]
            ys = self._column(metric, rows)
            inputs = {n: [getattr(o.setup, n) for o in rows] for n in INPUTS}
            out[metric] = slopes(inputs, ys)
        return out

    def landing_points(self) -> np.ndarray:
        """(n, 2) landing points (right, forward) of the available shots."""
        pts = [(o.trace[-1, 1], o.trace[-1, 0]) for o in self.available]
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def to_dict(self) -> dict[str, Any]:
        """JSON record: shots (setup, metrics, reason), summary, sensitivity."""
        shots = []
        for o in self.outcomes:
            landing = None
            if o.available:
                landing = [
                    float(o.trace[-1, 1]),
                    float(o.trace[-1, 0]),
                ]
            shots.append(
                {
                    "club_roll_deg": o.setup.club_roll_deg,
                    "ball_shift_m": o.setup.ball_shift_m,
                    "available": o.available,
                    "reason": o.reason,
                    "metrics": dict(o.metrics),
                    "notes": dict(o.notes),
                    "landing_right_forward_m": landing,
                }
            )
        return {
            "n_shots": len(self.outcomes),
            "n_available": len(self.available),
            "side_positive": "right of the target line seen from behind the ball",
            "shots": shots,
            "summary": self.summary(),
            "sensitivity_per_unit": self.sensitivity(),
        }


def run_dispersion(
    swing: Any,
    export_settings: Any,
    setups: Sequence[ShotSetup],
    evaluate: Evaluator | None = None,
) -> DispersionResult:
    """Evaluate every setup; an unavailable shot is kept with its reason.

    ``evaluate`` defaults to :func:`make_evaluator` (replay, no rendering);
    tests inject a synthetic one. Raises ``ValueError`` for no setups.
    """
    if not setups:
        raise ValueError("setups must not be empty")
    run = evaluate or make_evaluator(swing, export_settings)
    return DispersionResult(tuple(run(s) for s in setups))


def landing_ellipse(
    points: np.ndarray, sigma: float, minimum: int = 3
) -> tuple[np.ndarray, float, float, float] | None:
    """Covariance ellipse of (n, 2) points: centre, full width, full height and
    angle (degrees, counter-clockwise from the first axis) at ``sigma`` standard
    deviations; ``None`` for fewer than ``minimum`` points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(pts) < minimum:
        return None
    values, vectors = np.linalg.eigh(np.cov(pts.T))
    order = np.argsort(values)[::-1]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    widths = 2.0 * sigma * np.sqrt(values)
    return pts.mean(axis=0), float(widths[0]), float(widths[1]), angle
