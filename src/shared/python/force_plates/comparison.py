"""Model-versus-measured vertical ground-reaction comparison (OF-08, #13086).

Diagnostics, not an acceptance test: OF-12 decides whether a candidate can be
ruled out; this module reports how large the disagreement is and where.  Only
observed components are compared (vertical load and the sensing-plane CoP);
shear and free moment are never compared because the plates do not see them.

At each model frame the plates are read with OF-05 :func:`reading_at`, and
each residual is expressed in units of the reading's band-equivalent sigma
``sigma = random + bias / k``, so ``|z| <= k`` is exactly "inside the k-band".

* **Load:** ``r = Fz_model - N_measured``, ``z = r / sigma_load``.
* **CoP:** vertical cells balance moments about their sensing plane, a depth
  ``h`` below the foot, so the plate reports ``c' = c - h F_t / F_z``.  The
  model CoP ``c`` is projected the same way before the comparison, using the
  model's own shear; without that the residual would contain the lever-arm
  term.  Per axis, ``sigma = hypot(sigma_cop, |F_axis / F_z| sigma_h)``.
* **Lead fraction:** ``f = N_lead / (N_lead + N_trail)``; its first upward
  crossing of 0.5 is located by linear interpolation for model and
  measurement.  The measured crossing time has ``sigma_t = sigma_f / |df/dt|``
  (``sigma_f`` propagated to first order), combined in quadrature with the
  alignment's common-mode ``sigma_offset_s``.  A delta within ``k sigma`` is
  *unresolved* and reported without a value.

Frames whose time the trial does not cover are skipped and counted, never
extrapolated.  Every result carries the trial's qualification (OF-HW #13082),
the alignment kind and the world frame id.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np

from src.shared.python.biomechanics.ground_reaction import GroundReactionSeries
from src.shared.python.force_plates.alignment import PlateTimeAlignment
from src.shared.python.force_plates.contracts import (
    FOOT_LABELS,
    Array,
    non_empty_text,
)
from src.shared.python.force_plates.readings import (
    FootSample,
    PlacedTrial,
    PlateReading,
    ReadingOptions,
    covers,
    reading_at,
)

__all__ = [
    "ChannelResidual",
    "ComparisonOptions",
    "CrossingTiming",
    "VerticalComparison",
    "compare_vertical",
]

Purpose = Literal["diagnostic", "validation"]
#: Lead fraction whose crossing marks the weight shift onto the lead foot.
LEAD_CROSSING_FRACTION = 0.5
#: Largest model-ground / plate-surface height disagreement accepted [m].
GROUND_HEIGHT_TOL_M = 1e-3
_HORIZONTAL = slice(0, 2)
_VERTICAL = 2


@dataclass(frozen=True)
class ComparisonOptions:
    """How to compare: reading options, foot mapping, frame and purpose.

    ``foot_map`` maps each plate foot (``left``/``right``) to the model's foot
    key.  ``frame_id`` names the world both series are in (from the OF-03
    :class:`TargetedPlacement`).  ``purpose="validation"`` refuses an alignment
    that was cross-correlated against ``model_identity`` itself.
    """

    reading: ReadingOptions
    foot_map: Mapping[str, str]
    frame_id: str
    model_identity: str
    purpose: Purpose = "diagnostic"

    def __post_init__(self) -> None:
        if not isinstance(self.reading, ReadingOptions):
            raise TypeError("reading must be a ReadingOptions")
        mapping = dict(self.foot_map)
        if set(mapping) != set(FOOT_LABELS) or not all(
            isinstance(v, str) and v for v in mapping.values()
        ):
            raise ValueError(f"foot_map must map exactly {FOOT_LABELS} to model keys")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(
                "foot_map must map each plate foot to a distinct model key"
            )
        non_empty_text(self.frame_id, "frame_id")
        non_empty_text(self.model_identity, "model_identity")
        if self.purpose not in get_args(Purpose):
            raise ValueError(f"purpose must be one of {get_args(Purpose)}")
        object.__setattr__(self, "foot_map", MappingProxyType(mapping))


@dataclass(frozen=True, eq=False)
class ChannelResidual:
    """Residuals of one foot and quantity over the compared frames.

    ``residual``, ``sigma`` and ``z`` are ``(n,)`` for load and ``(n, 2)`` for
    CoP (world x, y).  Empty arrays when nothing could be compared.
    """

    times_s: Array
    residual: Array
    sigma: Array
    z: Array

    @property
    def n_frames(self) -> int:
        """Number of compared frames."""
        return int(self.times_s.shape[0])

    @property
    def rms_z(self) -> float | None:
        """Root-mean-square ``z`` over every compared value, or ``None``."""
        return None if self.z.size == 0 else float(np.sqrt(np.mean(self.z**2)))

    @property
    def max_abs_z(self) -> float | None:
        """Largest ``|z|``, or ``None`` when nothing was compared."""
        return None if self.z.size == 0 else float(np.max(np.abs(self.z)))

    @property
    def time_of_max_s(self) -> float | None:
        """Model time of the largest ``|z|``, or ``None``."""
        if self.z.size == 0:
            return None
        per_frame = np.abs(self.z).reshape(self.n_frames, -1).max(axis=1)
        return float(self.times_s[int(np.argmax(per_frame))])

    def summary(self) -> dict[str, Any]:
        """JSON-compatible summary."""
        return {
            "n_frames": self.n_frames,
            "rms_z": self.rms_z,
            "max_abs_z": self.max_abs_z,
            "time_of_max_s": self.time_of_max_s,
        }


@dataclass(frozen=True)
class CrossingTiming:
    """First upward crossing of the lead fraction through 0.5.

    ``delta_s = model_s - measured_s`` is reported only when ``resolved``
    (``|delta| > k * sigma_s``); otherwise it is ``None``, never a number.
    """

    model_s: float | None
    measured_s: float | None
    sigma_s: float | None
    resolved: bool
    delta_s: float | None


@dataclass(frozen=True, eq=False)
class VerticalComparison:
    """Comparison of one model run with one placed plate trial."""

    load: Mapping[str, ChannelResidual]
    cop: Mapping[str, ChannelResidual]
    lead_crossing: CrossingTiming
    n_compared: int
    n_skipped: int
    qualified: bool
    alignment_kind: str
    frame_id: str
    purpose: str
    coverage: float = field(default=0.0)

    @property
    def label(self) -> str:
        """``"qualified measurement"`` or ``"unqualified measurement"``."""
        return "qualified measurement" if self.qualified else "unqualified measurement"

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible summary (residual arrays omitted)."""
        return {
            "label": self.label,
            "qualified": self.qualified,
            "alignment_kind": self.alignment_kind,
            "frame_id": self.frame_id,
            "purpose": self.purpose,
            "coverage": self.coverage,
            "n_compared": self.n_compared,
            "n_skipped": self.n_skipped,
            "load": {k: v.summary() for k, v in self.load.items()},
            "cop": {k: v.summary() for k, v in self.cop.items()},
            "lead_crossing": asdict(self.lead_crossing),
        }


# ------------------------------------------------------------ checks --
def _check_inputs(
    model: object, placed: object, alignment: object, options: object
) -> None:
    if not isinstance(model, GroundReactionSeries):
        raise TypeError("model must be a GroundReactionSeries")
    if not isinstance(placed, PlacedTrial):
        raise TypeError("placed must be a PlacedTrial")
    if not isinstance(alignment, PlateTimeAlignment):
        raise TypeError("alignment must be a PlateTimeAlignment")
    if not isinstance(options, ComparisonOptions):
        raise TypeError("options must be a ComparisonOptions")


def _check_model(
    model: GroundReactionSeries, placed: PlacedTrial, options: ComparisonOptions
) -> None:
    missing = [k for k in options.foot_map.values() if k not in model.force_n]
    if missing:
        raise ValueError(f"model has no foot keys {missing} named by foot_map")
    ground = np.asarray(model.ground_height_m, dtype=float)
    surface = placed.placement.surface_height_m
    if ground.size and np.any(np.abs(ground - surface) > GROUND_HEIGHT_TOL_M):
        raise ValueError(
            f"model ground height differs from the plate surface ({surface} m) by "
            f"more than {GROUND_HEIGHT_TOL_M} m"
        )


def _check_purpose(alignment: PlateTimeAlignment, options: ComparisonOptions) -> None:
    if options.purpose == "validation" and not alignment.validates(
        options.model_identity
    ):
        raise ValueError(
            f"the alignment was derived from {options.model_identity!r}; a "
            "validation-labelled comparison against that model is refused"
        )


# --------------------------------------------------------- residuals --
def _stack(
    times: list[float], rows: list[tuple[Any, Any]], width: int
) -> ChannelResidual:
    shape = (0,) if width == 1 else (0, width)
    if not rows:
        empty = np.zeros(shape)
        return ChannelResidual(np.zeros(0), empty, empty, empty)
    residual = np.array([r for r, _ in rows], dtype=float)
    sigma = np.array([s for _, s in rows], dtype=float)
    return ChannelResidual(np.array(times), residual, sigma, residual / sigma)


def _load_residual(
    model: GroundReactionSeries,
    key: str,
    frames: list[int],
    samples: Mapping[int, FootSample],
) -> tuple[list[float], list[tuple[float, float]]]:
    times, rows = [], []
    for i in frames:
        fz = float(model.force_n[key][i, _VERTICAL])
        sample = samples[i]
        if math.isfinite(fz):
            times.append(float(model.times_s[i]))
            rows.append((fz - sample.load_n, sample.sigma_load_n))
    return times, rows


def _projected_cop(force: Array, cop: Array, depth_m: float) -> Array | None:
    """Model CoP moved onto the sensing plane: ``c - h F_t / F_z`` (x, y)."""
    fz = float(force[_VERTICAL])
    if not (fz > 0.0 and np.isfinite(force).all() and np.isfinite(cop).all()):
        return None
    return cop[_HORIZONTAL] - depth_m * force[_HORIZONTAL] / fz


@dataclass(frozen=True)
class _FootFrames:
    """One foot's plate samples by model frame, and the plate sensing depth."""

    samples: Mapping[int, FootSample]
    depth_m: float


def _cop_residual(
    model: GroundReactionSeries, key: str, frames: list[int], foot: _FootFrames
) -> tuple[list[float], list[tuple[Array, Array]]]:
    times, rows = [], []
    for i in frames:
        sample, force = foot.samples[i], model.force_n[key][i]
        projected = _projected_cop(force, model.cop_m[key][i], foot.depth_m)
        if sample.cop_m is None or projected is None or not model.in_contact[key][i]:
            continue
        lever = np.abs(force[_HORIZONTAL] / force[_VERTICAL])
        sigma = np.hypot(sample.sigma_cop_m, lever * sample.sigma_sensing_m)
        times.append(float(model.times_s[i]))
        rows.append((projected - sample.cop_m[_HORIZONTAL], sigma))
    return times, rows


# ---------------------------------------------------- lead crossing --
def _first_upward_crossing(times: Array, fraction: Array) -> int | None:
    """Index ``i`` with ``f[i] < 0.5 <= f[i + 1]`` (first such), or ``None``."""
    below = fraction[:-1] < LEAD_CROSSING_FRACTION
    above = fraction[1:] >= LEAD_CROSSING_FRACTION
    hits = np.flatnonzero(below & above & np.isfinite(fraction[:-1]))
    return int(hits[0]) if hits.size else None


def _crossing_time(times: Array, fraction: Array) -> tuple[float, float, int] | None:
    """Interpolated crossing time, the local slope ``df/dt`` and its interval."""
    i = _first_upward_crossing(times, fraction)
    if i is None:
        return None
    dt = float(times[i + 1] - times[i])
    slope = float(fraction[i + 1] - fraction[i]) / dt
    t = float(times[i]) + (LEAD_CROSSING_FRACTION - float(fraction[i])) / slope
    return t, slope, i


def _fraction_and_sigma(
    loads: tuple[Array, Array], sigmas: tuple[Array, Array], floor_n: float
) -> tuple[Array, Array]:
    """``f = lead / total`` and its first-order sigma; NaN below ``floor_n``.

    ``d f / d lead = trail / total^2`` and ``d f / d trail = -lead / total^2``.
    """
    (lead, trail), (s_lead, s_trail) = loads, sigmas
    total = lead + trail
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(total < floor_n, np.nan, lead / total)
        sigma = np.hypot(trail * s_lead, lead * s_trail) / total**2
    return fraction, sigma


def _lead_crossing(context: dict, alignment: PlateTimeAlignment) -> CrossingTiming:
    times, floor = context["times"], context["floor_n"]
    zeros = (np.zeros_like(times), np.zeros_like(times))
    model_f, _ = _fraction_and_sigma(context["model_loads"], zeros, floor)
    meas_f, meas_sigma = _fraction_and_sigma(
        context["meas_loads"], context["meas_sigmas"], floor
    )
    model_cross = _crossing_time(times, model_f) if times.size > 1 else None
    meas_cross = _crossing_time(times, meas_f) if times.size > 1 else None
    if model_cross is None or meas_cross is None:
        return CrossingTiming(
            model_cross[0] if model_cross else None,
            meas_cross[0] if meas_cross else None,
            None,
            False,
            None,
        )
    t_meas, slope, i = meas_cross
    weight = (t_meas - float(times[i])) / float(times[i + 1] - times[i])
    sigma_f = (1.0 - weight) * meas_sigma[i] + weight * meas_sigma[i + 1]
    sigma = math.hypot(sigma_f / abs(slope), alignment.sigma_offset_s)
    delta = model_cross[0] - t_meas
    resolved = abs(delta) > context["coverage"] * sigma
    return CrossingTiming(
        model_cross[0], t_meas, sigma, resolved, delta if resolved else None
    )


# ------------------------------------------------------------- entry --
def _read_frames(
    model: GroundReactionSeries,
    placed: PlacedTrial,
    alignment: PlateTimeAlignment,
    reading: ReadingOptions,
) -> dict[int, PlateReading]:
    readings = {}
    for i, t in enumerate(model.times_s):
        if covers(placed, float(t), alignment, reading):
            readings[i] = reading_at(placed, float(t), alignment, reading)
    return readings


def _lead_trail_context(
    model: GroundReactionSeries,
    placed: PlacedTrial,
    readings: dict[int, PlateReading],
    options: ComparisonOptions,
) -> dict:
    frames = sorted(readings)
    trial = placed.trial
    feet = (trial.lead_foot, trial.trail_foot)
    model_keys = [options.foot_map[f] for f in feet]
    samples = [[readings[i].feet[f] for i in frames] for f in feet]
    lead, trail = samples
    return {
        "times": np.array([float(model.times_s[i]) for i in frames]),
        "model_loads": tuple(
            np.array([float(model.force_n[k][i, _VERTICAL]) for i in frames])
            for k in model_keys
        ),
        "meas_loads": (
            np.array([s.load_n for s in lead]),
            np.array([s.load_n for s in trail]),
        ),
        "meas_sigmas": (
            np.array([s.sigma_load_n for s in lead]),
            np.array([s.sigma_load_n for s in trail]),
        ),
        "floor_n": trial.cop_min_load_n,
        "coverage": options.reading.coverage,
    }


def compare_vertical(
    model: GroundReactionSeries,
    placed: PlacedTrial,
    alignment: PlateTimeAlignment,
    options: ComparisonOptions,
) -> VerticalComparison:
    """Compare ``model`` with the placed plate trial on observed components only.

    Preconditions: argument types (``TypeError``); ``options.foot_map`` names
    model foot keys that exist; the model ground height (when recorded) equals
    the plate surface within :data:`GROUND_HEIGHT_TOL_M`; a validation-purpose
    comparison does not use an alignment derived from that model
    (``ValueError``).  Postconditions: every compared value has a finite,
    positive sigma; frames outside the trial are counted in ``n_skipped``.
    """
    _check_inputs(model, placed, alignment, options)
    _check_model(model, placed, options)
    _check_purpose(alignment, options)
    readings = _read_frames(model, placed, alignment, options.reading)
    frames = sorted(readings)
    trial = placed.trial
    load, cop = {}, {}
    for foot, key in options.foot_map.items():
        per_foot = _FootFrames(
            {i: readings[i].feet[foot] for i in frames}, trial.sensing_depth_m
        )
        load[foot] = _stack(*_load_residual(model, key, frames, per_foot.samples), 1)
        cop[foot] = _stack(*_cop_residual(model, key, frames, per_foot), 2)
    lead_trail = _lead_trail_context(model, placed, readings, options)
    return VerticalComparison(
        load=MappingProxyType(load),
        cop=MappingProxyType(cop),
        lead_crossing=_lead_crossing(lead_trail, alignment),
        n_compared=len(frames),
        n_skipped=int(model.times_s.shape[0]) - len(frames),
        qualified=trial.provenance.qualified,
        alignment_kind=alignment.kind,
        frame_id=options.frame_id,
        purpose=options.purpose,
        coverage=options.reading.coverage,
    )
