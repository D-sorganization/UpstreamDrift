"""Plate readings at an arbitrary model time with an uncertainty budget (OF-05).

:func:`reading_at` evaluates a placed vertical plate trial at a model time and
reports, per foot, the load and the centre of pressure (CoP) together with an
effective 1-sigma that covers four first-order error sources:

* the declared measurement uncertainty of the plate;
* a within-sample **timing skew** between channels, ``skew_s * |dy/dt|``;
* the **linear-interpolation bias**, bounded pointwise by
  ``|(t - t_i)(t_{i+1} - t)| / 2 * M2`` where ``M2`` estimates ``max |y''|``
  from the neighbouring second divided differences (``2 * |f[a, b, c]|``; the
  bound is zero at a sample and at most ``h**2 / 8 * M2``);
* the common-mode clock offset, which is **not** folded in here: it is shared
  by every frame, so it is profiled with ``ReadingOptions.offset_shift_s``.

The deterministic bias is converted to an equivalent sigma with the band factor
``k = coverage`` so that the band ``k * sigma_eff`` equals
``k * hypot(sigma_meas, sigma_skew) + bias`` exactly.  These are first-order
bounds, not a qualification of the device or of the interpolation model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from src.shared.python.force_plates.alignment import PlateTimeAlignment
from src.shared.python.force_plates.contracts import (
    FOOT_LABELS,
    Array,
    VerticalPlateTrial,
    finite_non_negative,
    finite_positive,
    real_number,
)
from src.shared.python.force_plates.placement import PlatePlacement

__all__ = [
    "FootSample",
    "PlacedTrial",
    "PlateReading",
    "ReadingOptions",
    "reading_at",
]

#: Multiplier turning a second divided difference into a bound on ``|y''|``.
_CURVATURE_FACTOR = 2.0
_WORLD_DIM = 3


def _finite(value: object, name: str) -> float:
    number = real_number(value, name)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class ReadingOptions:
    """How a reading's uncertainty budget is formed.

    ``coverage`` is the band factor ``k > 0`` that converts the deterministic
    interpolation bias into an equivalent sigma (``bias / k``).  ``skew_s >= 0``
    is the declared timing skew between channels within one sample.
    ``offset_shift_s`` shifts the common-mode clock offset (profiling).

    Raises ``TypeError`` for non-numbers, ``ValueError`` for out-of-range values.
    """

    coverage: float
    skew_s: float = 0.0
    offset_shift_s: float = 0.0

    def __post_init__(self) -> None:
        finite_positive(self.coverage, "coverage")
        finite_non_negative(self.skew_s, "skew_s")
        _finite(self.offset_shift_s, "offset_shift_s")


@dataclass(frozen=True, eq=False)
class FootSample:
    """One foot's reading: load, CoP on the sensing plane, and 1-sigmas.

    ``cop_m`` is the world point ``(3,)`` on the plate's force-sensing plane
    (read-only), or ``None`` when the foot is unloaded at either bracketing
    sample.  ``sigma_load_n`` and ``sigma_cop_m`` are effective sigmas as
    defined in the module docstring; ``sigma_sensing_m`` is the uncertainty of
    the sensing-plane depth.
    """

    load_n: float
    sigma_load_n: float
    cop_m: Array | None
    sigma_cop_m: float
    sigma_sensing_m: float

    def __post_init__(self) -> None:
        _finite(self.load_n, "load_n")
        finite_non_negative(self.sigma_load_n, "sigma_load_n")
        finite_non_negative(self.sigma_cop_m, "sigma_cop_m")
        finite_non_negative(self.sigma_sensing_m, "sigma_sensing_m")
        if self.cop_m is None:
            return
        cop = np.array(self.cop_m, dtype=np.float64, copy=True)
        if cop.shape != (_WORLD_DIM,) or not np.isfinite(cop).all():
            raise ValueError(f"cop_m must be a finite 3-vector, got shape {cop.shape}")
        cop.flags.writeable = False
        object.__setattr__(self, "cop_m", cop)


@dataclass(frozen=True)
class PlateReading:
    """Both feet at one model time; ``feet`` is keyed by ``FOOT_LABELS``."""

    model_time_s: float
    trial_time_s: float
    feet: Mapping[str, FootSample]

    def __post_init__(self) -> None:
        _finite(self.model_time_s, "model_time_s")
        _finite(self.trial_time_s, "trial_time_s")
        if set(self.feet) != set(FOOT_LABELS):
            raise ValueError(f"feet must be keyed exactly by {FOOT_LABELS}")
        for label, sample in self.feet.items():
            if not isinstance(sample, FootSample):
                raise TypeError(f"feet[{label!r}] must be a FootSample")
        object.__setattr__(self, "feet", MappingProxyType(dict(self.feet)))


@dataclass(frozen=True)
class PlacedTrial:
    """A plate trial together with where its stance frame sits in the world."""

    trial: VerticalPlateTrial
    placement: PlatePlacement

    def __post_init__(self) -> None:
        if not isinstance(self.trial, VerticalPlateTrial):
            raise TypeError("trial must be a VerticalPlateTrial")
        if not isinstance(self.placement, PlatePlacement):
            raise TypeError("placement must be a PlatePlacement")

    def span_s(self) -> tuple[float, float]:
        """First and last sample time on the trial's own clock [s]."""
        times = self.trial.times_s
        return float(times[0]), float(times[-1])


# ----------------------------------------------------------- numerics --
@dataclass(frozen=True)
class _Estimate:
    """Interpolated value, interpolation-bias bound and first difference."""

    value: float
    bias: float
    rate: float


def _second_difference(times: Array, y: Array, start: int) -> float | None:
    """Second divided difference over samples ``start..start+2``, if finite."""
    if start < 0 or start + 2 >= times.shape[0]:
        return None
    t0, t1, t2 = times[start : start + 3]
    y0, y1, y2 = y[start : start + 3]
    value = ((y2 - y1) / (t2 - t1) - (y1 - y0) / (t1 - t0)) / (t2 - t0)
    return float(value) if math.isfinite(value) else None


def _curvature_bound(times: Array, y: Array, index: int) -> float:
    """Estimate of ``max |y''|`` near interval ``index``; 0 with only two samples."""
    found = [
        abs(dd)
        for dd in (
            _second_difference(times, y, index - 1),
            _second_difference(times, y, index),
        )
        if dd is not None
    ]
    return _CURVATURE_FACTOR * max(found, default=0.0)


def _estimate(times: Array, y: Array, index: int, t: float) -> _Estimate:
    """Linear interpolation of ``y`` in interval ``index`` with its bias bound."""
    t0, t1 = float(times[index]), float(times[index + 1])
    y0, y1 = float(y[index]), float(y[index + 1])
    slope = (y1 - y0) / (t1 - t0)
    bias = abs((t - t0) * (t1 - t)) / 2.0 * _curvature_bound(times, y, index)
    return _Estimate(value=y0 + slope * (t - t0), bias=bias, rate=abs(slope))


def _effective_sigma(
    base: float, bias: float, rate: float, options: ReadingOptions
) -> float:
    """``hypot(base, skew * rate) + bias / coverage`` (see the module docstring)."""
    return math.hypot(base, options.skew_s * rate) + bias / options.coverage


def _locate(times: Array, t: float) -> int:
    """Index ``i`` with ``times[i] <= t <= times[i + 1]``; refuses extrapolation."""
    if not times[0] <= t <= times[-1]:
        raise ValueError(
            f"trial time {t} s is outside the trial span "
            f"[{times[0]}, {times[-1]}] s; extrapolation is refused"
        )
    return min(int(np.searchsorted(times, t, side="right")) - 1, times.shape[0] - 2)


# ------------------------------------------------------------ readings --
@dataclass(frozen=True, eq=False)
class _CopEstimate:
    """Interpolated stance CoP ``(2,)`` with the worst-axis bias and rate."""

    xy: Array
    bias: float
    rate: float


def _cop_estimate(
    times: Array, cop: Array, index: int, t: float
) -> _CopEstimate | None:
    """Per-axis CoP estimates merged (max bias and rate); ``None`` if unloaded."""
    if np.isnan(cop[index : index + 2]).any():
        return None
    axes = [_estimate(times, cop[:, axis], index, t) for axis in range(2)]
    return _CopEstimate(
        xy=np.array([axis.value for axis in axes]),
        bias=max(axis.bias for axis in axes),
        rate=max(axis.rate for axis in axes),
    )


def _world_cop(placed: PlacedTrial, stance_xy: Array) -> Array:
    """World point on the sensing plane below the plate surface."""
    point = placed.placement.to_world(stance_xy[None, :])[0].copy()
    point[2] -= placed.trial.sensing_depth_m
    return point


def _foot_sample(
    placed: PlacedTrial, foot: str, bracket: tuple[int, float], options: ReadingOptions
) -> FootSample:
    """Reading of ``foot`` in interval ``bracket[0]`` at trial time ``bracket[1]``."""
    trial = placed.trial
    index, t = bracket
    load = _estimate(trial.times_s, trial.load_n[foot], index, t)
    base_load = float(trial.uncertainty.force_sigma_n(load.value))
    cop = _cop_estimate(trial.times_s, trial.cop_m[foot], index, t)
    base_cop = trial.uncertainty.cop_m
    return FootSample(
        load_n=load.value,
        sigma_load_n=_effective_sigma(base_load, load.bias, load.rate, options),
        cop_m=None if cop is None else _world_cop(placed, cop.xy),
        sigma_cop_m=(
            base_cop
            if cop is None
            else _effective_sigma(base_cop, cop.bias, cop.rate, options)
        ),
        sigma_sensing_m=trial.uncertainty.sensing_depth_sigma_m,
    )


def _check_reading_inputs(placed: object, alignment: object, options: object) -> None:
    for name, value, kind in (
        ("placed", placed, PlacedTrial),
        ("alignment", alignment, PlateTimeAlignment),
        ("options", options, ReadingOptions),
    ):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}")


def reading_at(
    placed: PlacedTrial,
    model_time_s: float,
    alignment: PlateTimeAlignment,
    options: ReadingOptions,
) -> PlateReading:
    """Per-foot load and CoP at ``model_time_s`` with an effective-sigma budget.

    The trial time is ``alignment.trial_time(model_time_s, options.offset_shift_s)``.
    Loads and CoPs are linearly interpolated on the actual (possibly
    non-uniform) sample times; a time outside the trial span is refused, never
    extrapolated.  A foot's CoP is ``None`` when it is unavailable at either
    bracketing sample (never interpolated across an unloaded sample); its CoP
    sigma then reports the plate's declared CoP uncertainty only.

    Preconditions: ``placed``, ``alignment`` and ``options`` of the right types
    (``TypeError``); ``model_time_s`` finite and mapping inside the trial span
    (``ValueError``).  Postcondition: every sigma is finite and non-negative.
    """
    _check_reading_inputs(placed, alignment, options)
    t = alignment.trial_time(model_time_s, options.offset_shift_s)
    bracket = (_locate(placed.trial.times_s, t), t)
    feet = {foot: _foot_sample(placed, foot, bracket, options) for foot in FOOT_LABELS}
    return PlateReading(model_time_s=model_time_s, trial_time_s=t, feet=feet)
