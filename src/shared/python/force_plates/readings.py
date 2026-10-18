"""Plate readings at an arbitrary model time with an uncertainty budget (OF-05).

:func:`reading_at` evaluates a placed vertical plate trial at a model time and
reports, per foot, the load and the centre of pressure (CoP) together with the
components of its uncertainty:

* the declared measurement uncertainty of the plate;
* a within-sample **timing skew** between channels, ``skew_s * |dy/dt|``
  (at an interior sample, the steeper adjacent interval);
* the **linear-interpolation bias estimate** of :mod:`.interpolation`;
* the common-mode clock offset, which is **not** folded in here: it is shared
  by every frame, so it is profiled with ``ReadingOptions.offset_shift_s``.

``*_sigma_random_*`` is ``hypot(measurement, skew)`` and ``*_bias_*`` the bias
estimate.  The convenience ``sigma_*`` is ``random + bias / coverage``: a
worst-case band equivalent (the bias is added, not combined in quadrature), so
``coverage * sigma`` equals ``coverage * random + bias`` exactly, and only for
that ``coverage``.  It is not a 1-sigma.

Validity of the bias estimate: it is an estimate, not a bound.  It is valid
when the signal is band-limited well below the sample rate so that ``y''`` is
bounded by its neighbouring second divided differences; it is NOT valid across
an under-sampled impact transient.  Callers must exclude or separately bound
such intervals.  A missing curvature (two samples, or NaN neighbours) is
refused unless a curvature floor is declared.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from src.shared.python.force_plates.alignment import PlateTimeAlignment
from src.shared.python.force_plates.contracts import (
    FOOT_LABELS,
    Array,
    VerticalPlateTrial,
    finite_non_negative,
    finite_positive,
    finite_real,
    require_foot_keys,
)
from src.shared.python.force_plates.interpolation import (
    Channel,
    Estimate,
    interpolate,
    locate,
    snap,
)
from src.shared.python.force_plates.placement import PlatePlacement

__all__ = [
    "FootSample",
    "PlacedTrial",
    "PlateReading",
    "ReadingOptions",
    "covers",
    "reading_at",
]

_WORLD_DIM = 3
_STANCE_DIM = 2
#: ``(var_x, var_y, cov_xy)``: the independent terms of a 2x2 covariance.
_COV_TERMS = 3
#: Relative slack on ``cov_xy^2 <= var_x var_y`` for floating-point rounding.
_PSD_REL_SLACK = 1.0e-12
#: Rounding allowance, in ulps of the largest clock term, for snapping a
#: trial time onto a sample (two subtractions each round by <= 1/2 ulp).
_SNAP_ULPS = 4


def _optional_non_negative(value: float | None, name: str) -> None:
    if value is not None:
        finite_non_negative(value, name)


@dataclass(frozen=True)
class ReadingOptions:
    """How a reading's uncertainty budget is formed.

    ``coverage`` is the band factor ``k > 0`` that converts the interpolation
    bias estimate into an equivalent sigma (``bias / k``).  ``skew_s >= 0`` is
    the declared timing skew between channels within one sample.
    ``offset_shift_s`` shifts the common-mode clock offset (profiling).
    ``load_curvature_floor`` [N/s^2] and ``cop_curvature_floor`` [m/s^2] are
    optional declared floors on ``|y''|`` for the load and the CoP; they are
    required wherever the curvature cannot be estimated from the data.
    ``cop_skew_independent`` declares that the two stance CoP channels' skews
    are independent; with ``skew_s > 0`` the CoP covariance is only reported
    when it is ``True`` (a skew common to both channels would correlate them
    along the signed CoP velocity, which a reading does not keep).

    Raises ``TypeError`` for non-numbers or a non-bool ``cop_skew_independent``,
    ``ValueError`` for out-of-range values.
    """

    coverage: float
    skew_s: float = 0.0
    offset_shift_s: float = 0.0
    load_curvature_floor: float | None = None
    cop_curvature_floor: float | None = None
    cop_skew_independent: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.cop_skew_independent, bool):
            raise TypeError("cop_skew_independent must be a bool")
        finite_positive(self.coverage, "coverage")
        finite_non_negative(self.skew_s, "skew_s")
        finite_real(self.offset_shift_s, "offset_shift_s")
        _optional_non_negative(self.load_curvature_floor, "load_curvature_floor")
        _optional_non_negative(self.cop_curvature_floor, "cop_curvature_floor")


@dataclass(frozen=True, eq=False)
class FootSample:
    """One foot's reading: load, CoP on the sensing plane, and uncertainties.

    ``cop_m`` is the world point ``(3,)`` on the plate's force-sensing plane
    (read-only), or ``None`` when the foot is unloaded at either bracketing
    sample.  ``sigma_load_n`` and ``sigma_cop_m`` are worst-case band-equivalent
    sigmas (module docstring), the CoP one per horizontal axis for the worst
    axis; ``*_sigma_random_*`` and ``*_bias_*`` are their components (``None``
    when a sample was built without them).  ``cop_sigma_random_xy_m`` is the
    CoP's random 1-sigma for world x and y separately, whose maximum is
    ``cop_sigma_random_m``; it bounds each world axis through ``|R_xy|``.
    ``cop_cov_random_xy_m2`` is the CoP's random covariance in world
    ``(var_x, var_y, cov_xy)`` [m^2], its principal axes the plate's; ``None``
    when a skew is declared without declaring the channels' skews independent.
    ``sigma_sensing_m`` is the uncertainty of the sensing-plane depth.
    """

    load_n: float
    sigma_load_n: float
    cop_m: Array | None
    sigma_cop_m: float
    sigma_sensing_m: float
    load_sigma_random_n: float | None = None
    load_bias_n: float | None = None
    cop_sigma_random_m: float | None = None
    cop_bias_m: float | None = None
    cop_sigma_random_xy_m: tuple[float, float] | None = None
    cop_cov_random_xy_m2: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        finite_real(self.load_n, "load_n")
        finite_non_negative(self.sigma_load_n, "sigma_load_n")
        finite_non_negative(self.sigma_cop_m, "sigma_cop_m")
        finite_non_negative(self.sigma_sensing_m, "sigma_sensing_m")
        for name in (
            "load_sigma_random_n",
            "load_bias_n",
            "cop_sigma_random_m",
            "cop_bias_m",
        ):
            _optional_non_negative(getattr(self, name), name)
        _check_axis_pair(self.cop_sigma_random_xy_m, "cop_sigma_random_xy_m")
        _check_covariance(self.cop_cov_random_xy_m2, "cop_cov_random_xy_m2")
        if self.cop_m is not None:
            object.__setattr__(self, "cop_m", _frozen_world_point(self.cop_m))


def _check_axis_pair(value: tuple[float, float] | None, name: str) -> None:
    """``None`` or two finite non-negative numbers (world x, world y)."""
    if value is None:
        return
    if len(value) != _STANCE_DIM:
        raise ValueError(f"{name} must hold one value per world axis, got {value!r}")
    for axis in value:
        finite_non_negative(axis, name)


def _check_covariance(value: tuple[float, float, float] | None, name: str) -> None:
    """``None`` or ``(var_x, var_y, cov_xy)`` of a positive semi-definite matrix."""
    if value is None:
        return
    if len(value) != _COV_TERMS:
        raise ValueError(f"{name} must be (var_x, var_y, cov_xy), got {value!r}")
    var_x, var_y, cov_xy = value
    finite_non_negative(var_x, name)
    finite_non_negative(var_y, name)
    finite_real(cov_xy, name)
    # Rounding of R diag(v) R^T can overshoot the bound by a few ulps.
    slack = _PSD_REL_SLACK * max(var_x, var_y) ** 2
    if cov_xy**2 > var_x * var_y + slack:
        raise ValueError(f"{name} is not a covariance: |cov_xy| > sqrt(var_x var_y)")


def _frozen_world_point(value: Array) -> Array:
    cop = np.array(value, dtype=np.float64, copy=True)
    if cop.shape != (_WORLD_DIM,) or not np.isfinite(cop).all():
        raise ValueError(f"cop_m must be a finite 3-vector, got shape {cop.shape}")
    cop.flags.writeable = False
    return cop


@dataclass(frozen=True)
class PlateReading:
    """Both feet at one model time; ``feet`` is keyed by ``FOOT_LABELS``.

    ``coverage`` records the band factor the sigmas were formed with.
    """

    model_time_s: float
    trial_time_s: float
    feet: Mapping[str, FootSample]
    coverage: float | None = None

    def __post_init__(self) -> None:
        finite_real(self.model_time_s, "model_time_s")
        finite_real(self.trial_time_s, "trial_time_s")
        require_foot_keys(self.feet, "feet")
        for label, sample in self.feet.items():
            if not isinstance(sample, FootSample):
                raise TypeError(f"feet[{label!r}] must be a FootSample")
        if self.coverage is not None:
            finite_positive(self.coverage, "coverage")
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

    def min_sample_interval_s(self) -> float:
        """Smallest interval between consecutive samples [s]."""
        return float(np.min(np.diff(self.trial.times_s)))

    def horizontal_rotation(self) -> Array:
        """``R_xy`` (2x2): a stance-x/y error ``e`` is the world error ``R_xy e``."""
        return np.array(self.placement.rotation[:2, :2])

    def world_axis_gain(self) -> Array:
        """``|R_xy|`` (2x2): ``|R_xy| @ |e|`` bounds each world-axis component."""
        return np.abs(self.horizontal_rotation())


# ------------------------------------------------------------ readings --
@dataclass(frozen=True)
class _Budget:
    """Random part, bias estimate and band-equivalent sigma of one quantity."""

    random: float
    bias: float
    sigma: float


def _budget(base: float, estimate: Estimate, options: ReadingOptions) -> _Budget:
    """``random = hypot(base, skew * rate)``; ``sigma = random + bias / coverage``."""
    random = math.hypot(base, options.skew_s * estimate.rate)
    return _Budget(random, estimate.bias, random + estimate.bias / options.coverage)


@dataclass(frozen=True, eq=False)
class _CopEstimate:
    """Interpolated stance CoP ``(2,)``, the worst-axis bias and rate, and the
    rate on each stance and (bounded through ``|R_xy|``) world axis ``(2,)``."""

    xy: Array
    estimate: Estimate
    stance_rates: Array
    world_rates: Array


def _cop_estimate(
    channel: Channel, index: int, t: float, gain: Array
) -> _CopEstimate | None:
    """Per-axis CoP estimates in world axes; ``None`` if unloaded.

    ``channel.values`` is the ``(T, 2)`` stance CoP and ``gain`` the
    :meth:`PlacedTrial.world_axis_gain`.  Stance bias and rate magnitudes are
    carried to world axes by ``gain`` (conservative under yaw); the worst
    world axis is reported, so bias and rate may come from different axes.
    """
    if np.isnan(channel.values[index : index + 2]).any():
        return None
    axes = [
        interpolate(replace(channel, values=channel.values[:, axis]), index, t)
        for axis in range(_STANCE_DIM)
    ]
    stance_rates = np.array([axis.rate for axis in axes])
    world_rates = gain @ stance_rates
    merged = Estimate(
        value=0.0,
        bias=float(np.max(gain @ np.array([axis.bias for axis in axes]))),
        rate=float(np.max(world_rates)),
    )
    return _CopEstimate(
        np.array([axis.value for axis in axes]), merged, stance_rates, world_rates
    )


def _cop_random_xy(
    cop: _CopEstimate | None, base: float, skew_s: float
) -> tuple[float, float]:
    """Random 1-sigma on world x and y: ``hypot(base, skew * world rate)``.

    An unloaded CoP has no rate, so both axes carry the declared ``base``.
    """
    if cop is None:
        return base, base
    sx, sy = (math.hypot(base, skew_s * float(rate)) for rate in cop.world_rates)
    return sx, sy


def _cop_random_cov(
    cop: _CopEstimate | None, base: float, options: ReadingOptions, rotation_xy: Array
) -> tuple[float, float, float] | None:
    """World random covariance ``(var_x, var_y, cov_xy)`` of the CoP, or ``None``.

    With independent channel skews (declared, or no skew at all) the stance
    covariance is ``diag(base^2 + (skew * stance rate)^2)`` and the yaw
    carries it to world axes as ``R_xy diag R_xy^T``.  An unloaded CoP has no
    rate: ``base^2`` on both stance axes, which no rotation changes.  A skew
    without declared independence gives ``None``: its cross term is unknown.
    """
    if options.skew_s > 0.0 and not options.cop_skew_independent:
        return None
    rates = np.zeros(_STANCE_DIM) if cop is None else cop.stance_rates
    stance_var = base**2 + (options.skew_s * rates) ** 2
    cov = rotation_xy @ np.diag(stance_var) @ rotation_xy.T
    # Average the off-diagonals so the reported matrix is exactly symmetric.
    return float(cov[0, 0]), float(cov[1, 1]), 0.5 * float(cov[0, 1] + cov[1, 0])


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
    times = trial.times_s
    load_channel = Channel(
        times,
        trial.load_n[foot],
        options.load_curvature_floor,
        f"{foot} load_n",
        "load_curvature_floor",
    )
    load = interpolate(load_channel, index, t)
    load_budget = _budget(
        float(trial.uncertainty.force_sigma_n(load.value)), load, options
    )
    cop_channel = Channel(
        times,
        trial.cop_m[foot],
        options.cop_curvature_floor,
        f"{foot} cop_m",
        "cop_curvature_floor",
    )
    cop = _cop_estimate(cop_channel, index, t, placed.world_axis_gain())
    base_cop = trial.uncertainty.cop_m
    cop_budget = (
        _Budget(base_cop, 0.0, base_cop)
        if cop is None
        else _budget(base_cop, cop.estimate, options)
    )
    return FootSample(
        load_n=load.value,
        sigma_load_n=load_budget.sigma,
        cop_m=None if cop is None else _world_cop(placed, cop.xy),
        sigma_cop_m=cop_budget.sigma,
        sigma_sensing_m=trial.uncertainty.sensing_depth_sigma_m,
        load_sigma_random_n=load_budget.random,
        load_bias_n=load_budget.bias,
        cop_sigma_random_m=cop_budget.random,
        cop_bias_m=cop_budget.bias,
        cop_sigma_random_xy_m=_cop_random_xy(cop, base_cop, options.skew_s),
        cop_cov_random_xy_m2=_cop_random_cov(
            cop, base_cop, options, placed.horizontal_rotation()
        ),
    )


def _check_reading_inputs(placed: object, alignment: object, options: object) -> None:
    for name, value, kind in (
        ("placed", placed, PlacedTrial),
        ("alignment", alignment, PlateTimeAlignment),
        ("options", options, ReadingOptions),
    ):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}")


def _trial_time(
    times: Array,
    model_time_s: float,
    alignment: PlateTimeAlignment,
    options: ReadingOptions,
) -> float:
    """Trial time of ``model_time_s``, snapped onto a sample within rounding."""
    t = alignment.trial_time(model_time_s, options.offset_shift_s)
    scale = max(
        abs(model_time_s), abs(alignment.offset_s), abs(options.offset_shift_s), abs(t)
    )
    return snap(times, t, _SNAP_ULPS * math.ulp(scale))


def reading_at(
    placed: PlacedTrial,
    model_time_s: float,
    alignment: PlateTimeAlignment,
    options: ReadingOptions,
) -> PlateReading:
    """Per-foot load and CoP at ``model_time_s`` with an uncertainty budget.

    The trial time is ``alignment.trial_time(model_time_s, options.offset_shift_s)``.
    Loads and CoPs are linearly interpolated on the actual (possibly
    non-uniform) sample times; a time outside the trial span is refused, never
    extrapolated.  A foot's CoP is ``None`` when it is unavailable at either
    bracketing sample (never interpolated across an unloaded sample); its CoP
    sigma then reports the plate's declared CoP uncertainty only.

    The interpolation bias is an *estimate*, valid when the signal is
    band-limited well below the sample rate so that ``y''`` is bounded by its
    neighbouring second divided differences; NOT valid across an under-sampled
    impact transient, which callers must exclude or separately bound.  Where
    the curvature cannot be estimated (two samples, NaN neighbours) a
    ``ReadingOptions`` curvature floor must be declared or ``ValueError`` is
    raised.  See the module docstring for the sigma composition.

    Preconditions: ``placed``, ``alignment`` and ``options`` of the right types
    (``TypeError``); ``model_time_s`` finite and mapping inside the trial span
    (``ValueError``).  Postcondition: every sigma is finite and non-negative.
    """
    _check_reading_inputs(placed, alignment, options)
    times = placed.trial.times_s
    t = _trial_time(times, model_time_s, alignment, options)
    bracket = (locate(times, t), t)
    feet = {foot: _foot_sample(placed, foot, bracket, options) for foot in FOOT_LABELS}
    return PlateReading(model_time_s, t, feet, coverage=options.coverage)


def covers(
    placed: PlacedTrial,
    model_time_s: float,
    alignment: PlateTimeAlignment,
    options: ReadingOptions,
) -> bool:
    """Whether :func:`reading_at` can read ``model_time_s`` (inside the trial span).

    Uses the same trial-time mapping and rounding snap as :func:`reading_at`,
    so a ``True`` here never meets an extrapolation refusal there.
    """
    _check_reading_inputs(placed, alignment, options)
    start, end = placed.span_s()
    t = _trial_time(placed.trial.times_s, model_time_s, alignment, options)
    return start <= t <= end
