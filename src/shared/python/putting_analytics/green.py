"""Stochastic green-surface sampler (#14172, epic #14168).

Variation is split by how often it is redrawn:

* **Per green** (epistemic, shared by every putt on that green):
  :func:`sample_green` -> :class:`GreenSample` (stimp, slope, fall line,
  grain).
* **Per putt** (aleatory): :func:`sample_roughness` ->
  :class:`RoughnessField`, a sum of random Fourier modes with an analytic
  gradient, evaluated cheaply for any number of balls.

Every component draws from its own ``SeedSequence`` substream keyed by
``(crc32(component), *index)``.  Zeroing one component's SD therefore leaves
every other draw bit-identical (common random numbers / ablation), and a zero
SD returns the nominal value exactly.  Surface laws (friction from stimp,
planar slope convention) are reused from ``putting_dynamics``, not restated.

Truncation: grade, grain strength and stimp are drawn by rejection from their
physical ranges, so a very wide SD slightly shifts the realised moments; the
ranges are inactive for realistic configurations.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import ensure, require, require_finite
from src.shared.python.putting_dynamics.friction import (
    FrictionParams,
    require_grain,
    stimp_to_rolling_mu,
)
from src.shared.python.putting_dynamics.surfaces import (
    FrictionField,
    HeightField,
    SurfaceSpec,
)

__all__ = [
    "GreenSample",
    "GreenVariability",
    "RoughnessField",
    "sample_green",
    "sample_roughness",
]

_STIMP_RANGE_FT = (3.0, 16.0)
_GRADE_RANGE_PCT = (0.0, 10.0)
_GRAIN_RANGE = (0.0, 0.9)
_MAX_REJECTIONS = 10_000
_MAX_AMPLITUDE_M = 0.05
_MAX_CORRELATION_M = 20.0
_MAX_MODES = 4096
_DEFAULT_EXTENT_M = 40.0
_DEFAULT_SPACING_M = 0.25


@dataclass(frozen=True)
class GreenVariability:
    """Nominal green and the spread of every random component.

    Attributes:
        stimp_ft: Nominal stimp [ft]; the lognormal draw has exactly this
            mean.
        stimp_cv: Coefficient of variation of stimp (exact, lognormal).
        grade_percent: Nominal slope grade [%].
        grade_sd_percent: SD of the grade (normal, truncated to [0, 10]).
        aspect_deg: Nominal downhill direction, CCW from +x [deg].
        aspect_sd_deg: SD of the fall-line direction [deg].
        grain_strength: Mean grain anisotropy in [0, 0.9).
        grain_strength_sd: SD of grain strength (truncated to [0, 0.9)).
        grain_direction_deg: Nominal grain direction [deg].
        grain_direction_sd_deg: SD of grain direction [deg] (wrapped normal).
        roughness_amplitude_m: SD of micro-roughness height [m].
        roughness_correlation_m: Roughness correlation length ``ell`` [m].
        roughness_modes: Number of random Fourier modes.
    """

    stimp_ft: float = 10.0
    stimp_cv: float = 0.0
    grade_percent: float = 0.0
    grade_sd_percent: float = 0.0
    aspect_deg: float = 0.0
    aspect_sd_deg: float = 0.0
    grain_strength: float = 0.0
    grain_strength_sd: float = 0.0
    grain_direction_deg: float = 0.0
    grain_direction_sd_deg: float = 0.0
    roughness_amplitude_m: float = 0.0
    roughness_correlation_m: float = 1.0
    roughness_modes: int = 64

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            require_finite(getattr(self, name), name)
        stimp_lo, stimp_hi = _STIMP_RANGE_FT
        require(
            stimp_lo <= self.stimp_ft <= stimp_hi, "stimp in [3, 16]", self.stimp_ft
        )
        require(0.0 <= self.stimp_cv <= 0.5, "stimp_cv in [0, 0.5]", self.stimp_cv)
        require(
            _GRADE_RANGE_PCT[0] <= self.grade_percent <= _GRADE_RANGE_PCT[1],
            "grade in [0, 10] %",
            self.grade_percent,
        )
        require(0.0 <= self.grade_sd_percent <= 5.0, "grade SD in [0, 5]")
        require(-360.0 <= self.aspect_deg <= 360.0, "aspect in [-360, 360]")
        require(0.0 <= self.aspect_sd_deg <= 90.0, "aspect SD in [0, 90] deg")
        require_grain(self.grain_strength, math.radians(self.grain_direction_deg))
        require(0.0 <= self.grain_strength_sd <= 0.5, "grain SD in [0, 0.5]")
        require(0.0 <= self.grain_direction_sd_deg <= 180.0, "grain dir SD in [0, 180]")
        require(
            0.0 <= self.roughness_amplitude_m <= _MAX_AMPLITUDE_M,
            "roughness amplitude in [0, 0.05] m",
            self.roughness_amplitude_m,
        )
        require(
            0.0 < self.roughness_correlation_m <= _MAX_CORRELATION_M,
            "correlation length in (0, 20] m",
            self.roughness_correlation_m,
        )
        require(
            isinstance(self.roughness_modes, int)
            and 1 <= self.roughness_modes <= _MAX_MODES,
            "roughness_modes int in [1, 4096]",
            self.roughness_modes,
        )


@dataclass(frozen=True)
class RoughnessField:
    """Per-putt micro-roughness: a sum of random Fourier modes.

    ``h(x, y) = A sqrt(2/K) sum_i cos(k_i . r + phi_i)`` with wave vectors
    ``k_i`` drawn from the spectral density of the Gaussian covariance
    ``C(r) = A^2 exp(-|r|^2 / (2 ell^2))``, i.e. each axis of ``k`` is
    ``Normal(0, 1/ell^2)``, and phases ``phi_i ~ U[0, 2 pi)``.  Then
    ``Var h = A^2`` and ``Corr(h(r), h(r+d)) = exp(-|d|^2/(2 ell^2))``
    (ensemble sense; ``exp(-1/2)`` at lag ``ell``).  Resolve ``ell`` with
    several grid cells to avoid aliasing when sampled onto a grid.

    Attributes:
        amplitude_m: SD of the height [m].
        correlation_length_m: ``ell`` in the covariance above [m].
        wave_vectors: ``(K, 2)`` array of ``k_i`` [rad/m].
        phases: ``(K,)`` array of ``phi_i`` [rad].
    """

    amplitude_m: float
    correlation_length_m: float
    wave_vectors: np.ndarray
    phases: np.ndarray

    def __post_init__(self) -> None:
        require_finite(self.amplitude_m, "amplitude_m")
        require(self.amplitude_m >= 0.0, "amplitude must be >= 0", self.amplitude_m)
        require_finite(self.correlation_length_m, "correlation_length_m")
        require(self.correlation_length_m > 0.0, "correlation length must be > 0")
        require(
            self.wave_vectors.ndim == 2 and self.wave_vectors.shape[1] == 2,
            "wave_vectors must be (K, 2)",
            self.wave_vectors.shape,
        )
        require(
            self.phases.shape == (self.wave_vectors.shape[0],),
            "phases must be (K,)",
            self.phases.shape,
        )
        require_finite(self.wave_vectors, "wave_vectors")
        require_finite(self.phases, "phases")

    def _terms(self, x_m: object, y_m: object) -> tuple[np.ndarray, np.ndarray]:
        """Mode phase arguments and the amplitude scale for query points."""
        x_arr = np.asarray(x_m, dtype=float)
        y_arr = np.asarray(y_m, dtype=float)
        arg = (
            x_arr[..., None] * self.wave_vectors[:, 0]
            + y_arr[..., None] * self.wave_vectors[:, 1]
            + self.phases
        )
        scale = self.amplitude_m * math.sqrt(2.0 / self.phases.size)
        return arg, np.asarray(scale)

    def height(self, x_m: object, y_m: object) -> np.ndarray:
        """Roughness height [m] at broadcastable ``x_m``, ``y_m`` arrays."""
        shape = np.broadcast(np.asarray(x_m), np.asarray(y_m)).shape
        if self.amplitude_m == 0.0:
            return np.zeros(shape)
        arg, scale = self._terms(x_m, y_m)
        return scale * np.cos(arg).sum(axis=-1)

    def gradient(self, x_m: object, y_m: object) -> tuple[np.ndarray, np.ndarray]:
        """Analytic ``(dh/dx, dh/dy)`` at broadcastable points."""
        shape = np.broadcast(np.asarray(x_m), np.asarray(y_m)).shape
        if self.amplitude_m == 0.0:
            return np.zeros(shape), np.zeros(shape)
        arg, scale = self._terms(x_m, y_m)
        sin_arg = np.sin(arg)
        dhdx = -scale * (sin_arg * self.wave_vectors[:, 0]).sum(axis=-1)
        dhdy = -scale * (sin_arg * self.wave_vectors[:, 1]).sum(axis=-1)
        return dhdx, dhdy


@dataclass(frozen=True)
class GreenSample:
    """One green draw, shared by all putts on that green.

    Attributes:
        stimp_ft: Realised stimp [ft].
        grade_percent: Realised slope grade [%].
        aspect_deg: Realised downhill direction, CCW from +x [deg].
        grain_strength: Realised grain anisotropy in [0, 0.9).
        grain_direction_rad: Realised grain direction [rad].
    """

    stimp_ft: float
    grade_percent: float
    aspect_deg: float
    grain_strength: float
    grain_direction_rad: float

    def friction_params(self) -> FrictionParams:
        """Friction law for this green (rolling mu from the stimp law)."""
        return FrictionParams(
            mu_roll0=stimp_to_rolling_mu(self.stimp_ft),
            grain_strength=self.grain_strength,
            grain_direction_rad=self.grain_direction_rad,
        )

    def to_surface_spec(
        self,
        roughness: RoughnessField | None = None,
        extent_m: float = _DEFAULT_EXTENT_M,
        spacing_m: float = _DEFAULT_SPACING_M,
    ) -> SurfaceSpec:
        """Solver-ready surface: planar slope plus optional roughness.

        Roughness is sampled onto the planar grid nodes; the friction field
        is uniform and grain/stimp enter through :meth:`friction_params`.
        """
        base = HeightField.planar(
            self.grade_percent, self.aspect_deg, extent_m, spacing_m
        )
        height = base
        if roughness is not None:
            ny, nx = base.heights_m.shape
            xs = base.origin_m[0] + base.spacing_m * np.arange(nx)
            ys = base.origin_m[1] + base.spacing_m * np.arange(ny)
            grid_x, grid_y = np.meshgrid(xs, ys)  # indexed [iy, ix]
            height = HeightField(
                heights_m=base.heights_m + roughness.height(grid_x, grid_y),
                spacing_m=base.spacing_m,
                origin_m=base.origin_m,
            )
        return SurfaceSpec(
            height=height,
            friction_field=FrictionField.uniform(extent_m, spacing_m),
            friction=self.friction_params(),
        )


def _stream(seed: int, component: str, *index: int) -> np.random.Generator:
    """Independent substream keyed by ``(crc32(component), *index)``."""
    require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        "seed must be an int >= 0",
        seed,
    )
    require(all(i >= 0 for i in index), "stream indices must be >= 0", index)
    key = (zlib.crc32(component.encode()), *index)
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def _draw_in_range(
    draw: Callable[[], float], low: float, high: float, closed_high: bool
) -> float:
    """Rejection-sample ``draw()`` into ``[low, high]`` (or ``[low, high)``)."""
    for _ in range(_MAX_REJECTIONS):
        value = float(draw())
        if low <= value and (value <= high if closed_high else value < high):
            return value
    raise ValueError("rejection sampling failed; SD too wide for the range")


def _normal_in_range(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    bounds: tuple[float, float],
    closed_high: bool = True,
) -> float:
    """Truncated normal; ``mean`` is returned exactly when ``sd == 0``."""
    if sd == 0.0:
        return mean
    low, high = bounds
    return _draw_in_range(lambda: rng.normal(mean, sd), low, high, closed_high)


def _wrap_deg(angle_deg: float) -> float:
    """Wrap an angle to ``[-180, 180)`` degrees."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _lognormal_stimp(rng: np.random.Generator, mean: float, cv: float) -> float:
    """Lognormal with exact mean ``mean`` and exact CV ``cv``."""
    if cv == 0.0:
        return mean
    sigma2 = math.log1p(cv * cv)
    mu = math.log(mean) - 0.5 * sigma2
    low, high = _STIMP_RANGE_FT
    return _draw_in_range(
        lambda: math.exp(rng.normal(mu, math.sqrt(sigma2))), low, high, True
    )


def sample_green(config: GreenVariability, seed: int) -> GreenSample:
    """Draw one per-green sample with an independent stream per component.

    Args:
        config: Nominal values and spreads.
        seed: Green seed (>= 0); the same seed gives the same green, and
            zeroing one SD changes only that component.

    Returns:
        The realised :class:`GreenSample`.
    """
    stimp = _lognormal_stimp(_stream(seed, "stimp"), config.stimp_ft, config.stimp_cv)
    grade = _normal_in_range(
        _stream(seed, "grade"),
        config.grade_percent,
        config.grade_sd_percent,
        _GRADE_RANGE_PCT,
    )
    aspect = config.aspect_deg
    if config.aspect_sd_deg > 0.0:
        aspect = _wrap_deg(
            aspect + _stream(seed, "aspect").normal(0.0, config.aspect_sd_deg)
        )
    grain = _normal_in_range(
        _stream(seed, "grain_strength"),
        config.grain_strength,
        config.grain_strength_sd,
        _GRAIN_RANGE,
        closed_high=False,
    )
    grain_dir = config.grain_direction_deg
    if config.grain_direction_sd_deg > 0.0:
        grain_dir = _wrap_deg(
            grain_dir
            + _stream(seed, "grain_direction").normal(
                0.0, config.grain_direction_sd_deg
            )
        )
    sample = GreenSample(stimp, grade, aspect, grain, math.radians(grain_dir))
    ensure(
        _STIMP_RANGE_FT[0] <= sample.stimp_ft <= _STIMP_RANGE_FT[1],
        "sampled stimp within physical range",
        sample.stimp_ft,
    )
    return sample


def sample_roughness(
    config: GreenVariability, seed: int, putt_index: int
) -> RoughnessField:
    """Draw the per-putt roughness field for putt ``putt_index``.

    Args:
        config: Provides amplitude, correlation length and mode count.
        seed: Master seed (>= 0).
        putt_index: Putt number (>= 0); each putt has its own substream.

    Returns:
        A :class:`RoughnessField` (identically zero when the amplitude is 0;
        wave vectors are still drawn so streams stay aligned).
    """
    rng = _stream(seed, "roughness", putt_index)
    modes = config.roughness_modes
    wave_vectors = rng.normal(0.0, 1.0 / config.roughness_correlation_m, (modes, 2))
    phases = rng.uniform(0.0, 2.0 * math.pi, modes)
    return RoughnessField(
        amplitude_m=config.roughness_amplitude_m,
        correlation_length_m=config.roughness_correlation_m,
        wave_vectors=wave_vectors,
        phases=phases,
    )
