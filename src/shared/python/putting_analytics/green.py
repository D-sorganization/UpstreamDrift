"""Stochastic green-surface sampler (#14172, epic #14168).

Per-green variation (stimp, slope, fall line, grain) is drawn once per green by
:func:`sample_green`; per-putt micro-roughness is a random-Fourier-mode field
from :func:`sample_roughness` with an analytic gradient.  Each component uses
its own ``SeedSequence`` substream keyed by ``(crc32(component), *index)``, so
zeroing one SD leaves every other draw bit-identical (ablation) and a zero SD
returns the nominal value exactly.  Friction and slope conventions come from
``putting_dynamics``.  Grade, grain strength and stimp are drawn by rejection
from their physical ranges, which is inactive for realistic configurations.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require, require_finite
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
_MAX_MODES = 4096
_CONFIG_RANGES = {
    "stimp_ft": _STIMP_RANGE_FT,
    "stimp_cv": (0.0, 0.5),
    "grade_percent": _GRADE_RANGE_PCT,
    "grade_sd_percent": (0.0, 5.0),
    "aspect_deg": (-360.0, 360.0),
    "aspect_sd_deg": (0.0, 90.0),
    "grain_strength": _GRAIN_RANGE,
    "grain_strength_sd": (0.0, 0.5),
    "grain_direction_deg": (-360.0, 360.0),
    "grain_direction_sd_deg": (0.0, 180.0),
    "roughness_amplitude_m": (0.0, 0.05),
    "roughness_correlation_m": (1e-3, 20.0),
}
_DEFAULT_EXTENT_M = 40.0
_DEFAULT_SPACING_M = 0.25


@dataclass(frozen=True)
class GreenVariability:
    """Nominal green and the spread of every random component.

    Stimp is lognormal with exact mean/CV; grade and grain strength are
    truncated normals; aspect and grain direction are wrapped normals.
    Angles are in degrees, grade in percent, roughness amplitude (SD of the
    height) and correlation length ``ell`` in metres; ``roughness_modes`` is
    the number of random Fourier modes.  Every ``*_sd`` / ``*_cv`` of 0
    disables that component.
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
        for name, (low, high) in _CONFIG_RANGES.items():
            value = getattr(self, name)
            require_finite(value, name)
            require(low <= value <= high, f"{name} in [{low}, {high}]", value)
        require_grain(self.grain_strength, math.radians(self.grain_direction_deg))
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
    ``k_i`` from the spectral density of the Gaussian covariance
    ``C(r) = A^2 exp(-|r|^2 / (2 ell^2))`` (each axis ``Normal(0, 1/ell^2)``)
    and phases ``U[0, 2 pi)``.  Then ``Var h = A^2`` and, in the ensemble
    sense, ``Corr(h(r), h(r+d)) = exp(-|d|^2/(2 ell^2))`` (``exp(-1/2)`` at
    lag ``ell``).  Resolve ``ell`` with several grid cells to avoid aliasing.

    Attributes: ``amplitude_m`` (A), ``correlation_length_m`` (ell),
    ``wave_vectors`` ``(K, 2)`` [rad/m] and ``phases`` ``(K,)`` [rad].
    """

    amplitude_m: float
    correlation_length_m: float
    wave_vectors: np.ndarray
    phases: np.ndarray

    def __post_init__(self) -> None:
        require_finite(self.amplitude_m, "amplitude_m")
        require(self.amplitude_m >= 0.0, "amplitude must be >= 0", self.amplitude_m)
        require(self.correlation_length_m > 0.0, "correlation length must be > 0")
        shape = self.wave_vectors.shape
        require(len(shape) == 2 and shape[1] == 2, "wave_vectors must be (K, 2)", shape)
        require(self.phases.shape == (shape[0],), "phases must be (K,)")
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

    Stimp [ft], grade [%], downhill ``aspect_deg`` (CCW from +x), grain
    strength in [0, 0.9) and grain direction [rad].
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
        """Planar slope plus optional roughness sampled onto the grid nodes."""
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


def _draw_in_range(draw: Callable[[], float], low: float, high: float) -> float:
    """Rejection-sample ``draw()`` into ``[low, high)``."""
    for _ in range(_MAX_REJECTIONS):
        value = float(draw())
        if low <= value < high:
            return value
    raise ValueError("rejection sampling failed; SD too wide for the range")


def _normal_in_range(
    rng: np.random.Generator, mean: float, sd: float, bounds: tuple[float, float]
) -> float:
    """Truncated normal; ``mean`` is returned exactly when ``sd == 0``."""
    if sd == 0.0:
        return mean
    return _draw_in_range(lambda: rng.normal(mean, sd), *bounds)


def _wrapped_normal(rng: np.random.Generator, mean_deg: float, sd_deg: float) -> float:
    """Angle ``mean + N(0, sd)`` wrapped to [-180, 180); exact ``mean`` if sd=0."""
    if sd_deg == 0.0:
        return mean_deg
    return (mean_deg + rng.normal(0.0, sd_deg) + 180.0) % 360.0 - 180.0


def _lognormal_stimp(rng: np.random.Generator, mean: float, cv: float) -> float:
    """Lognormal with exact mean ``mean`` and exact CV ``cv``."""
    if cv == 0.0:
        return mean
    sigma2 = math.log1p(cv * cv)
    mu = math.log(mean) - 0.5 * sigma2
    sigma = math.sqrt(sigma2)
    return _draw_in_range(lambda: math.exp(rng.normal(mu, sigma)), *_STIMP_RANGE_FT)


def sample_green(config: GreenVariability, seed: int) -> GreenSample:
    """Draw one per-green sample (same ``seed`` -> same green)."""
    stimp = _lognormal_stimp(_stream(seed, "stimp"), config.stimp_ft, config.stimp_cv)
    grade = _normal_in_range(
        _stream(seed, "grade"),
        config.grade_percent,
        config.grade_sd_percent,
        _GRADE_RANGE_PCT,
    )
    aspect = _wrapped_normal(
        _stream(seed, "aspect"), config.aspect_deg, config.aspect_sd_deg
    )
    grain = _normal_in_range(
        _stream(seed, "grain_strength"),
        config.grain_strength,
        config.grain_strength_sd,
        _GRAIN_RANGE,
    )
    grain_dir = _wrapped_normal(
        _stream(seed, "grain_direction"),
        config.grain_direction_deg,
        config.grain_direction_sd_deg,
    )
    sample = GreenSample(stimp, grade, aspect, grain, math.radians(grain_dir))
    return sample


def sample_roughness(
    config: GreenVariability, seed: int, putt_index: int
) -> RoughnessField:
    """Per-putt roughness field; zero amplitude gives an exactly-zero field.

    Wave vectors are drawn even then, so streams stay aligned for ablation.
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
