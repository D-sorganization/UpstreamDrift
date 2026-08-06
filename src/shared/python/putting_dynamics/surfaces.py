"""Heterogeneous green surfaces: height, friction, bumpiness (#8345, P2).

Reused shared infrastructure (AGENTS.md section A discovery):

* DbC helpers from ``src.shared.python.contracts``.
* Seeding discipline reused from the Tools variation engine
  (``swing_sim.variation.engine._stream_for``, Tools branch
  ``feat/putting-vertical``): every stochastic field draws from its own
  ``numpy`` generator keyed by ``[seed, crc32(field_key)]`` so adding or
  removing one field never perturbs another field's draws, and the same
  seed always reproduces the identical field (test-pinned).

Model
-----
A green is a rectangular grid.  :class:`HeightField` stores node
elevations and answers ``elevation`` / ``gradient`` queries by bilinear
interpolation (the gradient is the exact derivative of the bilinear
patch, so the two are always consistent).  :class:`FrictionField`
stores per-node multipliers applied on top of the base
:class:`~.friction.FrictionParams` coefficients.  Bumpy variants add
seeded, correlation-smoothed noise: white noise on the grid is smoothed
with ``n`` box-blur passes whose footprint matches the requested
correlation length (three passes approximate a Gaussian kernel by the
central limit theorem), then rescaled to the requested amplitude
(standard deviation for heights, fractional deviation for friction).

Queries outside the grid clamp to the border cell (flat continuation),
so the solver never extrapolates unboundedly.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import ensure, require, require_finite

__all__ = [
    "FrictionField",
    "HeightField",
    "SurfaceSpec",
    "bumpy_friction_field",
    "bumpy_height_field",
]

from .friction import FrictionParams

#: Default grid extent [m] (40 m covers the Tools putting range).
_DEFAULT_EXTENT_M = 40.0

#: Default grid spacing [m].
_DEFAULT_SPACING_M = 0.25

#: Box-blur passes used to approximate Gaussian smoothing (CLT).
_BLUR_PASSES = 3


def _validate_grid(values: np.ndarray, spacing_m: float, name: str) -> None:
    """Shared grid preconditions for the field dataclasses."""
    require(isinstance(values, np.ndarray), f"{name} must be a numpy array")
    require(values.ndim == 2, f"{name} must be a 2-D grid", values.shape)
    require(
        values.shape[0] >= 2 and values.shape[1] >= 2,
        f"{name} needs at least a 2x2 grid",
        values.shape,
    )
    require_finite(values, name)
    require_finite(spacing_m, "spacing_m")
    require(0.001 <= spacing_m <= 10.0, "spacing in [0.001, 10] m", spacing_m)


def _validate_origin(origin_m: tuple[float, float]) -> None:
    """Validate the shared two-dimensional field origin contract."""
    require(len(origin_m) == 2, "origin_m must contain x and y", origin_m)
    require_finite(origin_m[0], "origin_m[0]")
    require_finite(origin_m[1], "origin_m[1]")


def _validate_factory_grid(extent_m: float, spacing_m: float) -> int:
    """Validate public factory dimensions and return the node count."""
    require_finite(extent_m, "extent_m")
    require(1.0 <= extent_m <= 200.0, "extent in [1, 200] m", extent_m)
    require_finite(spacing_m, "spacing_m")
    require(0.001 <= spacing_m <= 10.0, "spacing in [0.001, 10] m", spacing_m)
    node_count = int(round(extent_m / spacing_m)) + 1
    require(node_count >= 2, "grid must contain at least two nodes", node_count)
    return node_count


def _bilinear_setup(
    x_m: float, y_m: float, origin: tuple[float, float], spacing: float, shape: tuple
) -> tuple[int, int, float, float]:
    """Clamped cell index and in-cell fractions for a query point."""
    fx = (x_m - origin[0]) / spacing
    fy = (y_m - origin[1]) / spacing
    ix = int(min(max(math.floor(fx), 0), shape[1] - 2))
    iy = int(min(max(math.floor(fy), 0), shape[0] - 2))
    tx = min(max(fx - ix, 0.0), 1.0)
    ty = min(max(fy - iy, 0.0), 1.0)
    return ix, iy, tx, ty


@dataclass(frozen=True)
class HeightField:
    """Bilinear-interpolated heightmap.

    Attributes:
        heights_m: Node elevations [m], indexed ``[iy, ix]``.
        spacing_m: Grid spacing [m] (square cells).
        origin_m: World coordinates of node ``[0, 0]`` (x, y) [m].
    """

    heights_m: np.ndarray
    spacing_m: float = _DEFAULT_SPACING_M
    origin_m: tuple[float, float] = (-_DEFAULT_EXTENT_M / 2.0, -_DEFAULT_EXTENT_M / 2.0)

    def __post_init__(self) -> None:
        _validate_grid(self.heights_m, self.spacing_m, "heights_m")
        _validate_origin(self.origin_m)
        owned = np.array(self.heights_m, dtype=float, copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "heights_m", owned)

    @staticmethod
    def flat(
        extent_m: float = _DEFAULT_EXTENT_M,
        spacing_m: float = _DEFAULT_SPACING_M,
    ) -> HeightField:
        """Perfectly flat green centered on the origin.

        Args:
            extent_m: Side length of the square grid [m].
            spacing_m: Grid spacing [m].

        Returns:
            A zero-elevation :class:`HeightField`.
        """
        n = _validate_factory_grid(extent_m, spacing_m)
        return HeightField(
            heights_m=np.zeros((n, n)),
            spacing_m=spacing_m,
            origin_m=(-extent_m / 2.0, -extent_m / 2.0),
        )

    @staticmethod
    def planar(
        grade_percent: float,
        aspect_deg: float,
        extent_m: float = _DEFAULT_EXTENT_M,
        spacing_m: float = _DEFAULT_SPACING_M,
    ) -> HeightField:
        """Uniformly sloped plane.

        Convention matches Tools ``swing_sim.putting.green`` (restated
        with credit): ``aspect_deg`` is the compass direction the green
        falls *toward*, CCW from +x (0 = downhill along +x).

        Args:
            grade_percent: Slope grade [%] in [0, 10].
            aspect_deg: Downhill direction, CCW from +x [deg].
            extent_m: Side length of the square grid [m].
            spacing_m: Grid spacing [m].

        Returns:
            A planar :class:`HeightField` whose gradient magnitude is
            ``grade_percent / 100`` everywhere.
        """
        require_finite(grade_percent, "grade_percent")
        require(0.0 <= grade_percent <= 10.0, "grade in [0, 10] %", grade_percent)
        require_finite(aspect_deg, "aspect_deg")
        require(-360.0 <= aspect_deg <= 360.0, "aspect in [-360, 360]", aspect_deg)
        n = _validate_factory_grid(extent_m, spacing_m)
        aspect = math.radians(aspect_deg)
        grade = grade_percent / 100.0
        # Downhill toward `aspect` means height decreases along it.
        xs = np.arange(n) * spacing_m - extent_m / 2.0
        gx, gy = -grade * math.cos(aspect), -grade * math.sin(aspect)
        heights = gy * xs[:, None] + gx * xs[None, :]
        return HeightField(
            heights_m=heights,
            spacing_m=spacing_m,
            origin_m=(-extent_m / 2.0, -extent_m / 2.0),
        )

    def elevation(self, x_m: float, y_m: float) -> float:
        """Bilinear surface elevation at a point [m]."""
        require_finite(x_m, "x_m")
        require_finite(y_m, "y_m")
        ix, iy, tx, ty = _bilinear_setup(
            x_m, y_m, self.origin_m, self.spacing_m, self.heights_m.shape
        )
        h = self.heights_m
        return float(
            h[iy, ix] * (1 - tx) * (1 - ty)
            + h[iy, ix + 1] * tx * (1 - ty)
            + h[iy + 1, ix] * (1 - tx) * ty
            + h[iy + 1, ix + 1] * tx * ty
        )

    def gradient(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Exact gradient of the bilinear patch, ``(dh/dx, dh/dy)``."""
        require_finite(x_m, "x_m")
        require_finite(y_m, "y_m")
        ix, iy, tx, ty = _bilinear_setup(
            x_m, y_m, self.origin_m, self.spacing_m, self.heights_m.shape
        )
        h = self.heights_m
        dhdx = (
            (h[iy, ix + 1] - h[iy, ix]) * (1 - ty)
            + (h[iy + 1, ix + 1] - h[iy + 1, ix]) * ty
        ) / self.spacing_m
        dhdy = (
            (h[iy + 1, ix] - h[iy, ix]) * (1 - tx)
            + (h[iy + 1, ix + 1] - h[iy, ix + 1]) * tx
        ) / self.spacing_m
        return float(dhdx), float(dhdy)


@dataclass(frozen=True)
class FrictionField:
    """Spatial friction multipliers on the same grid convention.

    Multipliers scale the base :class:`~.friction.FrictionParams`
    coefficients locally; 1.0 everywhere reproduces a uniform green.

    Attributes:
        roll_multiplier: Rolling-resistance multipliers, ``[iy, ix]``.
        slide_multiplier: Sliding/static-friction multipliers.
        spacing_m: Grid spacing [m].
        origin_m: World coordinates of node ``[0, 0]`` (x, y) [m].
    """

    roll_multiplier: np.ndarray
    slide_multiplier: np.ndarray
    spacing_m: float = _DEFAULT_SPACING_M
    origin_m: tuple[float, float] = (-_DEFAULT_EXTENT_M / 2.0, -_DEFAULT_EXTENT_M / 2.0)

    def __post_init__(self) -> None:
        _validate_grid(self.roll_multiplier, self.spacing_m, "roll_multiplier")
        _validate_grid(self.slide_multiplier, self.spacing_m, "slide_multiplier")
        _validate_origin(self.origin_m)
        require(
            self.roll_multiplier.shape == self.slide_multiplier.shape,
            "multiplier grids must share a shape",
            (self.roll_multiplier.shape, self.slide_multiplier.shape),
        )
        require(
            bool(np.all(self.roll_multiplier > 0.0)),
            "roll multipliers must be positive",
        )
        require(
            bool(np.all(self.slide_multiplier > 0.0)),
            "slide multipliers must be positive",
        )
        roll = np.array(self.roll_multiplier, dtype=float, copy=True)
        slide = np.array(self.slide_multiplier, dtype=float, copy=True)
        roll.setflags(write=False)
        slide.setflags(write=False)
        object.__setattr__(self, "roll_multiplier", roll)
        object.__setattr__(self, "slide_multiplier", slide)

    @staticmethod
    def uniform(
        extent_m: float = _DEFAULT_EXTENT_M,
        spacing_m: float = _DEFAULT_SPACING_M,
    ) -> FrictionField:
        """Unit multipliers everywhere (homogeneous green)."""
        n = _validate_factory_grid(extent_m, spacing_m)
        ones = np.ones((n, n))
        return FrictionField(
            roll_multiplier=ones,
            slide_multiplier=ones.copy(),
            spacing_m=spacing_m,
            origin_m=(-extent_m / 2.0, -extent_m / 2.0),
        )

    def multipliers(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Bilinear ``(roll, slide)`` multipliers at a point."""
        require_finite(x_m, "x_m")
        require_finite(y_m, "y_m")
        ix, iy, tx, ty = _bilinear_setup(
            x_m, y_m, self.origin_m, self.spacing_m, self.roll_multiplier.shape
        )
        out = []
        for grid in (self.roll_multiplier, self.slide_multiplier):
            out.append(
                float(
                    grid[iy, ix] * (1 - tx) * (1 - ty)
                    + grid[iy, ix + 1] * tx * (1 - ty)
                    + grid[iy + 1, ix] * (1 - tx) * ty
                    + grid[iy + 1, ix + 1] * tx * ty
                )
            )
        ensure(out[0] > 0.0 and out[1] > 0.0, "multipliers stay positive", out)
        return out[0], out[1]


def _keyed_rng(seed: int, field_key: str) -> np.random.Generator:
    """Per-field generator keyed by ``[seed, crc32(field_key)]``.

    Reuses the Tools variation-engine seeding discipline
    (``swing_sim.variation.engine._stream_for``): independent streams
    per field so one field's draws never shift another's.
    """
    require(seed >= 0, "seed must be >= 0", seed)
    return np.random.default_rng([seed, zlib.crc32(field_key.encode())])


def _smoothed_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    correlation_cells: float,
) -> np.ndarray:
    """Unit-variance smoothed white noise (box-blur Gaussian approx)."""
    noise = rng.standard_normal(shape)
    radius = max(int(round(correlation_cells / 2.0)), 0)
    if radius == 0:
        return noise
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    for _ in range(_BLUR_PASSES):
        padded = np.pad(noise, radius, mode="reflect")
        noise = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"), 1, padded
        )[radius:-radius, :]
        padded = np.pad(noise, radius, mode="reflect")
        noise = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="valid"), 0, padded
        )[:, radius:-radius]
    std = float(noise.std())
    ensure(std > 0.0, "smoothed noise must retain variance", std)
    return noise / std


def bumpy_height_field(
    seed: int,
    amplitude_m: float,
    correlation_length_m: float,
    base: HeightField | None = None,
) -> HeightField:
    """Seeded stochastic height perturbation of a base field.

    Deterministic: the same ``(seed, amplitude, correlation, base)``
    always produces the identical field (test-pinned), and zero
    amplitude returns the base heights exactly.

    Args:
        seed: Master seed (>= 0); the height stream is keyed
            independently of the friction stream.
        amplitude_m: Standard deviation of the height bumps [m];
            realistic greens sit well under 0.02.
        correlation_length_m: Bump correlation length [m].
        base: Field to perturb; a default flat field when omitted.

    Returns:
        A new :class:`HeightField` with bumps added.

    Raises:
        ValueError: If inputs are out of range.
    """
    require(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an int")
    require(seed >= 0, "seed must be >= 0", seed)
    require_finite(amplitude_m, "amplitude_m")
    require(0.0 <= amplitude_m <= 0.05, "amplitude in [0, 0.05] m", amplitude_m)
    require_finite(correlation_length_m, "correlation_length_m")
    require(
        0.0 < correlation_length_m <= 20.0,
        "correlation length in (0, 20] m",
        correlation_length_m,
    )
    field = base if base is not None else HeightField.flat()
    if amplitude_m == 0.0:
        return field
    rng = _keyed_rng(seed, "putting_dynamics.height")
    bumps = amplitude_m * _smoothed_noise(
        rng, field.heights_m.shape, correlation_length_m / field.spacing_m
    )
    return HeightField(
        heights_m=field.heights_m + bumps,
        spacing_m=field.spacing_m,
        origin_m=field.origin_m,
    )


def bumpy_friction_field(
    seed: int,
    amplitude: float,
    correlation_length_m: float,
    base: FrictionField | None = None,
) -> FrictionField:
    """Seeded fractional friction-multiplier perturbation.

    Both multiplier grids get independent keyed streams (roll vs
    slide), each ``base * (1 + amplitude * smoothed_noise)`` clipped
    to stay positive.  Zero amplitude returns the base field exactly.

    Args:
        seed: Master seed (>= 0).
        amplitude: Fractional deviation (std) in [0, 0.5].
        correlation_length_m: Patch correlation length [m].
        base: Field to perturb; a default uniform field when omitted.

    Returns:
        A new :class:`FrictionField`.

    Raises:
        ValueError: If inputs are out of range.
    """
    require(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an int")
    require(seed >= 0, "seed must be >= 0", seed)
    require_finite(amplitude, "amplitude")
    require(0.0 <= amplitude <= 0.5, "amplitude in [0, 0.5]", amplitude)
    require_finite(correlation_length_m, "correlation_length_m")
    require(
        0.0 < correlation_length_m <= 20.0,
        "correlation length in (0, 20] m",
        correlation_length_m,
    )
    field = base if base is not None else FrictionField.uniform()
    if amplitude == 0.0:
        return field
    cells = correlation_length_m / field.spacing_m
    grids = []
    for key, grid in (
        ("putting_dynamics.friction.roll", field.roll_multiplier),
        ("putting_dynamics.friction.slide", field.slide_multiplier),
    ):
        noise = _smoothed_noise(_keyed_rng(seed, key), grid.shape, cells)
        grids.append(np.clip(grid * (1.0 + amplitude * noise), 0.05, None))
    return FrictionField(
        roll_multiplier=grids[0],
        slide_multiplier=grids[1],
        spacing_m=field.spacing_m,
        origin_m=field.origin_m,
    )


@dataclass(frozen=True)
class SurfaceSpec:
    """Complete green description consumed by the solver.

    Attributes:
        height: Elevation field.
        friction_field: Spatial friction multipliers.
        friction: Base friction-law parameters.
    """

    height: HeightField
    friction_field: FrictionField
    friction: FrictionParams

    @staticmethod
    def flat_uniform(
        stimp_ft: float = 10.0,
        friction: FrictionParams | None = None,
        extent_m: float = _DEFAULT_EXTENT_M,
    ) -> SurfaceSpec:
        """Flat, homogeneous green — the analytic-limit surface.

        Args:
            stimp_ft: Green speed used when ``friction`` is omitted.
            friction: Full parameter override.
            extent_m: Side length of the square grid [m].

        Returns:
            A :class:`SurfaceSpec` on which the solver must reproduce
            the Tools closed-form skid/roll results (test-enforced).
        """
        from .friction import stimp_to_rolling_mu

        params = friction or FrictionParams(mu_roll0=stimp_to_rolling_mu(stimp_ft))
        return SurfaceSpec(
            height=HeightField.flat(extent_m=extent_m),
            friction_field=FrictionField.uniform(extent_m=extent_m),
            friction=params,
        )
