"""Named design spaces and low-discrepancy sampling.

A :class:`DesignSpace` is an ordered set of named, bounded, unit-carrying
parameters. It is the single authority for the mapping between the physical
design vector a solver consumes and the unit hypercube every sampler,
sensitivity estimator and surrogate works in.

Sampling goes through :mod:`scipy.stats.qmc` (no new dependency, per
ADR-0032). The one place we are stricter than SciPy: a Sobol' sequence only
has its balance property at sizes that are a power of two, and SciPy merely
*warns* when asked for another size. Silently losing low discrepancy is
exactly the kind of failure a design study cannot detect downstream, so we
raise instead.

The same reasoning gives :meth:`DesignSpace.check_wedge_camber` its one
domain-aware duty (issue #8698): a box that sweeps sole width or bounce can
leave the band of camber areas a sole can physically realise, and the sweep
would then answer a different question than the one asked. The wedge
knowledge itself lives in :mod:`bunkershot3d.geometry.design_bounds` and is
imported on call, so this module stays independent of it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.stats import qmc

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bunkershot3d.geometry.wedge import WedgeGeometry

from src.shared.python.core.contracts import require

from .manifest import StudyManifest
from .rng import SeedRecord, new_seed_record

__all__ = [
    "DesignParameter",
    "DesignSample",
    "DesignSpace",
    "SamplerKind",
    "is_power_of_two",
    "make_qmc_engine",
]

#: Supported low-discrepancy samplers.
#:
#: ``"lhs_oa"`` is orthogonal-array Latin hypercube sampling
#: (``scipy.stats.qmc.LatinHypercube(strength=2)``): it guarantees balance on
#: every two-dimensional projection, at the cost of requiring the sample size
#: to be the square of a prime.
SamplerKind = Literal["sobol", "lhs", "lhs_oa", "halton", "random"]

_SAMPLER_KINDS: tuple[str, ...] = ("sobol", "lhs", "lhs_oa", "halton", "random")

#: SciPy renamed the QMC engines' ``seed`` argument to ``rng`` in 1.15; the
#: declared floor for this repo is 1.13.1, so resolve the name once at import.
_QMC_RNG_KEYWORD = (
    "rng" if "rng" in inspect.signature(qmc.Sobol.__init__).parameters else "seed"
)


def _qmc_rng_kwargs(generator: np.random.Generator) -> dict[str, Any]:
    """Build the version-appropriate seeding keyword for a QMC engine.

    Args:
        generator: The seeded generator driving the engine's scramble.

    Returns:
        A single-entry mapping, ``{"rng": generator}`` or ``{"seed": ...}``.
    """
    return {_QMC_RNG_KEYWORD: generator}


def make_qmc_engine(
    method: str,
    dimension: int,
    generator: np.random.Generator,
) -> qmc.QMCEngine:
    """Construct a seeded SciPy QMC engine.

    Shared by :meth:`DesignSpace.sample` and the Saltelli sampler, which needs
    a ``2 d``-dimensional Sobol' sequence rather than a ``d``-dimensional one.

    Args:
        method: One of ``"sobol"``, ``"halton"``, ``"lhs"`` or ``"lhs_oa"``.
        dimension: Engine dimension.
        generator: Seeded generator driving the scramble.

    Returns:
        The constructed engine.

    Raises:
        ValueError: If ``method`` has no QMC engine.
    """
    seeding = _qmc_rng_kwargs(generator)
    if method == "sobol":
        return qmc.Sobol(d=dimension, scramble=True, **seeding)
    if method == "halton":
        return qmc.Halton(d=dimension, scramble=True, **seeding)
    if method == "lhs_oa":
        return qmc.LatinHypercube(d=dimension, strength=2, **seeding)
    if method == "lhs":
        return qmc.LatinHypercube(d=dimension, **seeding)
    raise ValueError(f"no QMC engine for method {method!r}")


def is_power_of_two(value: int) -> bool:
    """Report whether ``value`` is a positive power of two.

    Args:
        value: Candidate sample size.

    Returns:
        ``True`` when ``value`` is 1, 2, 4, 8, ...
    """
    return value > 0 and value & (value - 1) == 0


def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``.

    Args:
        value: Candidate sample size.

    Returns:
        The next power of two, at least 1.
    """
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _is_prime(value: int) -> bool:
    """Report whether ``value`` is prime by trial division.

    Args:
        value: Candidate integer.

    Returns:
        ``True`` when ``value`` is prime.
    """
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    factor = 3
    while factor * factor <= value:
        if value % factor == 0:
            return False
        factor += 2
    return True


@dataclass(frozen=True, slots=True)
class DesignParameter:
    """One named, bounded design variable.

    Attributes:
        name: Identifier, unique within a :class:`DesignSpace`.
        lower: Inclusive lower bound, in :attr:`units`.
        upper: Inclusive upper bound, in :attr:`units`.
        units: SI-style unit suffix (``"mm"``, ``"deg"``, ``"-"``...).
    """

    name: str
    lower: float
    upper: float
    units: str = "-"

    def __post_init__(self) -> None:
        """Validate the parameter.

        Raises:
            ValueError: If the name is empty, a bound is non-finite, or the
                bounds are not strictly increasing.
        """
        if not self.name:
            raise ValueError("parameter name must be non-empty")
        lower = float(self.lower)
        upper = float(self.upper)
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ValueError(
                f"parameter {self.name!r} has non-finite bounds "
                f"[{self.lower}, {self.upper}]"
            )
        if not upper > lower:
            raise ValueError(
                f"parameter {self.name!r} needs upper > lower, got [{lower}, {upper}]"
            )

    @property
    def span(self) -> float:
        """Width of the parameter range.

        Returns:
            ``upper - lower``.
        """
        return float(self.upper) - float(self.lower)


@dataclass(frozen=True, eq=False)
class DesignSample:
    """A batch of design points, stored structure-of-arrays.

    Attributes:
        space: The space the points were drawn from.
        unit_cube: ``(n, d)`` sample in :math:`[0, 1]^d`.
        values: ``(n, d)`` sample in physical units.
        manifest: Provenance, including the entropy that produced it.
    """

    space: DesignSpace
    unit_cube: np.ndarray
    values: np.ndarray
    manifest: StudyManifest

    @property
    def n_samples(self) -> int:
        """Number of design points.

        Returns:
            The row count of :attr:`values`.
        """
        return int(self.values.shape[0])

    def column(self, name: str) -> np.ndarray:
        """Extract one physical parameter column by name.

        Args:
            name: Parameter name.

        Returns:
            A ``(n,)`` view of that parameter's sampled values.

        Raises:
            KeyError: If ``name`` is not a parameter of the space.
        """
        return self.values[:, self.space.index_of(name)]


@dataclass(frozen=True, eq=False)
class DesignSpace:
    """An ordered collection of named design parameters.

    Attributes:
        parameters: The parameters, in column order.
    """

    parameters: tuple[DesignParameter, ...]

    def __post_init__(self) -> None:
        """Validate the space.

        Raises:
            ValueError: If the space is empty or names are not unique.
        """
        if not self.parameters:
            raise ValueError("a design space needs at least one parameter")
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate parameter names: {duplicates}")

    @classmethod
    def from_bounds(
        cls,
        bounds: dict[str, tuple[float, float]],
        units: dict[str, str] | None = None,
    ) -> DesignSpace:
        """Build a space from a ``{name: (lower, upper)}`` mapping.

        Args:
            bounds: Parameter bounds, keyed by name. Insertion order becomes
                column order.
            units: Optional ``{name: unit}`` mapping; missing entries get
                ``"-"``.

        Returns:
            The constructed design space.
        """
        unit_map = units or {}
        return cls(
            tuple(
                DesignParameter(
                    name=name,
                    lower=low,
                    upper=high,
                    units=unit_map.get(name, "-"),
                )
                for name, (low, high) in bounds.items()
            )
        )

    @property
    def dimension(self) -> int:
        """Number of parameters.

        Returns:
            The design-space dimension ``d``.
        """
        return len(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names in column order.

        Returns:
            A tuple of names.
        """
        return tuple(p.name for p in self.parameters)

    @property
    def units(self) -> tuple[str, ...]:
        """Parameter units in column order.

        Returns:
            A tuple of unit strings.
        """
        return tuple(p.units for p in self.parameters)

    @property
    def lower(self) -> np.ndarray:
        """Lower bounds as a ``(d,)`` array.

        Returns:
            The lower bound of each parameter.
        """
        return np.array([p.lower for p in self.parameters], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        """Upper bounds as a ``(d,)`` array.

        Returns:
            The upper bound of each parameter.
        """
        return np.array([p.upper for p in self.parameters], dtype=float)

    @property
    def span(self) -> np.ndarray:
        """Parameter ranges as a ``(d,)`` array.

        Returns:
            ``upper - lower`` per parameter.
        """
        return self.upper - self.lower

    def check_wedge_camber(
        self, geometry: WedgeGeometry, *, n_points: int = 48
    ) -> tuple[str, ...]:
        """Screen this space against the constructible sole-camber band.

        A sweep over sole width or bounce moves the camber areas a sole can
        physically realise, so corners of the box can fall outside the band
        and be lofted as the nearest constructible section instead. The sweep
        still completes, but a sensitivity estimator then attributes variance
        to a camber the user believes is pinned (issue #8698). This is the
        cheap up-front check that catches it; it costs a handful of profile
        solves rather than a full design of experiments.

        The wedge knowledge lives in
        :func:`bunkershot3d.geometry.design_bounds.check_camber_design_space`
        and is imported here on call, so this module keeps working - and
        keeps being testable against analytic functions - without it.

        Args:
            geometry: The :class:`~bunkershot3d.geometry.wedge.WedgeGeometry`
                the study pins everything outside this space from.
            n_points: Sole samples used to evaluate the band; pass the value
                the sweep will loft at.

        Returns:
            One finding per offending corner, empty when the space is clean.
            A space with no camber, sole-width or bounce parameter is not
            screened and returns an empty tuple.

        Raises:
            TypeError: If ``geometry`` is not a ``WedgeGeometry``.
        """
        from bunkershot3d.geometry.design_bounds import check_camber_design_space

        return check_camber_design_space(self, geometry, n_points=n_points)

    def index_of(self, name: str) -> int:
        """Return the column index of a named parameter.

        Args:
            name: Parameter name.

        Returns:
            The zero-based column index.

        Raises:
            KeyError: If ``name`` is not in the space.
        """
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"unknown parameter {name!r}; space has {list(self.names)}"
            ) from exc

    def to_physical(self, unit_cube: np.ndarray) -> np.ndarray:
        """Map unit-cube coordinates to physical values.

        Args:
            unit_cube: ``(n, d)`` or ``(d,)`` array in :math:`[0, 1]^d`.

        Returns:
            An array of the same shape in physical units.

        Raises:
            ValueError: If the shape is wrong or a coordinate is outside
                :math:`[0, 1]`.
        """
        points = self._as_matrix(unit_cube, "unit_cube")
        if not np.all(np.isfinite(points)):
            raise ValueError("unit_cube contains non-finite values")
        if points.size and (points.min() < 0.0 or points.max() > 1.0):
            raise ValueError(
                "unit_cube values must lie in [0, 1]; got "
                f"[{points.min()}, {points.max()}]"
            )
        scaled = self.lower + points * self.span
        return scaled.reshape(np.shape(unit_cube))

    def to_unit_cube(self, values: np.ndarray) -> np.ndarray:
        """Map physical values to unit-cube coordinates.

        Args:
            values: ``(n, d)`` or ``(d,)`` array in physical units.

        Returns:
            An array of the same shape in :math:`[0, 1]^d`.

        Raises:
            ValueError: If the shape is wrong or a value is out of bounds.
        """
        points = self._as_matrix(values, "values")
        if not np.all(np.isfinite(points)):
            raise ValueError("values contains non-finite values")
        below = points < self.lower - 1e-12 * np.maximum(1.0, np.abs(self.lower))
        above = points > self.upper + 1e-12 * np.maximum(1.0, np.abs(self.upper))
        if np.any(below | above):
            bad = int(np.argmax(np.any(below | above, axis=1)))
            raise ValueError(
                f"design point {bad} is outside the space bounds: "
                f"{points[bad].tolist()} not within "
                f"{self.lower.tolist()}..{self.upper.tolist()}"
            )
        unit = (points - self.lower) / self.span
        return np.clip(unit, 0.0, 1.0).reshape(np.shape(values))

    def contains(self, values: np.ndarray) -> np.ndarray:
        """Test which physical points lie inside the bounds.

        Args:
            values: ``(n, d)`` or ``(d,)`` array in physical units.

        Returns:
            A boolean array of shape ``(n,)`` (or a scalar-shaped ``()``
            array for a single point).
        """
        points = self._as_matrix(values, "values")
        inside = np.all(
            (points >= self.lower) & (points <= self.upper) & np.isfinite(points),
            axis=1,
        )
        return inside[0] if np.ndim(values) == 1 else inside

    def sample(
        self,
        n_samples: int,
        method: SamplerKind = "sobol",
        seed: int | SeedRecord | None = None,
    ) -> DesignSample:
        """Draw a low-discrepancy sample of the space.

        Args:
            n_samples: Number of design points. Sobol' requires a power of
                two; ``"lhs_oa"`` requires the square of a prime.
            method: One of ``"sobol"``, ``"lhs"``, ``"lhs_oa"``, ``"halton"``
                or ``"random"``.
            seed: Explicit entropy or seed record; ``None`` draws fresh
                entropy from :func:`secrets.randbits`.

        Returns:
            A :class:`DesignSample` carrying both representations and the
            manifest needed to replay it.

        Raises:
            ValueError: If ``method`` is unknown or ``n_samples`` is invalid
                for the chosen sampler.
        """
        if method not in _SAMPLER_KINDS:
            raise ValueError(
                f"unknown sampler {method!r}; expected one of {list(_SAMPLER_KINDS)}"
            )
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        self._validate_sample_size(n_samples, method)

        record = new_seed_record(seed)
        generator = record.generator()
        unit_cube = self._draw_unit_cube(n_samples, method, generator)

        require(
            unit_cube.shape == (n_samples, self.dimension),
            "sampler returned the wrong shape",
            value=unit_cube.shape,
        )
        manifest = StudyManifest(
            seed=record,
            method=method,
            parameter_names=self.names,
            n_samples=n_samples,
        )
        return DesignSample(
            space=self,
            unit_cube=unit_cube,
            values=self.to_physical(unit_cube),
            manifest=manifest,
        )

    def _draw_unit_cube(
        self,
        n_samples: int,
        method: str,
        generator: np.random.Generator,
    ) -> np.ndarray:
        """Dispatch to the SciPy QMC engine for ``method``.

        Args:
            n_samples: Number of points to draw.
            method: Validated sampler name.
            generator: Seeded NumPy generator driving the scramble.

        Returns:
            An ``(n, d)`` array in the unit hypercube.
        """
        if method == "random":
            return generator.random((n_samples, self.dimension))
        engine = make_qmc_engine(method, self.dimension, generator)
        return np.asarray(engine.random(n_samples), dtype=float)

    def _validate_sample_size(self, n_samples: int, method: str) -> None:
        """Reject sample sizes that would silently degrade the sampler.

        Args:
            n_samples: Requested number of points.
            method: Validated sampler name.

        Raises:
            ValueError: If the size is incompatible with the sampler.
        """
        if method == "sobol" and not is_power_of_two(n_samples):
            raise ValueError(
                f"Sobol' sampling requires a power-of-two sample size to keep "
                f"its balance property; got {n_samples}. Use "
                f"{_next_power_of_two(n_samples)} (or "
                f"{_next_power_of_two(n_samples) // 2}), or pick "
                f"method='halton' / 'lhs' if the size is fixed."
            )
        if method != "lhs_oa":
            return
        root = int(round(n_samples**0.5))
        if root * root != n_samples or not _is_prime(root):
            raise ValueError(
                "orthogonal-array LHS (strength=2) requires the sample size to "
                f"be the square of a prime; got {n_samples}. Valid nearby sizes: "
                f"{_nearby_prime_squares(n_samples)}."
            )
        if self.dimension > root + 1:
            raise ValueError(
                f"orthogonal-array LHS with n={n_samples} (p={root}) supports at "
                f"most {root + 1} parameters; this space has {self.dimension}."
            )

    def _as_matrix(self, array: np.ndarray, label: str) -> np.ndarray:
        """Coerce a design array to a validated ``(n, d)`` matrix.

        Args:
            array: ``(n, d)`` or ``(d,)`` input.
            label: Name used in error messages.

        Returns:
            A two-dimensional float array with ``d`` columns.

        Raises:
            ValueError: If the array is not 1- or 2-D, or has the wrong
                number of columns.
        """
        points = np.atleast_2d(np.asarray(array, dtype=float))
        if points.ndim != 2:
            raise ValueError(f"{label} must be 1- or 2-dimensional, got {points.ndim}D")
        if points.shape[1] != self.dimension:
            raise ValueError(
                f"{label} has {points.shape[1]} columns but the space has "
                f"{self.dimension} parameters {list(self.names)}"
            )
        return points


def _nearby_prime_squares(n_samples: int, count: int = 3) -> list[int]:
    """List prime squares bracketing ``n_samples``, for error messages.

    Args:
        n_samples: The rejected sample size.
        count: How many candidates to return.

    Returns:
        A sorted list of valid orthogonal-array LHS sample sizes.
    """
    squares = [p * p for p in range(2, 200) if _is_prime(p)]
    squares.sort(key=lambda s: (abs(s - n_samples), s))
    return sorted(squares[:count])
