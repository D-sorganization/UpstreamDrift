"""A bounded coefficient domain for canonical C1 cubic Hermite trajectories.

Bounding all four segment Bernstein controls is a sufficient, conservative
whole-segment coordinate bound. This module changes coordinates, not solvers.
It does not certify nonlinear closure, ground contact, or physical motion.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np

CoordinateBound = tuple[float, float] | None


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_bounds(
    bounds: tuple[CoordinateBound, ...],
) -> tuple[CoordinateBound, ...]:
    if not bounds:
        raise ValueError("coordinate_bounds must contain at least one coordinate")
    result: list[CoordinateBound] = []
    for bound in bounds:
        if bound is None:
            result.append(None)
            continue
        if len(bound) != 2:
            raise ValueError("each coordinate bound must have two finite endpoints")
        lower = _finite_number(bound[0], "lower bound")
        upper = _finite_number(bound[1], "upper bound")
        if lower > upper:
            raise ValueError("lower bound must not exceed upper bound")
        if not np.isfinite(upper - lower):
            raise ValueError("coordinate bound width must be representable")
        result.append((lower, upper))
    return tuple(result)


def _active_extreme(
    candidates: list[tuple[float, float]], *, maximum: bool
) -> tuple[float, float]:
    """Value and mean slope of exactly tied active branches.

    At a tie the slope is a symmetric generalized derivative, not a classical
    derivative. No proximity tolerance changes the represented domain.
    """
    extreme = (max if maximum else min)(value for value, _ in candidates)
    slopes = [slope for value, slope in candidates if value == extreme]
    return extreme, float(sum(slopes) / len(slopes))


@dataclass(frozen=True)
class HermiteBoundsDomain:
    """Immutable physical-to-box decision adapter, without output clipping.

    Physical packing is all knot positions, then all knot velocities, both
    knot-major/coordinate-minor, matching CubicHermiteSplineTrajectory.
    Decisions keep that order with fixed coordinates removed. Bounded velocity
    decisions are normalized slopes in [-1, 1]; unbounded ones are physical.
    ``None`` means unbounded; finite equal endpoints mean q=L and v=0.
    Arrays/iterables supplied at construction are copied to immutable tuples.
    Units and coordinate identity remain the caller's explicit responsibility.
    """

    knot_times: tuple[float, ...]
    coordinate_bounds: tuple[CoordinateBound, ...]

    def __post_init__(self) -> None:
        times = tuple(_finite_number(value, "knot time") for value in self.knot_times)
        if len(times) < 2 or any(
            right <= left for left, right in zip(times, times[1:], strict=False)
        ):
            raise ValueError("knot_times must contain at least two increasing times")
        bounds = _normalize_bounds(self.coordinate_bounds)
        for left, right in zip(times, times[1:], strict=False):
            duration = right - left
            if not np.isfinite(duration) or not np.isfinite(3.0 / duration):
                raise ValueError(
                    "segment duration and reciprocal must be representable"
                )
            for bound in bounds:
                if bound is not None and not np.isfinite(
                    (bound[1] - bound[0]) * (3.0 / duration)
                ):
                    raise ValueError(
                        "coordinate velocity interval width must be representable"
                    )
        object.__setattr__(self, "knot_times", times)
        object.__setattr__(self, "coordinate_bounds", bounds)

    @property
    def n_knots(self) -> int:
        """Number of strictly increasing knots."""
        return len(self.knot_times)

    @property
    def n_dof(self) -> int:
        """Number of physical coordinates, including fixed coordinates."""
        return len(self.coordinate_bounds)

    @property
    def physical_size(self) -> int:
        """Count of ordinary q/v coefficients."""
        return 2 * self.n_knots * self.n_dof

    @property
    def _active_coordinates(self) -> tuple[int, ...]:
        return tuple(
            i
            for i, bound in enumerate(self.coordinate_bounds)
            if bound is None or bound[0] < bound[1]
        )

    @property
    def decision_size(self) -> int:
        """Count of independent box-constrained decisions; may be zero."""
        return 2 * self.n_knots * len(self._active_coordinates)

    def decision_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Fresh lower/upper box vectors, with no equal-endpoint columns."""
        positions = [
            self.coordinate_bounds[i] or (-np.inf, np.inf)
            for _ in self.knot_times
            for i in self._active_coordinates
        ]
        velocities = [
            (-1.0, 1.0) if self.coordinate_bounds[i] is not None else (-np.inf, np.inf)
            for _ in self.knot_times
            for i in self._active_coordinates
        ]
        entries = positions + velocities
        return (
            np.array([pair[0] for pair in entries], dtype=float),
            np.array([pair[1] for pair in entries], dtype=float),
        )

    def _vector(self, values: np.ndarray, size: int, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite vector of size {size}")
        return vector

    def _decision(self, decision: np.ndarray) -> np.ndarray:
        vector = self._vector(decision, self.decision_size, "decision")
        lower, upper = self.decision_bounds()
        if np.any(vector < lower) or np.any(vector > upper):
            raise ValueError(
                "decision exceeds its position or normalized velocity bounds"
            )
        return vector

    def _velocity_limits(
        self, q: float, knot: int, coordinate: int
    ) -> tuple[float, float, float, float]:
        bound = self.coordinate_bounds[coordinate]
        if bound is None:
            raise ValueError("velocity limits require a bounded coordinate")
        lower, upper = bound
        lows: list[tuple[float, float]] = []
        highs: list[tuple[float, float]] = []
        if knot < self.n_knots - 1:
            factor = 3.0 / (self.knot_times[knot + 1] - self.knot_times[knot])
            lows.append((factor * (lower - q), -factor))
            highs.append((factor * (upper - q), -factor))
        if knot > 0:
            factor = 3.0 / (self.knot_times[knot] - self.knot_times[knot - 1])
            lows.append((factor * (q - upper), factor))
            highs.append((factor * (q - lower), factor))
        lo, d_lo = _active_extreme(lows, maximum=True)
        hi, d_hi = _active_extreme(highs, maximum=False)
        return lo, hi, d_lo, d_hi

    def encode(self, coefficients: np.ndarray) -> np.ndarray:
        """Encode a feasible physical candidate; reject instead of repairing it.

        Collapsed interior velocity intervals encode zero with normalized slope
        zero. That one point is non-injective: other normalized slopes decode
        to the same physical zero velocity. Fixed coordinates require exact
        q=L, v=0. No implicit feasible initialization occurs here.
        """
        physical = self._vector(coefficients, self.physical_size, "coefficients")
        q, v = physical.reshape(2, self.n_knots, self.n_dof)
        active = self._active_coordinates
        normalized = np.empty((self.n_knots, len(active)))
        for coordinate, bound in enumerate(self.coordinate_bounds):
            if bound is None:
                continue
            if np.any(q[:, coordinate] < bound[0]) or np.any(
                q[:, coordinate] > bound[1]
            ):
                raise ValueError("physical position exceeds coordinate bounds")
            if bound[0] == bound[1] and np.any(v[:, coordinate] != 0.0):
                raise ValueError("fixed coordinates require zero physical velocity")
        for column, coordinate in enumerate(active):
            for knot in range(self.n_knots):
                velocity = v[knot, coordinate]
                if self.coordinate_bounds[coordinate] is None:
                    normalized[knot, column] = velocity
                    continue
                lo, hi, _, _ = self._velocity_limits(
                    q[knot, coordinate], knot, coordinate
                )
                if velocity < lo or velocity > hi:
                    raise ValueError(
                        "physical velocity violates Bernstein control bounds"
                    )
                normalized[knot, column] = (
                    0.0 if hi == lo else 2.0 * ((velocity - lo) / (hi - lo)) - 1.0
                )
        return np.concatenate((q[:, active].ravel(), normalized.ravel()))

    def decode(self, decision: np.ndarray) -> np.ndarray:
        """Decode box decisions to canonical physical coefficients."""
        x = self._decision(decision)
        active = self._active_coordinates
        split = self.decision_size // 2
        q = np.zeros((self.n_knots, self.n_dof))
        v = np.zeros_like(q)
        for coordinate, bound in enumerate(self.coordinate_bounds):
            if bound is not None and bound[0] == bound[1]:
                q[:, coordinate] = bound[0]
        for column, coordinate in enumerate(active):
            for knot in range(self.n_knots):
                index = knot * len(active) + column
                q[knot, coordinate] = x[index]
                if self.coordinate_bounds[coordinate] is None:
                    v[knot, coordinate] = x[split + index]
                else:
                    lo, hi, _, _ = self._velocity_limits(x[index], knot, coordinate)
                    theta = (x[split + index] + 1.0) / 2.0
                    v[knot, coordinate] = (1.0 - theta) * lo + theta * hi
        return np.concatenate((q.ravel(), v.ravel()))

    def decode_jacobian(self, decision: np.ndarray) -> np.ndarray:
        """Physical-coefficient rows by internal decision columns.

        Exact ordinary derivatives apply away from active-branch ties. At ties
        this returns a symmetric generalized derivative (mean active slopes);
        one-sided derivatives can differ. Optimizer convergence is not implied.
        """
        x = self._decision(decision)
        active = self._active_coordinates
        split = self.decision_size // 2
        physical_split = self.physical_size // 2
        jacobian = np.zeros((self.physical_size, self.decision_size))
        for column, coordinate in enumerate(active):
            for knot in range(self.n_knots):
                index = knot * len(active) + column
                row = knot * self.n_dof + coordinate
                jacobian[row, index] = 1.0
                if self.coordinate_bounds[coordinate] is None:
                    jacobian[physical_split + row, split + index] = 1.0
                else:
                    lo, hi, d_lo, d_hi = self._velocity_limits(
                        x[index], knot, coordinate
                    )
                    theta = (x[split + index] + 1.0) / 2.0
                    jacobian[physical_split + row, index] = (
                        1.0 - theta
                    ) * d_lo + theta * d_hi
                    jacobian[physical_split + row, split + index] = (hi - lo) / 2.0
        return jacobian
