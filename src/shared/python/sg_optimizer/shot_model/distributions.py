"""Tilted bivariate Gaussian shot dispersion model.

Models the 2D landing offset (Δ‖ along-target, Δ⊥ lateral) relative to the
aim line as a correlated bivariate normal. The positive correlation captures
the clubface-driven coupling between starting line and dynamic loft — long
misses tend to be left, short misses tend to be right (for a RH golfer).

See ``docs/sg_optimizer/STROKES_GAINED_OPTIMIZER_SPEC.md`` §1.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.contracts import require

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

# Hard bound on ρ to keep Σ strictly positive-definite under finite precision.
_RHO_BOUND = 0.99


@dataclass(frozen=True)
class TiltedBivariateGaussian:
    """A tilted bivariate Gaussian over (Δ‖, Δ⊥) in yards.

    Convention: Δ‖ positive = past target. Δ⊥ positive = left of target.
    """

    sigma_long: float
    sigma_lat: float
    rho: float = 0.0
    bias_long: float = 0.0
    bias_lat: float = 0.0

    def __post_init__(self) -> None:
        require(self.sigma_long > 0, "sigma_long must be > 0", self.sigma_long)
        require(self.sigma_lat > 0, "sigma_lat must be > 0", self.sigma_lat)
        require(
            -_RHO_BOUND < self.rho < _RHO_BOUND,
            f"rho must lie in ({-_RHO_BOUND}, {_RHO_BOUND})",
            self.rho,
        )
        require(
            math.isfinite(self.bias_long), "bias_long must be finite", self.bias_long
        )
        require(math.isfinite(self.bias_lat), "bias_lat must be finite", self.bias_lat)

    def covariance_matrix(self) -> NDArray[np.float64]:
        """Return the 2×2 covariance Σ."""
        c = self.rho * self.sigma_long * self.sigma_lat
        return np.array(
            [[self.sigma_long**2, c], [c, self.sigma_lat**2]], dtype=np.float64
        )

    def mean(self) -> NDArray[np.float64]:
        return np.array([self.bias_long, self.bias_lat], dtype=np.float64)

    def tilt_angle_degrees(self) -> float:
        """Tilt angle of the principal axis vs. the along-target axis."""
        denom = self.sigma_long**2 - self.sigma_lat**2
        num = 2.0 * self.rho * self.sigma_long * self.sigma_lat
        return 0.5 * math.degrees(math.atan2(num, denom))

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw ``n`` samples; returns shape (n, 2) = (Δ‖, Δ⊥)."""
        require(n >= 0, "n must be non-negative", n)
        if n == 0:
            return np.empty((0, 2), dtype=np.float64)
        return rng.multivariate_normal(self.mean(), self.covariance_matrix(), size=n)

    def scaled(self, mult_long: float, mult_lat: float) -> TiltedBivariateGaussian:
        """Return a new distribution with σ scaled (correlation preserved)."""
        require(mult_long > 0, "mult_long must be > 0", mult_long)
        require(mult_lat > 0, "mult_lat must be > 0", mult_lat)
        return replace(
            self,
            sigma_long=self.sigma_long * mult_long,
            sigma_lat=self.sigma_lat * mult_lat,
        )

    def shifted(self, dlong: float, dlat: float) -> TiltedBivariateGaussian:
        """Return a new distribution with additive bias shift."""
        return replace(
            self, bias_long=self.bias_long + dlong, bias_lat=self.bias_lat + dlat
        )

    def confidence_ellipse(self, level: float = 0.95) -> dict[str, float]:
        """Parameters of the ``level``-confidence ellipse around the mean.

        Returns ``{"a", "b", "angle_deg", "cx", "cy"}`` where a, b are
        semi-axes (yards) and ``angle_deg`` is the tilt from the along-target
        axis.
        """
        require(0.0 < level < 1.0, "level must lie in (0, 1)", level)
        # 2-DOF χ² quantile.
        k = -2.0 * math.log(1.0 - level)
        eigvals, eigvecs = np.linalg.eigh(self.covariance_matrix())
        idx = int(np.argmax(eigvals))
        a = float(math.sqrt(k * eigvals[idx]))
        b = float(math.sqrt(k * eigvals[1 - idx]))
        vmax = eigvecs[:, idx]
        angle = math.degrees(math.atan2(float(vmax[1]), float(vmax[0])))
        return {
            "a": a,
            "b": b,
            "angle_deg": angle,
            "cx": float(self.bias_long),
            "cy": float(self.bias_lat),
        }
