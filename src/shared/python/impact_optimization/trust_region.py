"""``TrustRegion``: weighted deviation from the matched (safe) trajectory.

The optimiser (C3) searches around the matched swing; this contract defines
"close to the fit".  The norm is
``sqrt(sum_j w_j * mean_t (x[t, j] - ref[t, j])**2)``, so it is independent of
the sample count and each channel's weight is interpretable.

Design by Contract:
    Preconditions: finite 2-D reference (samples x channels), one positive
    weight per channel, positive finite radius, candidate of identical shape.
    Postconditions: deviation >= 0 and exactly 0 on the reference;
    ``penalty`` is 0 inside the region and grows quadratically outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import ensure, require


@dataclass(frozen=True, eq=False)
class TrustRegion:
    """Ball of ``radius`` (weighted norm) around ``reference``."""

    reference: np.ndarray
    weights: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        ref = np.array(self.reference, dtype=float, copy=True)
        w = np.array(self.weights, dtype=float, copy=True)
        require(ref.ndim == 2 and ref.size > 0, "reference must be 2-D, non-empty")
        require(bool(np.all(np.isfinite(ref))), "reference must be finite")
        require(
            w.shape == (ref.shape[1],),
            "need one weight per reference channel",
            w.shape,
        )
        require(
            bool(np.all(np.isfinite(w)) and np.all(w > 0.0)),
            "weights must be positive and finite",
        )
        require(
            math.isfinite(self.radius) and self.radius > 0.0,
            "radius must be positive and finite",
            self.radius,
        )
        ref.setflags(write=False)
        w.setflags(write=False)
        object.__setattr__(self, "reference", ref)
        object.__setattr__(self, "weights", w)

    def deviation(self, candidate: np.ndarray) -> float:
        """Weighted RMS deviation of ``candidate`` from the reference."""
        x = np.asarray(candidate, dtype=float)
        require(x.shape == self.reference.shape, "candidate shape mismatch", x.shape)
        d2 = np.mean((x - self.reference) ** 2, axis=0)
        value = float(np.sqrt(np.sum(self.weights * d2)))
        ensure(math.isfinite(value) and value >= 0.0, "deviation invalid", value)
        return value

    def slack(self, candidate: np.ndarray) -> float:
        """``radius - deviation``: positive inside, negative outside."""
        return self.radius - self.deviation(candidate)

    def contains(self, candidate: np.ndarray) -> bool:
        return self.slack(candidate) >= 0.0

    def penalty(self, candidate: np.ndarray) -> float:
        """Quadratic exterior penalty (0 inside the region)."""
        return max(0.0, -self.slack(candidate)) ** 2
