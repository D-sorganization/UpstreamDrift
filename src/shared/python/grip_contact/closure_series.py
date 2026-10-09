"""Per-engine grip-closure residual series over a swing (OSV-2, #11728).

The residual is the translational gap (metres) between the two grip frames the
closure ties together.  Unavailable is never zero: an engine that cannot supply
the residual, or returns a non-finite one, yields a series whose ``residual_m``
is ``None`` with a reason, and ``max_m`` / ``within`` report ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require


@dataclass(frozen=True)
class GripClosureSeries:
    """Closure residual per frame for one engine, or why it is unavailable."""

    engine: str
    residual_m: np.ndarray | None
    reason: str | None = None

    def __post_init__(self) -> None:
        require(bool(self.engine), "engine name must be non-empty")
        if self.residual_m is None:
            require(bool(self.reason), "an unavailable series needs a reason")
        else:
            arr = np.asarray(self.residual_m, dtype=float)
            require(
                arr.ndim == 1 and arr.size > 0 and np.isfinite(arr).all(),
                "residual_m must be a non-empty finite 1-D series",
            )
            require((arr >= 0.0).all(), "residual norms must be non-negative")

    @classmethod
    def unavailable(cls, engine: str, reason: str) -> GripClosureSeries:
        return cls(engine, None, reason)

    @property
    def available(self) -> bool:
        return self.residual_m is not None

    @property
    def max_m(self) -> float | None:
        """Largest residual over the swing; ``None`` when unavailable."""
        if self.residual_m is None:
            return None
        return float(np.max(self.residual_m))

    def within(self, tolerance_m: float) -> bool | None:
        """True when every frame is within tolerance; ``None`` if unavailable."""
        require(
            math.isfinite(tolerance_m) and tolerance_m > 0.0,
            "tolerance_m must be finite and positive",
        )
        peak = self.max_m
        return None if peak is None else peak <= tolerance_m

    def as_document(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "available": self.available,
            "frames": None if self.residual_m is None else int(self.residual_m.size),
            "max_m": self.max_m,
            "mean_m": (
                None if self.residual_m is None else float(np.mean(self.residual_m))
            ),
            "reason": self.reason,
        }


def closure_series_from_residuals(
    engine: str,
    residuals: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
) -> GripClosureSeries:
    """Evaluate ``residuals(q_k)`` (first three entries, translation) per frame.

    ``NotImplementedError`` from the engine (closure not qualified) and a
    non-finite residual both give an unavailable series, never zeros.
    """
    frames = np.asarray(q, dtype=float)
    require(frames.ndim == 2 and frames.shape[0] > 0, "q must be a (frames, n) array")
    out = np.empty(frames.shape[0])
    for k, row in enumerate(frames):
        try:
            res = np.asarray(residuals(row), dtype=float).ravel()
        except NotImplementedError as exc:
            return GripClosureSeries.unavailable(engine, f"not supported: {exc}")
        if res.size < 3 or not np.isfinite(res[:3]).all():
            return GripClosureSeries.unavailable(
                engine, f"non-finite or short residual at frame {k}"
            )
        out[k] = float(np.linalg.norm(res[:3]))
    return GripClosureSeries(engine, out)
