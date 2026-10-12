"""Cross-engine parity harness (#12328).

Engines are compared through ``EngineResult``, so a later engine adapter only
needs a fixture. The clubhead-speed tolerance is the existing
``CrossEngineValidator`` velocity tolerance; it is not chosen or loosened here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.shared.python.data_io.provenance import ProvenanceInfo
from src.shared.python.engine_core.cross_engine_validator import CrossEngineValidator

from ..adapters_base import CLUBHEAD_SPEED, EngineResult
from ..builder import build_swing_report
from ..contract import Capture, SwingReport

VELOCITY_TOLERANCE: float = CrossEngineValidator.TOLERANCES["velocity"]


@dataclass(frozen=True)
class ParityOutcome:
    """Pointwise clubhead-speed agreement between two engine results.

    ``comparable`` is False (and ``passed`` False) when either engine lacks the
    channel or the time grids differ; ``reason`` says why.
    """

    engines: tuple[str, str]
    comparable: bool
    passed: bool
    max_abs_diff: float | None
    n_compared: int
    tolerance: float
    reason: str | None = None


def _incomparable(pair: tuple[str, str], tol: float, reason: str) -> ParityOutcome:
    return ParityOutcome(pair, False, False, None, 0, tol, reason)


def compare_clubhead_speed(
    a: EngineResult, b: EngineResult, tolerance: float = VELOCITY_TOLERANCE
) -> ParityOutcome:
    """Compare two results' clubhead speed sample by sample.

    Precondition: ``tolerance`` > 0. Samples missing (``None``) in either engine
    are skipped, never treated as zero; if none remain the result is not
    comparable.
    """
    if not tolerance > 0.0:
        raise ValueError("tolerance must be positive")
    pair = (a.engine, b.engine)
    if not (a.healthy and b.healthy):
        return _incomparable(pair, tolerance, "engine unavailable")
    qa, qb = a.quantities.get(CLUBHEAD_SPEED), b.quantities.get(CLUBHEAD_SPEED)
    if qa is None or qb is None:
        return _incomparable(pair, tolerance, "clubhead_speed unavailable")
    if qa.time_s != qb.time_s or qa.units != qb.units:
        return _incomparable(pair, tolerance, "time grid or units differ")
    diffs = [
        abs(x - y)
        for x, y in zip(qa.values, qb.values, strict=True)
        if x is not None and y is not None
    ]
    if not diffs:
        return _incomparable(pair, tolerance, "no overlapping finite samples")
    worst = max(diffs)
    return ParityOutcome(pair, True, worst <= tolerance, worst, len(diffs), tolerance)


def build_parity_report(
    results: Sequence[EngineResult],
    capture: Capture,
    scope: str = "full",
    *,
    tolerance: float = VELOCITY_TOLERANCE,
    provenance: ProvenanceInfo | None = None,
) -> tuple[SwingReport, tuple[ParityOutcome, ...]]:
    """Build the validated report and every pairwise speed-parity outcome."""
    report = build_swing_report(results, capture, scope, provenance=provenance)
    outcomes = tuple(
        compare_clubhead_speed(results[i], results[j], tolerance)
        for i in range(len(results))
        for j in range(i + 1, len(results))
    )
    return report, outcomes
