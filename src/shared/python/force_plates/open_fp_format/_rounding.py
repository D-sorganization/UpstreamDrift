"""Serialisation rounding added to a declared plate uncertainty (OF-02, OF-06).

A value rounded to the nearest multiple of ``q`` carries a uniform error of
half-width ``q / 2``, whose standard deviation is ``(q / 2) / sqrt(3)``.  Forms
that round (the CSV export, the LAN JSON) add that in quadrature to the
declared force floor and CoP sigma, so the rounding is declared, not hidden.
The NPZ form stores float64 and adds nothing.
"""

from __future__ import annotations

import dataclasses
import math

from src.shared.python.force_plates.contracts import PlateUncertainty
from src.shared.python.force_plates.open_fp_format import format as fmt

__all__ = ["widen_for_rounding"]

_SQRT3 = math.sqrt(3.0)


def widen_for_rounding(
    uncertainty: PlateUncertainty,
    *,
    force_quantum_n: float,
    cop_quantum_mm: float,
) -> PlateUncertainty:
    """``uncertainty`` with the rounding of a form added in quadrature.

    Force floor: ``hypot(force_floor_n, (q_F / 2) / sqrt(3))``; CoP:
    ``hypot(cop_m, (q_c / 2) / sqrt(3))`` with ``q_c`` converted from mm to m.
    Other fields are kept; ``source`` is amended.  Raises ``TypeError`` for a
    non-``PlateUncertainty`` and ``ValueError`` for a non-positive quantum.
    """
    if not isinstance(uncertainty, PlateUncertainty):
        raise TypeError("uncertainty must be a PlateUncertainty")
    for name, value in (
        ("force_quantum_n", force_quantum_n),
        ("cop_quantum_mm", cop_quantum_mm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    force_sigma = 0.5 * force_quantum_n / _SQRT3
    cop_sigma_m = 0.5 * cop_quantum_mm / fmt.MM_PER_M / _SQRT3
    return dataclasses.replace(
        uncertainty,
        cop_m=math.hypot(uncertainty.cop_m, cop_sigma_m),
        force_floor_n=math.hypot(uncertainty.force_floor_n, force_sigma),
        source=(
            f"{uncertainty.source} + rounding ({force_quantum_n:g} N, "
            f"{cop_quantum_mm:g} mm)"
        ),
    )
