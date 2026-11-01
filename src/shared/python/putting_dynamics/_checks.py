"""Scalar precondition fast path for the solver's inner loop.

``contracts.require_finite`` routes every value through ``numpy.isfinite``
and ``numpy.all``. The RK4 derivative calls it several times per stage, so
an aim solve spent most of its time there. This check has the same contract
for a single float, using ``math.isfinite``.
"""

from __future__ import annotations

import math

from src.shared.python.contracts import (
    ContractLevel,
    PreconditionError,
    get_contract_level,
)


def require_finite_scalar(value: float, name: str) -> None:
    """Require one scalar to be finite (no NaN / Inf).

    Skipped when the contract level is ``OFF``, like
    ``contracts.require_finite``.

    Raises:
        PreconditionError: If ``value`` is NaN or Inf.
    """
    if get_contract_level() is ContractLevel.OFF:
        return
    if not math.isfinite(value):
        raise PreconditionError(f"{name} contains NaN or Inf values")
