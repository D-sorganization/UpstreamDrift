"""Lie modifier for dispersion-table builds, with common random numbers (#14610).

Wraps ``physics.lie_interaction.make_lie_modifier``.  A fresh generator is
derived from ``SeedSequence([seed, 14610, lie_index])`` on **every call**, so the
noise block depends only on (seed, lie): all skills share it, the output is
identical on repeat calls, and the table does not change with loop order.  Rough
severity is fixed (``ROUGH_SEVERITY``) and recorded in the table provenance.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.physics.lie_interaction import make_lie_modifier
from src.shared.python.sg_optimizer.shot_model.dispersion_table import LieModifier

ROUGH_SEVERITY = 0.6
_LIES = ("tee", "fairway", "rough", "sand")
_STREAM = 14610


def lie_modifier_for_table(
    seed: int, *, rough_severity: float = ROUGH_SEVERITY
) -> LieModifier:
    """Non-identity impact-layer modifier; same (rows, lie) always gives the same rows."""

    def modify(rows: np.ndarray, lie: str) -> np.ndarray:
        require(lie in _LIES, f"lie must be one of {_LIES}", lie)
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, _STREAM, _LIES.index(lie)]))
        )
        return make_lie_modifier(rough_severity, rng)(rows, lie)

    return modify
