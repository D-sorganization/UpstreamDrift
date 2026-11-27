"""Lie modifier for dispersion-table builds, with common random numbers (#14610).

Wraps ``physics.lie_interaction.make_lie_modifier``.  A fresh generator is
derived from ``SeedSequence([seed, 14610, lie_index])`` on **every call**, so the
noise block depends only on (seed, lie): all skills share it, the output is
identical on repeat calls, and the table does not change with loop order.  The
seed and rough severity are exposed as ``provenance``, which ``build_grid_table``
hashes into the table.  Limit: ``perturb_impact_rows`` draws channel by channel,
so the noise for ``n/2`` rows is not the first half of the noise for ``n`` rows
(the table's prefix stability covers the impact draws, not the lie noise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.physics.lie_interaction import make_lie_modifier

ROUGH_SEVERITY = 0.6
_LIES = ("tee", "fairway", "rough", "sand")
_STREAM = 14610


@dataclass(frozen=True)
class TableLieModifier:
    """``LieModifier``: the same (rows, lie) always gives the same rows."""

    seed: int
    rough_severity: float = ROUGH_SEVERITY

    @property
    def provenance(self) -> dict[str, Any]:
        return {"lie_model": {"seed": self.seed, "rough_severity": self.rough_severity}}

    def with_spin(self, rows: np.ndarray, lie: str) -> tuple[np.ndarray, np.ndarray]:
        """The rows of ``__call__`` plus the per-shot lie spin factor."""
        require(lie in _LIES, f"lie must be one of {_LIES}", lie)
        seq = np.random.SeedSequence([self.seed, _STREAM, _LIES.index(lie)])
        rng = np.random.Generator(np.random.PCG64(seq))
        return make_lie_modifier(self.rough_severity, rng).with_spin(rows, lie)

    def __call__(self, rows: np.ndarray, lie: str) -> np.ndarray:
        return self.with_spin(rows, lie)[0]


def lie_modifier_for_table(
    seed: int, *, rough_severity: float = ROUGH_SEVERITY
) -> TableLieModifier:
    """Non-identity impact-layer modifier for a table build."""
    return TableLieModifier(seed, rough_severity)
