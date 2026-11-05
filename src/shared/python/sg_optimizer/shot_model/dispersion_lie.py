"""Lie modifier factory wired for dispersion-table builds (#14610)."""

from __future__ import annotations

import numpy as np

from src.shared.python.physics.lie_interaction import make_lie_modifier
from src.shared.python.sg_optimizer.shot_model.dispersion_table import LieModifier


def lie_modifier_for_table(seed: int, *, rough_severity: float = 0.6) -> LieModifier:
    """Non-identity impact-layer modifier; RNG is seed-stable per table build."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 14610]))
    return make_lie_modifier(rough_severity, rng)
