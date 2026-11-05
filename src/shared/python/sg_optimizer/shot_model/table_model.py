"""``physics`` backend: a ``ShotOutcomeModel`` that reads B3 dispersion tables."""

from __future__ import annotations

import numpy as np

from src.shared.python.sg_optimizer.shot_model.baseline_model import (
    BaselineShotModel,
    lie_modifiers,
)
from src.shared.python.sg_optimizer.shot_model.dispersion_table import DispersionTable
from src.shared.python.sg_optimizer.shot_model.outcome import (
    LandingSamples,
    ShotOutcomeModel,
)
from src.shared.python.sg_optimizer.shot_model.registry import register_shot_model

PHYSICS_BACKEND = "physics"
# Baseline club name -> skill family key (ASSUMPTION: iron groupings by number).
CLUB_SKILL: dict[str, str] = {
    "driver": "driver",
    "3_wood": "fairway_wood",
    "5_wood": "fairway_wood",
    "4_iron": "long_iron",
    "5_iron": "mid_iron",
    "6_iron": "mid_iron",
    "7_iron": "mid_iron",
    "8_iron": "short_iron",
    "9_iron": "short_iron",
    "pw": "wedge",
    "gw": "wedge",
    "sw": "wedge",
    "lw": "wedge",
}


class TableShotModel:
    """Sample landings from the table's calm/flat fairway cell, then apply the lie.

    Clubs without a table cell fall back to ``fallback`` (the tour baseline), so
    a partial table never breaks a solve. The table encodes the skill family, so
    the profile's own club multipliers are not applied here.
    """

    def __init__(self, table: DispersionTable, fallback: ShotOutcomeModel) -> None:
        self.table, self.fallback = table, fallback

    def sample(self, player, shot, lie, conditions, rng, n):  # noqa: ANN001
        skill = CLUB_SKILL.get(shot.club)
        key = (skill, skill, "fairway", "calm", "flat")
        if skill is None or not self.table.has_cell(*key):
            return self.fallback.sample(player, shot, lie, conditions, rng, n)
        cell = self.table.lookup(*key)
        dist, sl, slat = lie_modifiers(lie, conditions)
        z = rng.standard_normal((n, 2))
        z[:, 1] = cell.corr * z[:, 0] + np.sqrt(1.0 - cell.corr**2) * z[:, 1]
        return LandingSamples(
            along=cell.mean[0] * dist + cell.sd[0] * sl * z[:, 0],
            lateral=cell.mean[1] + cell.sd[1] * slat * z[:, 1],
        )


def register_physics_backend(
    table: DispersionTable, name: str = PHYSICS_BACKEND
) -> None:
    """Register ``name`` so ``get_shot_model(name)`` serves ``table``."""
    register_shot_model(
        name, lambda baseline: TableShotModel(table, BaselineShotModel(baseline))
    )


__all__ = ["CLUB_SKILL", "TableShotModel", "register_physics_backend"]
