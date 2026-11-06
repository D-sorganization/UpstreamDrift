"""``physics`` backend: a ``ShotOutcomeModel`` that reads B3 dispersion tables."""

from __future__ import annotations

import numpy as np

from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_NAMES
from src.shared.python.sg_optimizer.shot_model.baseline_model import (
    BaselineShotModel,
    lie_modifiers,
)
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    DispersionCell,
    DispersionTable,
)
from src.shared.python.sg_optimizer.shot_model.ground_runout import SKILL_CLUB
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
    """Sample landings from the table cell for the shot's lie, wind and slope.

    Cell routing (#14743), for a club the table covers:

    * the requested ``(lie, conditions.wind, conditions.slope)`` cell is used as
      is; its lie effect is already in the cell, so no lie multiplier is applied;
    * a lie the table has no cells for at all (for example ``sand`` on a
      fairway-only table) uses that wind/slope's fairway cell scaled by
      :func:`lie_modifiers` (the pre-#14743 behaviour, now explicit);
    * any other missing combination raises ``ContractViolationError`` rather
      than falling back to calm/flat.

    Clubs without a table cell fall back to ``fallback`` (the tour baseline), so
    a partial table never breaks a solve. The table encodes the skill family, so
    the profile's own club multipliers are not applied here.
    """

    def __init__(self, table: DispersionTable, fallback: ShotOutcomeModel) -> None:
        self.table, self.fallback = table, fallback
        self._lies = frozenset(cell.lie for cell in table.cells)
        self._skills = frozenset(cell.skill for cell in table.cells)

    def select_cell(
        self, club_name: str, lie: int, conditions: CourseConditions
    ) -> DispersionCell | None:
        """The cell a shot with ``club_name`` from ``lie`` reads, or ``None``.

        ``None`` means the table does not cover the club (use the fallback).
        Postcondition: a returned cell's ``wind``/``slope`` equal the requested bins.
        """
        skill = CLUB_SKILL.get(club_name)
        if skill not in self._skills:
            return None
        lie_name = LIE_NAMES.get(lie, "")
        table_lie = lie_name if lie_name in self._lies else "fairway"
        club = self.table.club_label(skill) or ""
        return self.table.lookup(
            skill or "", club, table_lie, conditions.wind, conditions.slope
        )

    def sample(self, player, shot, lie, conditions, rng, n):  # noqa: ANN001
        cell = self.select_cell(shot.club, lie, conditions)
        if cell is None:  # no cell for this club: use the baseline
            return self.fallback.sample(player, shot, lie, conditions, rng, n)
        dist, sl, slat = (
            (1.0, 1.0, 1.0)
            if cell.lie == LIE_NAMES.get(lie)
            else lie_modifiers(lie, conditions)
        )
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
