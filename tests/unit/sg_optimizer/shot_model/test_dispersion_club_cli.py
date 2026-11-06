"""SHOW-B3b remainder (#14610): club column, ground/lie CLI, convergence provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES
from src.shared.python.sg_optimizer.mdp.action import ShotAction
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    DispersionTable,
    TableGrid,
    build_grid_table,
    main,
)
from src.shared.python.sg_optimizer.shot_model.physics_simulator import (
    CLUB_LOFT_DEG,
    REPRESENTATIVE_CLUB,
)
from src.shared.python.sg_optimizer.shot_model.skill_family import SKILL_NAMES
from src.shared.python.sg_optimizer.shot_model.skill_registry import get_skill_family
from src.shared.python.sg_optimizer.shot_model.table_model import (
    CLUB_SKILL,
    TableShotModel,
)

pytestmark = pytest.mark.unit
FAMILY = get_skill_family("tour")


def _sim(impacts: np.ndarray, skill: str, wind: tuple[float, float]):
    return np.stack([2.0 * impacts[:, 0], 3.0 * impacts[:, 1]], axis=1)


def _grid(**kw) -> TableGrid:
    return TableGrid(("mid_iron", "driver"), ("fairway",), ("calm",), **kw)


def test_every_skill_has_a_representative_club_with_a_loft() -> None:
    assert set(REPRESENTATIVE_CLUB) == set(SKILL_NAMES) == set(CLUB_LOFT_DEG)
    for skill, club in REPRESENTATIVE_CLUB.items():
        if club in CLUB_SKILL:  # a bag club must map back to its own skill
            assert CLUB_SKILL[club] == skill


def test_representative_clubs_match_their_lofts() -> None:
    """A labelled club's loft stays within a few degrees of a real club's."""
    typical = {"8_iron": 37.0, "9_iron": 42.0, "pw": 45.0, "gw": 50.0}
    typical |= {"6_iron": 28.0, "4_iron": 23.0}
    for skill, club in REPRESENTATIVE_CLUB.items():
        if club in typical:
            assert abs(CLUB_LOFT_DEG[skill] - typical[club]) <= 3.0, (skill, club)


def test_club_column_is_distinct_from_the_skill_when_a_mapping_is_given() -> None:
    plain = build_grid_table(FAMILY, _sim, _grid(), n=8)
    assert {c.club for c in plain.cells} == {c.skill for c in plain.cells}
    clubbed = build_grid_table(FAMILY, _sim, _grid(clubs=REPRESENTATIVE_CLUB), n=8)
    cell = clubbed.lookup(
        "mid_iron", REPRESENTATIVE_CLUB["mid_iron"], "fairway", "calm", "flat"
    )
    assert cell.club != cell.skill
    with pytest.raises(ContractViolationError):
        clubbed.lookup("mid_iron", "mid_iron", "fairway", "calm", "flat")
    assert clubbed.provenance["clubs"]["mid_iron"] == cell.club
    assert clubbed.table_hash != plain.table_hash  # the labelling is hashed


def test_club_mapping_must_cover_every_skill_in_the_grid() -> None:
    with pytest.raises(ContractViolationError):
        build_grid_table(FAMILY, _sim, _grid(clubs={"driver": "driver"}), n=8)


def test_table_shot_model_reads_a_club_labelled_table() -> None:
    table = build_grid_table(FAMILY, _sim, _grid(clubs=REPRESENTATIVE_CLUB), n=64)

    class _Fallback:
        def sample(self, *args, **kw):  # pragma: no cover - must not be reached
            raise AssertionError("the labelled cell should be found")

    model = TableShotModel(table, _Fallback())
    out = model.sample(
        None,
        ShotAction("driver", 0.0),
        LIE_CODES["fairway"],
        CourseConditions.benign(),
        np.random.default_rng(1),
        16,
    )
    assert out.along.shape == (16,)


def test_cli_ground_table_has_lies_slopes_clubs_and_convergence(tmp_path: Path) -> None:
    out = tmp_path / "ground.json"
    argv = ["--family", "tour", "--n", "8", "--ground", "--out", str(out)]
    argv += ["--lies", "fairway,rough", "--slopes", "flat,up_5", "--winds", "calm"]
    assert main(argv) == 0
    table = DispersionTable.from_json(out)  # re-verifies the hash
    prov = table.provenance
    assert prov["ground_model"] and prov["lie_model"] and prov["clubs"]
    assert isinstance(prov["convergence"]["converged"], bool)
    assert {c.lie for c in table.cells} == {"fairway", "rough"}
    assert {c.slope for c in table.cells} == {"flat", "up_5"}
    assert all(c.club == REPRESENTATIVE_CLUB[c.skill] for c in table.cells)


def test_cli_without_ground_keeps_the_carry_table_and_skill_labels(
    tmp_path: Path,
) -> None:
    out = tmp_path / "carry.json"
    assert main(["--family", "tour", "--n", "8", "--out", str(out)]) == 0
    table = DispersionTable.from_json(out)
    assert "ground_model" not in table.provenance
    assert all(c.club == c.skill for c in table.cells)
