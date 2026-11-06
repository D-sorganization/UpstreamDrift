"""TableShotModel consumes the requested lie, wind and slope cells (#14743).

The model used to look up ``fairway/calm/flat`` whatever the shot's lie or the
course conditions, so a table's rough, wind and slope cells were never read.
These tests use distinguishable cells so a wrong selection shows in the mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES
from src.shared.python.sg_optimizer.mdp.action import ShotAction
from src.shared.python.sg_optimizer.shot_model.baseline import load_baseline
from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
    DispersionCell,
    DispersionTable,
)
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile
from src.shared.python.sg_optimizer.shot_model.registry import get_shot_model
from src.shared.python.sg_optimizer.shot_model.table_model import TableShotModel

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[4] / "data/sg_optimizer/baselines/pga_tour.yaml"
_CARRY = {
    ("fairway", "calm", "flat"): 250.0,
    ("rough", "calm", "flat"): 100.0,
    ("fairway", "head_5", "flat"): 230.0,
    ("fairway", "calm", "uphill"): 240.0,
}


def _model() -> TableShotModel:
    cells = tuple(
        DispersionCell(
            "driver", "driver", lie, wind, slope, (carry, 0.0), (0.5, 0.5), 0.0, 1000
        )
        for (lie, wind, slope), carry in _CARRY.items()
    )
    table = DispersionTable(cells=cells, provenance={"table_hash": "fixture"})
    return TableShotModel(table, get_shot_model("baseline", load_baseline(YAML)))


def _mean_carry(lie: str, conditions: CourseConditions) -> float:
    samples = _model().sample(
        PlayerProfile(name="p", baseline=str(YAML)),
        ShotAction("driver", 0.0),
        LIE_CODES[lie],
        conditions,
        np.random.default_rng(0),
        400,
    )
    return float(samples.along.mean())


def _conditions(**kw) -> CourseConditions:
    base = CourseConditions.tournament()
    return CourseConditions(base.rough, base.trees, base.greens, **kw)


def test_default_conditions_keep_the_fairway_calm_flat_cell() -> None:
    assert _mean_carry("fairway", CourseConditions.tournament()) == pytest.approx(
        250.0, abs=0.5
    )


def test_rough_lie_reads_the_rough_cell_not_a_scaled_fairway_cell() -> None:
    assert _mean_carry("rough", CourseConditions.tournament()) == pytest.approx(
        100.0, abs=0.5
    )


def test_requested_wind_bin_is_consumed() -> None:
    assert _mean_carry("fairway", _conditions(wind="head_5")) == pytest.approx(
        230.0, abs=0.5
    )


def test_requested_slope_bin_is_consumed() -> None:
    assert _mean_carry("fairway", _conditions(slope="uphill")) == pytest.approx(
        240.0, abs=0.5
    )


def test_absent_wind_bin_is_rejected_not_replaced_by_calm() -> None:
    with pytest.raises(ContractViolationError, match="tail_15"):
        _mean_carry("fairway", _conditions(wind="tail_15"))


def test_lie_without_table_cells_scales_the_fairway_cell() -> None:
    """A lie the table was not built for keeps the documented fairway + modifier path."""
    assert _mean_carry("sand", CourseConditions.tournament()) == pytest.approx(
        250.0 * 0.85, abs=0.5
    )


def test_selected_cell_names_the_requested_condition() -> None:
    cell = _model().select_cell("driver", LIE_CODES["rough"], _conditions(slope="flat"))
    assert cell is not None
    assert cell.key == ("driver", "driver", "rough", "calm", "flat")


def test_conditions_default_to_calm_flat_and_round_trip(tmp_path: Path) -> None:
    assert (CourseConditions.benign().wind, CourseConditions.benign().slope) == (
        "calm",
        "flat",
    )
    path = tmp_path / "c.yaml"
    _conditions(wind="cross_5", slope="downhill").to_yaml(path)
    loaded = CourseConditions.from_yaml(path)
    assert (loaded.wind, loaded.slope) == ("cross_5", "downhill")


def test_conditions_reject_an_empty_bin_name() -> None:
    with pytest.raises(ContractViolationError):
        _conditions(wind="")
