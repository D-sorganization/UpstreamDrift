"""SHOW-B3b (#14610): lie modifier with common random numbers across cells."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model.dispersion_lie import (
    ROUGH_SEVERITY,
    lie_modifier_for_table,
)
from src.shared.python.sg_optimizer.shot_model.skill_family import IMPACT_FIELDS

pytestmark = pytest.mark.unit
SPEED, FACE = (
    IMPACT_FIELDS.index("clubhead_speed_mph"),
    IMPACT_FIELDS.index("face_angle_deg"),
)


def _rows(n: int = 200, speed: float = 100.0) -> np.ndarray:
    rows = np.zeros((n, len(IMPACT_FIELDS)))
    rows[:, SPEED] = speed
    return rows


def test_same_rows_same_lie_give_identical_output_every_call() -> None:
    modify = lie_modifier_for_table(seed=5)
    np.testing.assert_array_equal(modify(_rows(), "rough"), modify(_rows(), "rough"))


def test_call_order_does_not_change_any_cell() -> None:
    a, b = lie_modifier_for_table(seed=5), lie_modifier_for_table(seed=5)
    first = [a(_rows(speed=s), lie) for s in (90.0, 110.0) for lie in ("rough", "sand")]
    second = [
        b(_rows(speed=s), lie) for s in (110.0, 90.0) for lie in ("sand", "rough")
    ]
    np.testing.assert_array_equal(first[0], second[3])
    np.testing.assert_array_equal(first[3], second[0])


def test_skills_share_the_same_lie_noise_block() -> None:
    modify = lie_modifier_for_table(seed=5)
    slow, fast = modify(_rows(speed=80.0), "rough"), modify(_rows(speed=120.0), "rough")
    np.testing.assert_allclose(slow[:, SPEED] / 80.0, fast[:, SPEED] / 120.0)
    np.testing.assert_array_equal(slow[:, FACE], fast[:, FACE])


def test_rough_is_non_identity_fairway_is_identity_and_seed_matters() -> None:
    modify = lie_modifier_for_table(seed=5)
    assert np.array_equal(modify(_rows(), "fairway"), _rows())
    rough = modify(_rows(), "rough")
    assert rough[:, SPEED].mean() < 100.0 and rough[:, SPEED].std() > 0.0
    assert not np.array_equal(rough, lie_modifier_for_table(seed=6)(_rows(), "rough"))


def test_unknown_lie_fails_closed_and_severity_is_exposed() -> None:
    with pytest.raises(ContractViolationError):
        lie_modifier_for_table(seed=5)(_rows(), "swamp")
    assert 0.0 < ROUGH_SEVERITY <= 1.0


def test_seed_and_severity_are_recorded_in_the_table_provenance() -> None:
    from src.shared.python.sg_optimizer.shot_model.dispersion_table import (
        TableGrid,
        build_grid_table,
    )
    from src.shared.python.sg_optimizer.shot_model.skill_registry import (
        get_skill_family,
    )

    def sim(impacts, skill, wind):
        return np.column_stack([impacts[:, SPEED], impacts[:, FACE]])

    grid = TableGrid(("mid_iron",), ("fairway", "rough"), ("calm",))
    table = build_grid_table(
        get_skill_family("tour"),
        sim,
        grid,
        n=16,
        lie_modifier=lie_modifier_for_table(7),
    )
    assert table.provenance["lie_model"] == {
        "seed": 7,
        "rough_severity": ROUGH_SEVERITY,
    }
