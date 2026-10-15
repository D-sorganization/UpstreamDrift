"""``explore_torque_tradeoffs`` keeps its shape and names itself a weight sweep (NSX-25, #12568)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.shared.python.motion_matching.force_nullspace import (
    ForceConstraints,
    ForceNullSpace,
    TorqueDistributionTradeoff,
    explore_torque_tradeoffs,
)

pytestmark = pytest.mark.unit


def _problem():
    a = np.array([[1.0, 1.0, 1.0]])
    b = np.array([100.0])
    space = ForceNullSpace.from_balance(a, b)
    constraints = ForceConstraints(lower=np.zeros(3), upper_bound=np.full(3, 500.0))
    return space, constraints


def test_shim_warns_that_it_is_a_weight_sweep_and_names_the_replacement() -> None:
    space, constraints = _problem()
    with pytest.warns(FutureWarning, match="weight sweep") as record:
        explore_torque_tradeoffs(
            space, constraints, lead_indices=[0], trail_indices=[1], ground_indices=[2]
        )
    text = str(record[0].message)
    assert "pareto_front" in text
    assert "not a Pareto" in text


def test_shim_still_returns_its_old_shape() -> None:
    space, constraints = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        out = explore_torque_tradeoffs(
            space, constraints, lead_indices=[0], trail_indices=[1], ground_indices=[2]
        )
    assert isinstance(out, TorqueDistributionTradeoff)
    names = [alt.name for alt in out.alternatives]
    assert "conservative_default" in names and "baseline_minimum_effort" in names
    assert len(out.alternatives) >= 5
    table = out.to_pareto_table()
    assert {"name", "strategy", "feasible", "cost", "trail_share"} <= set(table[0])
