"""Tests for legacy zero-side consistency shim (NSX-24, #12567)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.zero_side_shim import (
    ZeroSideConsistency,
    check_zero_side_consistency,
)

pytestmark = pytest.mark.unit


def test_consistent_unique_when_rhs_in_column_space() -> None:
    j_grip = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tau = j_grip.T @ np.array([2.0, -1.0])
    result = check_zero_side_consistency(
        tau,
        actuated_indices=[0, 1, 2],
        zero_joint_indices=[2],
        j_grip=j_grip,
    )
    assert result.status in (
        ZeroSideConsistency.CONSISTENT_UNIQUE,
        ZeroSideConsistency.CONSISTENT_NONUNIQUE,
    )
    assert result.residual_norm < 1e-9


def test_inconsistent_when_zero_side_conflicts_with_load() -> None:
    tau = np.array([10.0, 0.0, 5.0])
    result = check_zero_side_consistency(
        tau,
        actuated_indices=[0, 1, 2],
        zero_joint_indices=[0, 1],
    )
    assert result.status is ZeroSideConsistency.INCONSISTENT
    assert result.residual_norm > 1.0


def test_dbc_rejects_bad_atol() -> None:
    with pytest.raises(ValueError, match="atol"):
        check_zero_side_consistency(
            np.zeros(3),
            actuated_indices=[0, 1, 2],
            zero_joint_indices=[0],
            atol=0.0,
        )


def test_dbc_rejects_unactuated_or_out_of_range_zero_indices() -> None:
    with pytest.raises(ValueError, match="actuated_indices"):
        check_zero_side_consistency(
            np.zeros(6),
            actuated_indices=[3, 4, 5],
            zero_joint_indices=[1],
        )
    with pytest.raises(ValueError, match="out of range"):
        check_zero_side_consistency(
            np.zeros(4),
            actuated_indices=[1, 2, 3],
            zero_joint_indices=[4],
        )


def test_unrelated_root_residual_does_not_mark_zero_side_inconsistent() -> None:
    tau = np.array([0.005, 0.0, 2.0, -1.0])
    j_grip = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    result = check_zero_side_consistency(
        tau,
        actuated_indices=[2, 3],
        zero_joint_indices=[2],
        j_grip=j_grip,
    )
    assert result.status is not ZeroSideConsistency.INCONSISTENT
    assert result.residual_norm < 1e-9
