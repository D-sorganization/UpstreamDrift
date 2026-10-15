"""Public API tests for motion_matching.force_nullspace package split (#12552)."""

from __future__ import annotations

import pytest

from src.shared.python.motion_matching import force_nullspace
from src.shared.python.motion_matching.force_nullspace import (
    constraints,
    redistribution,
    space,
    tradeoffs,
)

pytestmark = pytest.mark.unit

EXPECTED_PUBLIC_NAMES = {
    "NullSpaceAnalysis",
    "ForceNullSpace",
    "FrictionCone",
    "ForceConstraints",
    "RedistributionResult",
    "redistribute_forces",
    "redistribute_trajectory",
    "TradeoffAlternative",
    "TorqueDistributionTradeoff",
    "explore_torque_tradeoffs",
}


def test_public_all_matches_pre_split_set() -> None:
    """Assert __all__ and top-level exports match the pre-split public name set."""
    assert set(force_nullspace.__all__) == EXPECTED_PUBLIC_NAMES
    for name in EXPECTED_PUBLIC_NAMES:
        assert hasattr(force_nullspace, name)


def test_submodule_exports_and_identity() -> None:
    """Assert each submodule exports its designated symbols and matches top-level re-exports."""
    assert space.NullSpaceAnalysis is force_nullspace.NullSpaceAnalysis
    assert space.ForceNullSpace is force_nullspace.ForceNullSpace

    assert constraints.FrictionCone is force_nullspace.FrictionCone
    assert constraints.ForceConstraints is force_nullspace.ForceConstraints

    assert redistribution.RedistributionResult is force_nullspace.RedistributionResult
    assert redistribution.redistribute_forces is force_nullspace.redistribute_forces
    assert (
        redistribution.redistribute_trajectory
        is force_nullspace.redistribute_trajectory
    )

    assert tradeoffs.TradeoffAlternative is force_nullspace.TradeoffAlternative
    assert (
        tradeoffs.TorqueDistributionTradeoff
        is force_nullspace.TorqueDistributionTradeoff
    )
    assert (
        tradeoffs.explore_torque_tradeoffs is force_nullspace.explore_torque_tradeoffs
    )


def test_private_helpers_preserved_on_package_and_submodules() -> None:
    """Assert private helpers and monkeypatch targets remain accessible on the package."""
    assert hasattr(force_nullspace, "_array")
    assert hasattr(force_nullspace, "_margin_violation")
    assert hasattr(force_nullspace, "_finalize_redistribution")
    assert hasattr(force_nullspace, "_is_solution_feasible")
    assert hasattr(force_nullspace, "minimize")
