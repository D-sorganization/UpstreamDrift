"""Golfer generator adapter control width follows the registry (#12670)."""

from __future__ import annotations

import pytest

from src.shared.python.neural_motion.matrix.adapters import (
    ConstrainedUpperBodyGeneratorAdapter,
)
from src.shared.python.tour_baselines.registry import get_golf_model

MODEL_ID = "constrained_upper_body_golfer"


def test_sampled_controls_match_registry_u_dim() -> None:
    identity = get_golf_model(MODEL_ID)
    traj = ConstrainedUpperBodyGeneratorAdapter(
        MODEL_ID
    ).generate_candidate_trajectory()
    assert identity.independent_dof == 6
    assert traj["u"].shape[1] == identity.independent_dof
    assert traj["q"].shape[1] == identity.dof
    assert traj["v"].shape[1] == identity.dof


def test_sealed_u_dim_disagreeing_with_model_raises() -> None:
    with pytest.raises(ValueError, match="u_dim"):
        ConstrainedUpperBodyGeneratorAdapter(MODEL_ID, sealed_u_dim=5)


def test_sealed_u_dim_agreeing_with_model_is_accepted() -> None:
    adapter = ConstrainedUpperBodyGeneratorAdapter(MODEL_ID, sealed_u_dim=6)
    assert adapter.u_dim == 6
