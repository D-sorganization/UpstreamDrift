"""Unit tests for sg_optimizer.mdp.tree_model (Phase 2).

Tests TreeModel punch-out probability, transition modification,
and distance/dispersion multipliers.
"""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.mdp.tree_model import TreeModel


# ---------------------------------------------------------------------------
# Contract checks
# ---------------------------------------------------------------------------


def test_negative_penalization_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        TreeModel(penalization=-0.01)


def test_penalization_over_one_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        TreeModel(penalization=1.01)


# ---------------------------------------------------------------------------
# is_forced_punch_out property
# ---------------------------------------------------------------------------


def test_decorative_not_forced():
    assert not TreeModel(penalization=0.1).is_forced_punch_out


def test_full_jail_is_forced():
    assert TreeModel(penalization=1.0).is_forced_punch_out


def test_threshold_boundary():
    just_below = TreeModel(penalization=0.85)
    just_above = TreeModel(penalization=0.86)
    assert not just_below.is_forced_punch_out
    assert just_above.is_forced_punch_out


# ---------------------------------------------------------------------------
# forced_punch_out_probability
# ---------------------------------------------------------------------------


def test_probability_zero_no_trees():
    from src.shared.python.sg_optimizer.course.features import StateFeatures

    model = TreeModel(penalization=0.9)
    sf = StateFeatures(
        distance_to_pin_m=50.0,
        distance_to_center_m=50.0,
        lie="fairway",
    )
    assert model.forced_punch_out_probability(sf) == 0.0


def test_probability_one_when_jail_and_in_trees():
    from src.shared.python.sg_optimizer.course.features import StateFeatures

    model = TreeModel(penalization=1.0)
    sf = StateFeatures(
        distance_to_pin_m=50.0,
        distance_to_center_m=50.0,
        lie="trees",
    )
    assert model.forced_punch_out_probability(sf) == pytest.approx(1.0)


def test_probability_monotone_with_penalization():
    """Higher penalization → higher punch-out probability (in trees)."""
    from src.shared.python.sg_optimizer.course.features import StateFeatures

    def p(pen: float) -> float:
        sf = StateFeatures(
            distance_to_pin_m=50.0, distance_to_center_m=50.0, lie="trees"
        )
        return TreeModel(penalization=pen).forced_punch_out_probability(sf)

    assert p(0.1) < p(0.3) < p(0.6) < p(0.9)


def test_probability_none_features():
    """With no state_features, probability depends only on penalization."""
    low = TreeModel(penalization=0.1).forced_punch_out_probability(None)
    high = TreeModel(penalization=0.9).forced_punch_out_probability(None)
    assert low < high


# ---------------------------------------------------------------------------
# apply_to_transition
# ---------------------------------------------------------------------------


def test_apply_no_trees_unchanged_totals():
    """Zero penalization → punch_out = 0, rest unchanged."""
    model = TreeModel(penalization=0.0)
    probs = {"normal": 0.7, "mishit": 0.3}
    result = model.apply_to_transition(probs)
    assert result["punch_out"] == pytest.approx(0.0)
    assert result["normal"] == pytest.approx(0.7)
    assert result["mishit"] == pytest.approx(0.3)


def test_apply_full_jail_all_punch_out():
    """Penalization=1.0 → all mass moves to punch_out."""
    model = TreeModel(penalization=1.0)
    probs = {"normal": 0.6, "mishit": 0.4}
    result = model.apply_to_transition(probs)
    assert result["punch_out"] == pytest.approx(1.0)
    assert result["normal"] == pytest.approx(0.0)
    assert result["mishit"] == pytest.approx(0.0)


def test_apply_sums_to_one():
    model = TreeModel(penalization=0.6)
    probs = {"a": 0.5, "b": 0.3, "c": 0.2}
    result = model.apply_to_transition(probs)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)


def test_apply_preserves_original():
    model = TreeModel(penalization=0.5)
    probs = {"a": 1.0}
    original = dict(probs)
    model.apply_to_transition(probs)
    assert probs == original


def test_apply_empty_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        TreeModel(penalization=0.5).apply_to_transition({})


def test_apply_non_summing_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        TreeModel(penalization=0.5).apply_to_transition({"a": 0.5, "b": 0.1})


# ---------------------------------------------------------------------------
# distance / dispersion multipliers
# ---------------------------------------------------------------------------


def test_distance_multiplier_decreases_with_penalization():
    low = TreeModel(penalization=0.1).distance_multiplier()
    high = TreeModel(penalization=0.9).distance_multiplier()
    assert low > high


def test_distance_multiplier_floor():
    """Even at max penalization, distance multiplier >= 0.05."""
    assert TreeModel(penalization=1.0).distance_multiplier() >= 0.05


def test_dispersion_multiplier_increases_with_penalization():
    low = TreeModel(penalization=0.1).dispersion_multiplier()
    high = TreeModel(penalization=0.9).dispersion_multiplier()
    assert high > low


def test_repr():
    m = TreeModel(penalization=0.5)
    assert "TreeModel" in repr(m)
    assert "0.5" in repr(m)
