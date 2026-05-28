"""Unit tests for sg_optimizer.mdp.tree_model — full TreeModel."""

from __future__ import annotations

import pytest

from src.shared.python.sg_optimizer.mdp.tree_model import TreeModel


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_valid():
    m = TreeModel(penalization=0.5)
    assert m.penalization == pytest.approx(0.5)


def test_construction_boundary_zero():
    m = TreeModel(penalization=0.0)
    assert m.penalization == pytest.approx(0.0)


def test_construction_boundary_one():
    m = TreeModel(penalization=1.0)
    assert m.penalization == pytest.approx(1.0)


def test_construction_invalid_raises():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        TreeModel(penalization=1.1)

    with pytest.raises(ContractViolationError):
        TreeModel(penalization=-0.1)


# ---------------------------------------------------------------------------
# forced_punch_out_probability — threshold
# ---------------------------------------------------------------------------


def test_forced_punch_out_above_threshold_returns_one():
    """penalization > 0.85 must return 1.0 (full jail)."""
    m = TreeModel(penalization=0.90)
    assert m.forced_punch_out_probability() == pytest.approx(1.0)


def test_forced_punch_out_at_threshold_boundary():
    """penalization == 0.85 is NOT strictly > 0.85, so NOT forced."""
    m = TreeModel(penalization=0.85)
    assert m.forced_punch_out_probability() < 1.0


def test_forced_punch_out_zero_penalization_is_near_zero():
    m = TreeModel(penalization=0.0)
    assert m.forced_punch_out_probability() == pytest.approx(0.0, abs=1e-9)


def test_forced_punch_out_increases_with_penalization():
    probs = [
        TreeModel(penalization=p).forced_punch_out_probability()
        for p in [0.1, 0.3, 0.5, 0.7, 0.84]
    ]
    assert probs == sorted(probs), f"probabilities not monotone: {probs}"


def test_forced_punch_out_with_non_tree_lie_is_zero():
    from src.shared.python.sg_optimizer.course.features import StateFeatures

    m = TreeModel(penalization=0.9)
    features = StateFeatures(
        distance_to_pin_m=100.0,
        distance_to_center_m=100.0,
        lie="fairway",
    )
    assert m.forced_punch_out_probability(features) == pytest.approx(0.0)


def test_forced_punch_out_with_trees_lie_above_threshold():
    from src.shared.python.sg_optimizer.course.features import StateFeatures

    m = TreeModel(penalization=0.9)
    features = StateFeatures(
        distance_to_pin_m=100.0,
        distance_to_center_m=100.0,
        lie="trees",
    )
    assert m.forced_punch_out_probability(features) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_forced_punch_out property
# ---------------------------------------------------------------------------


def test_is_forced_punch_out_true_above_threshold():
    assert TreeModel(penalization=0.86).is_forced_punch_out is True


def test_is_forced_punch_out_false_below_threshold():
    assert TreeModel(penalization=0.50).is_forced_punch_out is False


def test_is_forced_punch_out_matches_phase1_stub():
    """Phase 2 property must agree with Phase-1 stub threshold."""
    from src.shared.python.sg_optimizer.course.conditions import (
        TreeModel as Phase1TreeModel,
    )

    for p in [0.10, 0.50, 0.80, 0.85, 0.86, 0.90, 1.00]:
        p1 = Phase1TreeModel(penalization=p)
        p2 = TreeModel(penalization=p)
        assert p1.is_forced_punch_out() == p2.is_forced_punch_out, (
            f"Mismatch at penalization={p}: "
            f"phase1={p1.is_forced_punch_out()}, phase2={p2.is_forced_punch_out}"
        )


# ---------------------------------------------------------------------------
# apply_to_transition
# ---------------------------------------------------------------------------


def test_apply_to_transition_sums_to_one():
    m = TreeModel(penalization=0.5)
    probs = {"green": 0.6, "rough": 0.4}
    result = m.apply_to_transition(probs)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)


def test_apply_to_transition_punch_out_present():
    m = TreeModel(penalization=0.5)
    probs = {"green": 0.7, "rough": 0.3}
    result = m.apply_to_transition(probs)
    assert "punch_out" in result
    assert result["punch_out"] > 0.0


def test_apply_to_transition_no_penalty_unchanged_except_punch_out_zero():
    m = TreeModel(penalization=0.0)
    probs = {"green": 0.7, "rough": 0.3}
    result = m.apply_to_transition(probs)
    assert result["punch_out"] == pytest.approx(0.0, abs=1e-12)
    assert result["green"] == pytest.approx(0.7, abs=1e-9)


def test_apply_to_transition_full_jail_all_punch_out():
    m = TreeModel(penalization=1.0)
    probs = {"green": 0.6, "rough": 0.4}
    result = m.apply_to_transition(probs)
    assert result["punch_out"] == pytest.approx(1.0, abs=1e-9)
    assert result["green"] == pytest.approx(0.0, abs=1e-9)


def test_apply_to_transition_empty_raises():
    from src.shared.python.contracts import ContractViolationError

    m = TreeModel(penalization=0.5)
    with pytest.raises(ContractViolationError):
        m.apply_to_transition({})


def test_apply_to_transition_input_unchanged():
    m = TreeModel(penalization=0.5)
    original = {"green": 0.6, "rough": 0.4}
    probs = dict(original)
    m.apply_to_transition(probs)
    assert probs == original  # input not mutated


# ---------------------------------------------------------------------------
# Multipliers (inherited from Phase-1 interface)
# ---------------------------------------------------------------------------


def test_distance_multiplier_decreases_with_penalization():
    low = TreeModel(penalization=0.2).distance_multiplier()
    high = TreeModel(penalization=0.8).distance_multiplier()
    assert low > high


def test_dispersion_multiplier_increases_with_penalization():
    low = TreeModel(penalization=0.2).dispersion_multiplier()
    high = TreeModel(penalization=0.8).dispersion_multiplier()
    assert low < high


def test_repr():
    m = TreeModel(penalization=0.7)
    assert "0.7" in repr(m)
