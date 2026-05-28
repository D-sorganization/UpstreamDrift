"""Regression tests for HoleMDP / _action_q internals (issue #6343)."""

from __future__ import annotations


def test_condition_modifier_table_second_col_is_sigma_long() -> None:
    """Regression for #6343: cell_mods[..., 1] must be sigma_long_mult.

    In _action_q, dlong is scaled by cell_mods[..., 1]. Under heavy rough,
    dist_mult < 1 (shorter shots) but sigma_long_mult > 1 (more dispersion).
    Using dist_mult for the dispersion offset (the bug) would shrink the
    longitudinal spread instead of growing it, making condition severity
    have the wrong effect on shot scatter.
    """
    from src.shared.python.sg_optimizer.course.conditions import (
        CourseConditions,
        GreenModel,
        RoughModel,
        TreeModel,
    )
    from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES
    from src.shared.python.sg_optimizer.mdp.transition import _condition_modifiers

    conditions = CourseConditions(
        rough=RoughModel.preset("us_open"),
        trees=TreeModel.preset("decorative"),
        greens=GreenModel.preset("slow"),
    )
    dist_m, sl_m, slat_m = _condition_modifiers(LIE_CODES["rough"], conditions)

    # us_open rough: shot distance reduced, dispersion increased.
    assert dist_m < 1.0, f"expected dist_m < 1 in us_open rough, got {dist_m}"
    assert sl_m > 1.0, f"expected sl_m > 1 in us_open rough, got {sl_m}"
    # The two differ — confirms they carry distinct semantic values and
    # the vectorised _action_q cannot substitute one for the other.
    assert abs(dist_m - sl_m) > 0.1, (
        f"dist_m={dist_m} and sl_m={sl_m} must differ by >0.1 so the regression "
        "is detectable"
    )


def test_condition_modifier_fairway_identity() -> None:
    """Fairway/default lie must return (1, 1, 1) — no modifier applied."""
    from src.shared.python.sg_optimizer.course.conditions import (
        CourseConditions,
        GreenModel,
        RoughModel,
        TreeModel,
    )
    from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES
    from src.shared.python.sg_optimizer.mdp.transition import _condition_modifiers

    default = CourseConditions(
        rough=RoughModel(severity=0.0),
        trees=TreeModel.preset("decorative"),
        greens=GreenModel.preset("slow"),
    )
    dist_m, sl_m, slat_m = _condition_modifiers(LIE_CODES["fairway"], default)
    assert dist_m == 1.0
    assert sl_m == 1.0
    assert slat_m == 1.0
