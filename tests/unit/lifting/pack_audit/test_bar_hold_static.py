"""Engine-free static-hold split of a held bar (LIFT-4, #11744, GCV-7).

Runs without any physics engine: the rigid-weld adapters (Pinocchio, then
OpenSim and Drake) all route through this helper.
"""

from __future__ import annotations

import pytest

from src.shared.python.lifting.pack_audit.adapters.bar_hold_static import (
    static_hold_split,
)

pytestmark = pytest.mark.unit

_G = 9.81
_GRIP_L = (-0.3, 0.0, 1.0)
_GRIP_R = (0.3, 0.0, 1.0)


def test_split_symmetric_grip_balances_weight_evenly() -> None:
    bar_mass_kg = 20.0
    result = static_hold_split(
        bar_mass_kg, (0.0, 0.0, -_G), (0.0, 0.0, 1.0), _GRIP_L, _GRIP_R
    )

    left_f = result["hand_force_n"]["L"]
    right_f = result["hand_force_n"]["R"]
    weight_n = bar_mass_kg * _G
    assert left_f[2] == pytest.approx(weight_n / 2.0)
    assert right_f[2] == pytest.approx(weight_n / 2.0)
    # Sign (ADR-0052): the wrench is exerted by the hand ON the bar -> up.
    assert left_f[2] > 0.0
    assert right_f[2] > 0.0


def test_split_asymmetric_com_shifts_toward_nearer_hand() -> None:
    """Simple-beam statics: the support nearer the load carries more of it."""
    bar_mass_kg = 20.0
    # COM 0.2 m from the left grip and 0.4 m from the right grip.
    result = static_hold_split(
        bar_mass_kg, (0.0, 0.0, -_G), (-0.1, 0.0, 1.0), _GRIP_L, _GRIP_R
    )

    left_f = result["hand_force_n"]["L"]
    right_f = result["hand_force_n"]["R"]
    sum_vertical = left_f[2] + right_f[2]
    assert sum_vertical == pytest.approx(bar_mass_kg * _G)
    assert left_f[2] / sum_vertical == pytest.approx(2.0 / 3.0, abs=1e-9)


@pytest.mark.parametrize(
    ("bar_mass_kg", "gravity", "grip_l", "grip_r"),
    [
        (0.0, (0.0, 0.0, -_G), _GRIP_L, _GRIP_R),
        (-5.0, (0.0, 0.0, -_G), _GRIP_L, _GRIP_R),
        (float("nan"), (0.0, 0.0, -_G), _GRIP_L, _GRIP_R),
        (20.0, (0.0, 0.0, 0.0), _GRIP_L, _GRIP_R),
        (20.0, (0.0, 0.0, -_G), _GRIP_L, _GRIP_L),
        (20.0, (0.0, -_G), _GRIP_L, _GRIP_R),
    ],
)
def test_split_rejects_degenerate_inputs(
    bar_mass_kg: float,
    gravity: tuple[float, ...],
    grip_l: tuple[float, float, float],
    grip_r: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError):
        static_hold_split(bar_mass_kg, gravity, (0.0, 0.0, 1.0), grip_l, grip_r)
