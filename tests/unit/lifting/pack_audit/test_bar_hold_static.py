"""Engine-free static-hold split of a held bar (LIFT-4, #11744, GCV-7).

Runs without any physics engine: the rigid-weld adapters (Pinocchio, then
OpenSim and Drake) all route through this helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.pack_audit.adapters.bar_hold_static import (
    static_hold_result,
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


# --------------------------------------------------------------------------
# static_hold_result() -- shared EngineAdapter.bar_hold_wrench() dict builder.
# --------------------------------------------------------------------------


def test_static_hold_result_builds_contract_dict_from_a_split() -> None:
    bar_mass_kg = 20.0
    split = static_hold_split(
        bar_mass_kg, (0.0, 0.0, -_G), (0.0, 0.0, 1.0), _GRIP_L, _GRIP_R
    )
    residuals = {"L": 0.0, "R": 1.5e-6}

    result = static_hold_result(bar_mass_kg, split, "test-method", 2, residuals)

    assert result["available"] is True
    assert result["reason"] == ""
    assert result["split_method"] == "allocation"
    assert result["bar_mass_kg"] == bar_mass_kg
    assert result["bar_weight_n"] == pytest.approx(bar_mass_kg * _G)
    assert result["hand_force_n"]["L"][2] == pytest.approx(bar_mass_kg * _G / 2.0)
    assert result["hand_force_n"]["R"][2] == pytest.approx(bar_mass_kg * _G / 2.0)
    assert result["sum_vertical_n"] == pytest.approx(bar_mass_kg * _G)
    assert result["relative_error"] == pytest.approx(0.0, abs=1e-9)
    assert result["split_left_fraction"] == pytest.approx(0.5)
    assert len(result["couple_at_midpoint_nm"]) == 3
    assert result["bar_linear_accel_mps2"] is None
    assert result["method"] == "test-method"
    assert result["n_welds"] == 2
    assert result["hand_grip_residual_m"] == residuals


def test_static_hold_result_rejects_zero_vertical_sum() -> None:
    split = {
        "hand_force_n": {"L": np.zeros(3), "R": np.zeros(3)},
        "couple_at_midpoint_nm": np.zeros(3),
        "gravity_mag_mps2": _G,
    }
    with pytest.raises(ValueError):
        static_hold_result(20.0, split, "test-method", 2, {"L": 0.0, "R": 0.0})


@pytest.mark.parametrize("bar_mass_kg", [0.0, -1.0, float("nan")])
def test_static_hold_result_rejects_invalid_bar_mass(bar_mass_kg: float) -> None:
    split = {
        "hand_force_n": {
            "L": np.array([0.0, 0.0, 100.0]),
            "R": np.array([0.0, 0.0, 100.0]),
        },
        "couple_at_midpoint_nm": np.zeros(3),
        "gravity_mag_mps2": _G,
    }
    with pytest.raises(ValueError):
        static_hold_result(bar_mass_kg, split, "test-method", 2, {"L": 0.0, "R": 0.0})
