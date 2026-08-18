"""The DRFT structural correction ``delta_h`` (issue #8611).

The research addendum's central warning is that applying the inertial
term without ``delta_h`` produced the **wrong sign** of sinkage at every
``lambda`` from 1 to 100.  ``delta_h`` is therefore not optional, and
these tests pin the three properties the default model was shaped to
guarantee -- plus the honesty flag, which no model is allowed to set.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bunkershot3d.solvers import (
    GRAVITY_M_S2,
    CalibrationError,
    CrossoverSaturatingDepression,
    DepressionInputs,
    StructuralCorrection,
    WheelAnalogueDepression,
    ZeroDepression,
    default_structural_correction,
)

pytestmark = pytest.mark.unit

_DEPTH_SCALE_PA_PER_M = 2.4e6
_INERTIAL_SCALE = 1.1 * 1550.0

_MODELS: list[StructuralCorrection] = [
    ZeroDepression(),
    WheelAnalogueDepression(),
    CrossoverSaturatingDepression(),
]


def _inputs(depth_m: float, speed_m_s: float) -> DepressionInputs:
    """One-element depression inputs at a given depth and normal speed."""
    return DepressionInputs(
        depth_m=np.array([depth_m]),
        normal_speed_m_s=np.array([speed_m_s]),
        depth_stress_scale_pa_per_m=np.array([_DEPTH_SCALE_PA_PER_M]),
        inertial_stress_scale_pa_s2_per_m2=_INERTIAL_SCALE,
    )


class TestEveryModel:
    """Invariants that hold for the whole family."""

    @pytest.mark.parametrize("model", _MODELS, ids=lambda m: m.name)
    def test_depression_is_never_negative(self, model: StructuralCorrection) -> None:
        for speed in (0.0, 0.5, 7.0, 25.0, 60.0):
            assert float(model.depression_m(_inputs(0.04, speed))[0]) >= 0.0

    @pytest.mark.parametrize("model", _MODELS, ids=lambda m: m.name)
    def test_no_model_claims_to_be_calibrated_for_a_wedge(
        self, model: StructuralCorrection
    ) -> None:
        # This is the honesty assertion. The wedge-specific form of
        # delta_h is unknown; the addendum says so explicitly, and no
        # implementation is allowed to imply otherwise.
        assert model.is_calibrated_for_wedge is False
        assert not model.provenance.is_measured

    @pytest.mark.parametrize("model", _MODELS, ids=lambda m: m.name)
    def test_rejects_a_signed_depth_coordinate(
        self, model: StructuralCorrection
    ) -> None:
        with pytest.raises(CalibrationError, match="signed z coordinate"):
            model.depression_m(_inputs(-0.04, 10.0))

    @pytest.mark.parametrize("model", _MODELS, ids=lambda m: m.name)
    def test_rejects_mismatched_array_shapes(self, model: StructuralCorrection) -> None:
        bad = DepressionInputs(
            depth_m=np.array([0.04, 0.05]),
            normal_speed_m_s=np.array([10.0]),
            depth_stress_scale_pa_per_m=np.array([_DEPTH_SCALE_PA_PER_M]),
            inertial_stress_scale_pa_s2_per_m2=_INERTIAL_SCALE,
        )
        with pytest.raises(CalibrationError):
            model.depression_m(bad)


class TestZeroDepression:
    """The measured plate limit."""

    def test_is_identically_zero(self) -> None:
        depression = ZeroDepression().depression_m(_inputs(0.04, 25.0))
        assert float(depression[0]) == 0.0


class TestWheelAnalogue:
    """The one published closed form, and why it does not transfer."""

    def test_reproduces_v_squared_over_g(self) -> None:
        depression = WheelAnalogueDepression().depression_m(_inputs(0.04, 25.0))
        assert float(depression[0]) == pytest.approx(25.0**2 / GRAVITY_M_S2, rel=1e-12)

    def test_at_clubhead_speed_it_is_tens_of_metres(self) -> None:
        # 63.7 m of "free-surface depression" on a 40 mm divot. Kept as a
        # test because it is the clearest statement that the wheel form
        # is a reference, not a usable default.
        depression = float(
            WheelAnalogueDepression().depression_m(_inputs(0.04, 25.0))[0]
        )
        assert depression > 50.0

    def test_rejects_a_negative_coefficient(self) -> None:
        with pytest.raises(CalibrationError):
            WheelAnalogueDepression(coefficient=-1.0)


class TestCrossoverSaturatingDefault:
    """The documented default and the three properties it was shaped for."""

    def test_is_the_package_default(self) -> None:
        assert isinstance(
            default_structural_correction(), CrossoverSaturatingDepression
        )

    def test_vanishes_in_the_quasi_static_limit(self) -> None:
        # Below the crossover the plate observation delta_h ~ 0 must be
        # recovered, so the model degrades to quasi-static RFT exactly
        # where quasi-static RFT is right.
        model = CrossoverSaturatingDepression()
        slow = float(model.depression_m(_inputs(0.04, 0.01))[0])
        assert slow / 0.04 < 1e-5

    def test_never_reaches_the_element_depth(self) -> None:
        # If delta_h could reach |z| the depth term would change sign,
        # which is the failure the correction exists to prevent.
        model = CrossoverSaturatingDepression()
        for speed in (10.0, 25.0, 100.0, 1000.0):
            depression = float(model.depression_m(_inputs(0.04, speed))[0])
            assert depression < 0.04
            assert depression <= model.saturation_fraction * 0.04

    def test_leaves_the_depth_term_at_about_one_percent_at_clubhead_speed(
        self,
    ) -> None:
        # Katsuragi & Durian's inertial-dominance criterion independently
        # puts the rate-independent term at ~1% of the load at 25 m/s and
        # 5 cm; the default reproduces that order. A sanity check on the
        # convention, explicitly not a calibration of it.
        model = CrossoverSaturatingDepression()
        depression = float(model.depression_m(_inputs(0.05, 25.0))[0])
        depth_stress_pa = _DEPTH_SCALE_PA_PER_M * (0.05 - depression)
        inertial_stress_pa = _INERTIAL_SCALE * 25.0**2
        share = depth_stress_pa / (depth_stress_pa + inertial_stress_pa)
        assert 0.003 < share < 0.05

    def test_rejects_a_saturation_fraction_that_could_invert_the_depth_term(
        self,
    ) -> None:
        with pytest.raises(CalibrationError, match="invert the depth term"):
            CrossoverSaturatingDepression(saturation_fraction=1.0)
        with pytest.raises(CalibrationError):
            CrossoverSaturatingDepression(saturation_fraction=-0.1)

    def test_rejects_a_non_positive_inertial_scale(self) -> None:
        model = CrossoverSaturatingDepression()
        bad = DepressionInputs(
            depth_m=np.array([0.04]),
            normal_speed_m_s=np.array([10.0]),
            depth_stress_scale_pa_per_m=np.array([_DEPTH_SCALE_PA_PER_M]),
            inertial_stress_scale_pa_s2_per_m2=0.0,
        )
        with pytest.raises(CalibrationError, match="lambda\\*rho"):
            model.depression_m(bad)

    def test_falls_back_when_an_element_carries_no_depth_response(self) -> None:
        model = CrossoverSaturatingDepression()
        inputs = DepressionInputs(
            depth_m=np.array([0.04]),
            normal_speed_m_s=np.array([10.0]),
            depth_stress_scale_pa_per_m=np.array([0.0]),
            inertial_stress_scale_pa_s2_per_m2=_INERTIAL_SCALE,
        )
        depression = float(model.depression_m(inputs)[0])
        assert np.isfinite(depression)
        assert 0.0 <= depression < 0.04

    @settings(deadline=None, max_examples=120)
    @given(
        depth_m=st.floats(min_value=1e-4, max_value=0.20),
        slow_m_s=st.floats(min_value=0.0, max_value=40.0),
        faster_m_s=st.floats(min_value=0.0, max_value=40.0),
    )
    def test_effective_depth_never_decreases_with_geometric_depth(
        self, depth_m: float, slow_m_s: float, faster_m_s: float
    ) -> None:
        model = CrossoverSaturatingDepression()
        speed = max(slow_m_s, faster_m_s)
        shallow = depth_m
        deep = depth_m * 1.5
        effective_shallow = shallow - float(
            model.depression_m(_inputs(shallow, speed))[0]
        )
        effective_deep = deep - float(model.depression_m(_inputs(deep, speed))[0])
        assert effective_deep >= effective_shallow - 1e-15

    @settings(deadline=None, max_examples=120)
    @given(
        depth_m=st.floats(min_value=1e-4, max_value=0.20),
        speed_m_s=st.floats(min_value=0.0, max_value=40.0),
    )
    def test_depression_never_decreases_with_speed(
        self, depth_m: float, speed_m_s: float
    ) -> None:
        model = CrossoverSaturatingDepression()
        slow = float(model.depression_m(_inputs(depth_m, speed_m_s))[0])
        fast = float(model.depression_m(_inputs(depth_m, speed_m_s + 1.0))[0])
        assert fast >= slow - 1e-15
