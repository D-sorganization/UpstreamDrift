"""Effective-loft / effective-bounce trigonometry tests (issue #8609).

The exact relation (rotation of the rigid head about the shaft axis by
Omega, lie angle lambda measured shaft-to-ground) is

    L_eff = arcsin[ sin L cos(Om)
                  + cos L cos(lam) sin(Om)
                  + sin^2(lam) sin L (1 - cos Om) ]

with the first-order decomposition dloft ~ dbounce ~ Om cos(lam) and
daim ~ Om sin(lam).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.geometry.bounce import GeometricBounce, MarketedBounce
from bunkershot3d.geometry.delivery import (
    DeliveryCondition,
    aim_offset_deg,
    aim_offset_first_order_deg,
    deliver_wedge,
    delivered_face_normal,
    effective_bounce_deg,
    effective_loft_closed_form_deg,
    effective_loft_deg,
    effective_loft_first_order_deg,
)
from bunkershot3d.geometry.wedge import WedgeGeometry

pytestmark = pytest.mark.unit


class TestWorkedCase:
    """56 deg loft, 64 deg lie, Omega = 20 deg -> L_eff = 64.5 deg."""

    def test_exact_effective_loft(self) -> None:
        loft_eff = effective_loft_deg(loft_deg=56.0, lie_deg=64.0, face_open_deg=20.0)
        # The research digest quotes 64.5 deg; the closed form evaluates to
        # 64.591 deg, so the digest's figure is a rounded restatement.
        assert loft_eff == pytest.approx(64.591, abs=0.005)
        assert abs(loft_eff - 64.5) < 0.1

    def test_the_gain_is_8_point_5_degrees_not_20(self) -> None:
        gain = (
            effective_loft_deg(loft_deg=56.0, lie_deg=64.0, face_open_deg=20.0) - 56.0
        )
        assert gain == pytest.approx(8.591, abs=0.005)
        assert 8.0 < gain < 9.0

    def test_first_order_matches_the_digest(self) -> None:
        approx = effective_loft_first_order_deg(
            loft_deg=56.0, lie_deg=64.0, face_open_deg=20.0
        )
        assert approx == pytest.approx(56.0 + 20.0 * math.cos(math.radians(64.0)))
        assert approx == pytest.approx(64.8, abs=0.05)

    def test_first_order_is_within_half_a_degree_of_exact(self) -> None:
        exact = effective_loft_deg(loft_deg=56.0, lie_deg=64.0, face_open_deg=20.0)
        approx = effective_loft_first_order_deg(
            loft_deg=56.0, lie_deg=64.0, face_open_deg=20.0
        )
        assert abs(exact - approx) < 0.5

    def test_first_order_aim(self) -> None:
        aim = aim_offset_first_order_deg(lie_deg=64.0, face_open_deg=20.0)
        assert aim == pytest.approx(18.0, abs=0.05)


class TestLimitingCases:
    @pytest.mark.parametrize("loft", [46.0, 52.0, 56.0, 60.0])
    @pytest.mark.parametrize("lie", [55.0, 64.0, 70.0])
    def test_zero_open_returns_the_static_loft(self, loft: float, lie: float) -> None:
        assert effective_loft_deg(
            loft_deg=loft, lie_deg=lie, face_open_deg=0.0
        ) == pytest.approx(loft, abs=1e-12)

    @pytest.mark.parametrize("omega", [0.0, 5.0, 20.0, 45.0])
    def test_vertical_shaft_only_changes_aim(self, omega: float) -> None:
        assert effective_loft_deg(
            loft_deg=56.0, lie_deg=90.0, face_open_deg=omega
        ) == pytest.approx(56.0, abs=1e-10)
        assert aim_offset_deg(
            loft_deg=56.0, lie_deg=90.0, face_open_deg=omega
        ) == pytest.approx(omega, abs=1e-10)

    @pytest.mark.parametrize("omega", [0.0, 5.0, 20.0])
    def test_horizontal_shaft_adds_loft_degree_for_degree(self, omega: float) -> None:
        assert effective_loft_deg(
            loft_deg=40.0, lie_deg=0.0, face_open_deg=omega
        ) == pytest.approx(40.0 + omega, abs=1e-10)
        assert aim_offset_deg(
            loft_deg=40.0, lie_deg=0.0, face_open_deg=omega
        ) == pytest.approx(0.0, abs=1e-10)


class TestExactMatchesRotation:
    """The arcsin closed form must equal the rigid-body rotation exactly."""

    @pytest.mark.parametrize("loft", [46.0, 56.0, 62.0])
    @pytest.mark.parametrize("lie", [0.0, 30.0, 64.0, 90.0])
    @pytest.mark.parametrize("omega", [-15.0, 0.0, 7.5, 20.0, 40.0])
    def test_closed_form_equals_rodrigues(
        self, loft: float, lie: float, omega: float
    ) -> None:
        normal = delivered_face_normal(
            loft_deg=loft, lie_deg=lie, face_open_deg=omega, shaft_lean_deg=0.0
        )
        from_rotation = math.degrees(math.asin(float(np.clip(normal[2], -1.0, 1.0))))
        closed_form = effective_loft_closed_form_deg(
            loft_deg=loft, lie_deg=lie, face_open_deg=omega
        )
        assert from_rotation == pytest.approx(closed_form, abs=1e-11)
        assert effective_loft_deg(
            loft_deg=loft, lie_deg=lie, face_open_deg=omega
        ) == pytest.approx(closed_form, abs=1e-11)

    def test_face_normal_is_a_unit_vector(self) -> None:
        normal = delivered_face_normal(
            loft_deg=56.0, lie_deg=64.0, face_open_deg=20.0, shaft_lean_deg=8.0
        )
        assert float(np.linalg.norm(normal)) == pytest.approx(1.0, abs=1e-12)


class TestShaftLean:
    @pytest.mark.parametrize("lean", [0.0, 4.0, 9.0, 14.0])
    def test_lean_subtracts_loft_degree_for_degree(self, lean: float) -> None:
        loft_eff = effective_loft_deg(
            loft_deg=56.0, lie_deg=64.0, face_open_deg=0.0, shaft_lean_deg=lean
        )
        assert loft_eff == pytest.approx(56.0 - lean, abs=1e-10)

    @pytest.mark.parametrize("lean", [0.0, 4.0, 9.0, 14.0])
    def test_lean_subtracts_bounce_degree_for_degree(self, lean: float) -> None:
        bounce = effective_bounce_deg(
            bounce_deg=12.0, lie_deg=64.0, face_open_deg=0.0, shaft_lean_deg=lean
        )
        assert bounce == pytest.approx(12.0 - lean, abs=1e-10)

    def test_low_bounce_is_consumed_by_tour_lean(self) -> None:
        bounce = effective_bounce_deg(
            bounce_deg=4.0, lie_deg=64.0, face_open_deg=0.0, shaft_lean_deg=10.0
        )
        assert bounce < 0.0


class TestEffectiveBounce:
    def test_opening_adds_bounce_at_the_first_order_rate(self) -> None:
        gained = (
            effective_bounce_deg(
                bounce_deg=10.0, lie_deg=64.0, face_open_deg=20.0, shaft_lean_deg=0.0
            )
            - 10.0
        )
        assert gained == pytest.approx(20.0 * math.cos(math.radians(64.0)), abs=0.6)

    @pytest.mark.parametrize("omega", [0.0, 10.0, 20.0])
    def test_vertical_shaft_does_not_change_bounce(self, omega: float) -> None:
        assert effective_bounce_deg(
            bounce_deg=10.0, lie_deg=90.0, face_open_deg=omega, shaft_lean_deg=0.0
        ) == pytest.approx(10.0, abs=1e-9)

    @pytest.mark.parametrize("omega", [0.0, 10.0, 20.0])
    def test_horizontal_shaft_adds_bounce_degree_for_degree(self, omega: float) -> None:
        assert effective_bounce_deg(
            bounce_deg=10.0, lie_deg=0.0, face_open_deg=omega, shaft_lean_deg=0.0
        ) == pytest.approx(10.0 + omega, abs=1e-9)


class TestDeliveredGeometry:
    def test_delivery_preserves_the_bounce_convention(
        self, wedge: WedgeGeometry
    ) -> None:
        condition = DeliveryCondition(
            face_open_deg=20.0, shaft_lean_deg=8.0, attack_angle_deg=-6.0
        )
        marketed = deliver_wedge(wedge, condition)
        assert isinstance(marketed.effective_bounce, MarketedBounce)

        geometric = deliver_wedge(wedge, condition, use_geometric_bounce=True)
        assert isinstance(geometric.effective_bounce, GeometricBounce)
        assert (
            geometric.effective_bounce.angle_deg > marketed.effective_bounce.angle_deg
        )

    def test_presentation_matches_the_digest_first_order_sum(
        self, wedge: WedgeGeometry
    ) -> None:
        condition = DeliveryCondition(
            face_open_deg=20.0, shaft_lean_deg=8.0, attack_angle_deg=-6.0
        )
        delivered = deliver_wedge(wedge, condition)
        digest = (
            wedge.marketed_bounce.angle_deg
            - 8.0
            + 20.0 * math.cos(math.radians(wedge.lie_deg))
            - 6.0
        )
        assert delivered.presentation_bounce_deg == pytest.approx(digest, abs=0.6)

    def test_a_steep_attack_angle_can_flip_the_sole_into_digging(
        self, wedge: WedgeGeometry
    ) -> None:
        skidding = deliver_wedge(
            wedge, DeliveryCondition(face_open_deg=20.0, attack_angle_deg=-4.0)
        )
        digging = deliver_wedge(
            wedge, DeliveryCondition(face_open_deg=0.0, attack_angle_deg=-12.0)
        )
        assert skidding.presentation_bounce_deg > 0.0
        assert digging.presentation_bounce_deg < skidding.presentation_bounce_deg

    def test_delivered_loft_and_aim_are_reported(self, wedge: WedgeGeometry) -> None:
        delivered = deliver_wedge(
            wedge,
            DeliveryCondition(
                face_open_deg=20.0, shaft_lean_deg=0.0, attack_angle_deg=0.0
            ),
        )
        assert delivered.effective_loft_deg == pytest.approx(64.591, abs=0.005)
        assert delivered.aim_offset_deg > 15.0

    def test_condition_validation(self) -> None:
        with pytest.raises(ValueError):
            DeliveryCondition(face_open_deg=float("nan"))
        with pytest.raises(ValueError):
            DeliveryCondition(attack_angle_deg=120.0)
