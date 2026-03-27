"""Tests for critical physics and constants fixes.

Covers:
    #2157 - Cone inertia about COM (not apex)
    #2158 - Unified gravity constant (9.80665 m/s^2)
    #2159 - Humanoid segment mass ratios sum to 1.0
    #2160 - Graphite density consistency (1750 kg/m^3)
    #2161 - ZMP fallback mass matches DEFAULT_MASS_KG (75 kg)
"""

from __future__ import annotations

import math

import pytest


class TestConeInertiaCOM:
    """#2157: Cone inertia must be computed about center of mass."""

    def test_cone_axial_inertia(self) -> None:
        """Axial inertia should be (3/10)*m*r^2 (same for apex and COM)."""
        from src.shared.python.model_generation.inertia.primitives import cone_inertia

        result = cone_inertia(mass=10.0, radius=0.5, height=1.0, axis="z")
        expected_axial = (3.0 / 10.0) * 10.0 * 0.5**2
        assert result["izz"] == pytest.approx(expected_axial, rel=1e-10)

    def test_cone_perpendicular_inertia_com(self) -> None:
        """Perpendicular inertia about COM: m*(3r^2/20 + 3h^2/80)."""
        from src.shared.python.model_generation.inertia.primitives import cone_inertia

        m, r, h = 10.0, 0.5, 1.0
        result = cone_inertia(mass=m, radius=r, height=h, axis="z")
        expected_perp = m * ((3.0 / 20.0) * r**2 + (3.0 / 80.0) * h**2)
        assert result["ixx"] == pytest.approx(expected_perp, rel=1e-10)
        assert result["iyy"] == pytest.approx(expected_perp, rel=1e-10)

    def test_cone_perp_less_than_apex_formula(self) -> None:
        """COM inertia must be less than the old apex inertia (3/5*h^2 term)."""
        from src.shared.python.model_generation.inertia.primitives import cone_inertia

        m, r, h = 10.0, 0.5, 2.0
        result = cone_inertia(mass=m, radius=r, height=h, axis="z")
        apex_perp = m * ((3.0 / 20.0) * r**2 + (3.0 / 5.0) * h**2)
        # COM perpendicular inertia must be strictly less than apex
        assert result["ixx"] < apex_perp

    def test_cone_off_diagonal_zero(self) -> None:
        """Off-diagonal elements should be zero for symmetric cone."""
        from src.shared.python.model_generation.inertia.primitives import cone_inertia

        result = cone_inertia(mass=5.0, radius=1.0, height=2.0)
        assert result["ixy"] == 0.0
        assert result["ixz"] == 0.0
        assert result["iyz"] == 0.0

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_cone_axis_symmetry(self, axis: str) -> None:
        """Axial moment should be assigned to the correct axis."""
        from src.shared.python.model_generation.inertia.primitives import cone_inertia

        result = cone_inertia(mass=10.0, radius=0.5, height=1.0, axis=axis)
        expected_axial = (3.0 / 10.0) * 10.0 * 0.5**2
        key = f"i{axis}{axis}"
        assert result[key] == pytest.approx(expected_axial, rel=1e-10)


class TestGravityConstantUnification:
    """#2158: All gravity constants must be 9.80665 m/s^2."""

    def test_physics_constants_gravity(self) -> None:
        from src.shared.python.core.physics_constants import GRAVITY_M_S2

        assert float(GRAVITY_M_S2) == pytest.approx(9.80665, abs=1e-10)

    def test_constants_gravity(self) -> None:
        from src.shared.python.core.constants import GRAVITY

        assert GRAVITY == pytest.approx(9.80665, abs=1e-5)

    def test_constants_gravity_float(self) -> None:
        from src.shared.python.core.constants import GRAVITY_FLOAT

        assert GRAVITY_FLOAT == pytest.approx(9.80665, abs=1e-10)

    def test_pendulum_gravity_mss(self) -> None:
        from src.shared.python.pendulum_simulator.constants import GRAVITY_MSS

        assert GRAVITY_MSS == pytest.approx(9.80665, abs=1e-10)

    def test_pendulum_gravity_standard(self) -> None:
        from src.shared.python.pendulum_simulator.constants import GRAVITY_STANDARD

        assert GRAVITY_STANDARD == pytest.approx(9.80665, abs=1e-10)

    def test_model_generation_gravity(self) -> None:
        from src.shared.python.model_generation.core.constants import GRAVITY_M_S2

        assert GRAVITY_M_S2 == pytest.approx(9.80665, abs=1e-10)

    def test_engines_common_standard_gravity(self) -> None:
        from src.engines.common.physics import STANDARD_GRAVITY

        assert STANDARD_GRAVITY == pytest.approx(9.80665, abs=1e-10)

    def test_engines_common_gravity_approx(self) -> None:
        from src.engines.common.physics import GRAVITY_APPROX

        assert GRAVITY_APPROX == pytest.approx(9.80665, abs=1e-10)


class TestSegmentMassRatios:
    """#2159: Humanoid segment mass ratios must sum to 1.0."""

    def test_mass_ratios_sum_to_one(self) -> None:
        from src.shared.python.humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        total = sum(seg.mass_ratio for seg in HUMANOID_SEGMENTS.values())
        assert total == pytest.approx(1.0, abs=1e-6), (
            f"Segment mass ratios sum to {total}, expected 1.0"
        )

    def test_all_mass_ratios_positive(self) -> None:
        from src.shared.python.humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        for name, seg in HUMANOID_SEGMENTS.items():
            assert seg.mass_ratio > 0, f"Segment '{name}' has non-positive mass ratio"

    def test_bilateral_symmetry(self) -> None:
        """Left and right segments should have equal mass ratios."""
        from src.shared.python.humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        for name, seg in HUMANOID_SEGMENTS.items():
            if name.startswith("left_"):
                right_name = name.replace("left_", "right_")
                right_seg = HUMANOID_SEGMENTS.get(right_name)
                assert right_seg is not None, f"Missing mirror segment: {right_name}"
                assert seg.mass_ratio == right_seg.mass_ratio, (
                    f"Asymmetric mass: {name}={seg.mass_ratio} vs {right_name}={right_seg.mass_ratio}"
                )


class TestGraphiteDensity:
    """#2160: Graphite density must be consistent across modules."""

    def test_physics_constants_graphite(self) -> None:
        from src.shared.python.core.physics_constants import GRAPHITE_DENSITY_KG_M3

        assert float(GRAPHITE_DENSITY_KG_M3) == 1750

    def test_flexible_shaft_graphite(self) -> None:
        from src.shared.python.physics.flexible_shaft import GRAPHITE_DENSITY

        assert GRAPHITE_DENSITY == 1750

    def test_consistency(self) -> None:
        from src.shared.python.core.physics_constants import GRAPHITE_DENSITY_KG_M3
        from src.shared.python.physics.flexible_shaft import GRAPHITE_DENSITY

        assert GRAPHITE_DENSITY == float(GRAPHITE_DENSITY_KG_M3)


class TestZMPFallbackMass:
    """#2161: ZMP fallback mass must match DEFAULT_MASS_KG."""

    def test_fallback_mass_equals_default(self) -> None:
        from unittest.mock import MagicMock

        from src.robotics.locomotion.zmp_computer import ZMPComputer

        mock_engine = MagicMock()
        mock_engine.get_total_mass = MagicMock(return_value=75.0)
        computer = ZMPComputer(engine=mock_engine)
        # Access private method to test fallback
        mass = computer._estimate_mass()
        # Non-humanoid engine should return fallback
        computer2 = ZMPComputer(engine=MagicMock(spec=[]))
        fallback = computer2._estimate_mass()
        assert fallback == 75.0

    def test_fallback_matches_model_generation_default(self) -> None:
        from unittest.mock import MagicMock

        from src.robotics.locomotion.zmp_computer import ZMPComputer
        from src.shared.python.model_generation.core.constants import DEFAULT_MASS_KG

        computer = ZMPComputer(engine=MagicMock(spec=[]))
        assert computer._estimate_mass() == DEFAULT_MASS_KG
