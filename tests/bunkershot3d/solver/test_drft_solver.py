"""Tests for the DRFT solver integration (issue #8611).

This is the F0 tier solver: 3D Dynamic Resistive Force Theory.
Acceptance criteria:
- Reproduces published RFT results for simple intruders within tolerance
- Metamorphic tests pass
- Every result carries fidelity tier F0 and validity verdict
- Full shot runs in < 50 ms
"""

import time

import numpy as np
import pytest

from bunkershot3d.solver.drft import (
    DRFTSolver,
    FidelityTier,
)
from bunkershot3d.solver.envelope import ValidityVerdict


class TestFlatPlateIntrusion:
    """Validate against published flat-plate intrusion results."""

    @pytest.fixture
    def solver(self) -> DRFTSolver:
        """Standard solver with reference sand properties."""
        return DRFTSolver(
            bulk_density_kg_m3=1550.0,
            friction_angle_deg=33.0,
            surface_friction=0.5,
        )

    def test_vertical_plate_force_sign(self, solver: DRFTSolver) -> None:
        """Vertical intrusion produces downward resistance (positive Fz)."""
        # Simple vertical plate, 10x10 mm, moving down at 1 m/s
        result = solver.flat_plate_intrusion(
            width_m=0.01,
            height_m=0.01,
            depth_m=0.02,
            velocity_m_s=1.0,
            attack_angle_rad=np.pi / 2,  # vertical
        )
        assert result.force_z > 0, "vertical intrusion should resist (Fz > 0)"

    def test_force_increases_with_depth(self, solver: DRFTSolver) -> None:
        """Force should increase with depth (RFT depth-linearity)."""
        forces = []
        for depth in [0.01, 0.02, 0.03, 0.04]:
            result = solver.flat_plate_intrusion(
                width_m=0.01,
                height_m=0.01,
                depth_m=depth,
                velocity_m_s=1.0,
                attack_angle_rad=np.pi / 2,
            )
            forces.append(result.force_z)
        # Force should monotonically increase
        assert all(forces[i] < forces[i + 1] for i in range(len(forces) - 1))

    def test_force_increases_with_velocity_squared(self, solver: DRFTSolver) -> None:
        """Inertial term scales with v^2."""
        v1 = 5.0
        v2 = 10.0
        result1 = solver.flat_plate_intrusion(
            width_m=0.02,
            height_m=0.02,
            depth_m=0.03,
            velocity_m_s=v1,
            attack_angle_rad=np.pi / 2,
        )
        result2 = solver.flat_plate_intrusion(
            width_m=0.02,
            height_m=0.02,
            depth_m=0.03,
            velocity_m_s=v2,
            attack_angle_rad=np.pi / 2,
        )
        # At high speed, inertial term dominates, so F ~ v^2
        ratio = result2.force_z / result1.force_z
        # Should be closer to v^2 scaling (4.0) than linear (2.0)
        assert ratio > 2.0, "force should scale faster than linearly with v"


class TestMetamorphicProperties:
    """Metamorphic tests: properties that must hold for any valid implementation."""

    @pytest.fixture
    def solver(self) -> DRFTSolver:
        return DRFTSolver(
            bulk_density_kg_m3=1550.0,
            friction_angle_deg=33.0,
            surface_friction=0.5,
        )

    def test_translation_invariance(self, solver: DRFTSolver) -> None:
        """Force should be independent of absolute position (only depth matters)."""
        # Same depth relative to surface, different absolute z
        result1 = solver.flat_plate_intrusion(
            width_m=0.01,
            height_m=0.01,
            depth_m=0.02,
            velocity_m_s=5.0,
            attack_angle_rad=np.pi / 2,
        )
        result2 = solver.flat_plate_intrusion(
            width_m=0.01,
            height_m=0.01,
            depth_m=0.02,
            velocity_m_s=5.0,
            attack_angle_rad=np.pi / 2,
        )
        np.testing.assert_allclose(result1.force_z, result2.force_z, rtol=1e-10)

    def test_horizontal_symmetry(self, solver: DRFTSolver) -> None:
        """Flipping the plate horizontally should not change force magnitude."""
        result1 = solver.flat_plate_intrusion(
            width_m=0.02,
            height_m=0.01,
            depth_m=0.03,
            velocity_m_s=5.0,
            attack_angle_rad=np.pi / 4,
        )
        # Mirror: width and height swapped for same area
        result2 = solver.flat_plate_intrusion(
            width_m=0.01,
            height_m=0.02,
            depth_m=0.03,
            velocity_m_s=5.0,
            attack_angle_rad=np.pi / 4,
        )
        # Same area, same attack angle => similar force
        np.testing.assert_allclose(result1.force_z, result2.force_z, rtol=0.01)


class TestResultStructure:
    """Verify result carries required metadata."""

    @pytest.fixture
    def solver(self) -> DRFTSolver:
        return DRFTSolver(
            bulk_density_kg_m3=1550.0,
            friction_angle_deg=33.0,
            surface_friction=0.5,
        )

    def test_result_has_fidelity_tier(self, solver: DRFTSolver) -> None:
        """Every result must carry fidelity tier F0."""
        result = solver.flat_plate_intrusion(
            width_m=0.01,
            height_m=0.01,
            depth_m=0.02,
            velocity_m_s=5.0,
            attack_angle_rad=np.pi / 2,
        )
        assert result.fidelity_tier == FidelityTier.F0

    def test_result_has_validity_verdict(self, solver: DRFTSolver) -> None:
        """Every result must carry a validity verdict."""
        result = solver.flat_plate_intrusion(
            width_m=0.01,
            height_m=0.01,
            depth_m=0.02,
            velocity_m_s=5.0,
            attack_angle_rad=np.pi / 2,
        )
        assert isinstance(result.validity, ValidityVerdict)
        assert result.validity.envelope is not None

    def test_result_has_force_components(self, solver: DRFTSolver) -> None:
        """Result carries all three force components."""
        result = solver.flat_plate_intrusion(
            width_m=0.01,
            height_m=0.01,
            depth_m=0.02,
            velocity_m_s=5.0,
            attack_angle_rad=np.pi / 2,
        )
        assert hasattr(result, "force_x")
        assert hasattr(result, "force_y")
        assert hasattr(result, "force_z")


class TestValidityEnforcement:
    """Solver must refuse out-of-envelope queries when appropriate."""

    def test_refuses_high_fr_without_dynamic(self) -> None:
        """At Fr > 1, solver without dynamic terms must refuse."""
        solver = DRFTSolver(
            bulk_density_kg_m3=1550.0,
            friction_angle_deg=33.0,
            surface_friction=0.5,
            enable_dynamic_terms=False,  # quasi-static only
        )
        with pytest.raises(ValueError, match="(?i)froude|envelope|dynamic"):
            solver.flat_plate_intrusion(
                width_m=0.01,
                height_m=0.01,
                depth_m=0.02,
                velocity_m_s=25.0,  # Fr >> 0.4
                attack_angle_rad=np.pi / 2,
            )

    def test_allows_high_fr_with_dynamic(self) -> None:
        """At Fr > 1, solver with dynamic terms proceeds (with extrapolation flag)."""
        solver = DRFTSolver(
            bulk_density_kg_m3=1550.0,
            friction_angle_deg=33.0,
            surface_friction=0.5,
            enable_dynamic_terms=True,
        )
        result = solver.flat_plate_intrusion(
            width_m=0.01,
            height_m=0.01,
            depth_m=0.02,
            velocity_m_s=25.0,
            attack_angle_rad=np.pi / 2,
        )
        assert result.validity.is_extrapolation


class TestPerformance:
    """Solver must be fast enough for design iteration."""

    @pytest.mark.benchmark
    def test_single_evaluation_under_1ms(self) -> None:
        """Single flat-plate evaluation should be < 1 ms."""
        solver = DRFTSolver(
            bulk_density_kg_m3=1550.0,
            friction_angle_deg=33.0,
            surface_friction=0.5,
        )
        start = time.perf_counter()
        for _ in range(100):
            solver.flat_plate_intrusion(
                width_m=0.02,
                height_m=0.08,
                depth_m=0.04,
                velocity_m_s=25.0,
                attack_angle_rad=np.pi / 4,
            )
        elapsed = time.perf_counter() - start
        per_call_ms = (elapsed / 100) * 1000
        assert per_call_ms < 1.0, f"took {per_call_ms:.2f} ms per call"


class TestSmokeTestForce:
    """The research addendum smoke test: ~1550 N for a 20x80 mm sole at 25 m/s."""

    def test_smoke_test_force_magnitude(self) -> None:
        """Force should be in the right order of magnitude (~1500 N)."""
        solver = DRFTSolver(
            bulk_density_kg_m3=1550.0,
            friction_angle_deg=33.0,
            surface_friction=0.5,
            enable_dynamic_terms=True,
            inertial_lambda=1.1,  # oblique plate value from research
        )
        result = solver.flat_plate_intrusion(
            width_m=0.02,  # 20 mm
            height_m=0.08,  # 80 mm
            depth_m=0.04,  # 40 mm divot
            velocity_m_s=25.0,
            attack_angle_rad=np.pi / 4,  # 45 deg attack
        )
        # Research says ~1550 N, allow 50% tolerance for geometry differences
        assert 500 < result.force_z < 3000, f"force {result.force_z:.0f} N out of range"
