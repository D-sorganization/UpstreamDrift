"""Capability register tests (issue #9688, ADR-0044).

Verifies the fail-closed sand-motion capability register so no caller can
present F0, F1, proxy, or decorative tracer output as genuine 3-D individual-grain
trajectories or spherical ball spin.
"""

from __future__ import annotations

import pytest

from bunkershot3d.solvers.capability import (
    SAND_MOTION_CAPABILITIES,
    SandMotionCapability,
    SandMotionKind,
    capability,
    require_ball_spin_3d,
    require_grain_trajectories_3d,
    require_physical,
)
from bunkershot3d.solvers.exceptions import (
    CapabilityError,
    OutOfEnvelopeError,
    SolverError,
)
from bunkershot3d.solvers.protocol import FidelityTier

pytestmark = pytest.mark.unit


class TestSandMotionCapabilityRegister:
    """Acceptance tests for the fail-closed capability register."""

    def test_capability_f1_has_no_3d_grain_trajectories(self) -> None:
        cap = capability("F1")
        assert cap.grain_trajectories_3d is False
        assert cap.kind is SandMotionKind.CONTINUUM_MATERIAL_POINTS
        assert cap.spatial_dims == 2

    @pytest.mark.parametrize(
        "pathway",
        [
            "F0",
            "F1",
            "sandvolume_extruded",
            "backends.mpm_proxy",
            "backends.chrono",
        ],
    )
    def test_require_grain_trajectories_3d_raises_for_current_pathways(
        self, pathway: str
    ) -> None:
        with pytest.raises(CapabilityError) as exc_info:
            require_grain_trajectories_3d(pathway)
        message = str(exc_info.value)
        assert pathway in message
        assert "#9688" in message
        assert "ADR-0044" in message
        # Verify the kind name is in the message
        cap = capability(pathway)
        assert cap.kind.value in message

    def test_require_ball_spin_3d_raises_for_f1(self) -> None:
        with pytest.raises(CapabilityError) as exc_info:
            require_ball_spin_3d("F1")
        message = str(exc_info.value)
        assert "F1" in message
        assert "#9688" in message
        assert "ADR-0044" in message
        assert SandMotionKind.CONTINUUM_MATERIAL_POINTS.value in message

    def test_require_physical_raises_for_tracers(self) -> None:
        with pytest.raises(CapabilityError) as exc_info:
            require_physical("tracers")
        message = str(exc_info.value)
        assert "tracers" in message
        assert "#9688" in message
        assert "ADR-0044" in message
        assert SandMotionKind.DECORATIVE_TRACERS.value in message

    def test_require_helpers_return_capability_when_satisfied(self) -> None:
        cap_phys = require_physical("F0")
        assert isinstance(cap_phys, SandMotionCapability)
        assert cap_phys.physical is True

        cap_f1_phys = require_physical("F1")
        assert cap_f1_phys.physical is True


class TestSandMotionCapabilityInvariants:
    """Construction invariants enforced by SandMotionCapability.__post_init__."""

    def test_grain_trajectories_3d_with_spatial_dims_2_raises(self) -> None:
        with pytest.raises(ValueError, match="spatial_dims == 3"):
            SandMotionCapability(
                pathway="invalid_2d_grains",
                kind=SandMotionKind.DISCRETE_GRAINS,
                spatial_dims=2,
                grain_trajectories_3d=True,
                ball_spin_3d=False,
                physical=True,
                notes="cannot have 3d trajectories in 2d",
            )

    def test_grain_trajectories_3d_with_non_discrete_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="requires kind in"):
            SandMotionCapability(
                pathway="invalid_continuum_grains",
                kind=SandMotionKind.CONTINUUM_MATERIAL_POINTS,
                spatial_dims=3,
                grain_trajectories_3d=True,
                ball_spin_3d=False,
                physical=True,
                notes="continuum material points are not discrete grains",
            )

    def test_decorative_tracers_with_physical_true_raises(self) -> None:
        with pytest.raises(ValueError, match="requires physical is False"):
            SandMotionCapability(
                pathway="invalid_tracers",
                kind=SandMotionKind.DECORATIVE_TRACERS,
                spatial_dims=3,
                grain_trajectories_3d=False,
                ball_spin_3d=False,
                physical=True,
                notes="decorative tracers are not physical",
            )

    def test_spatial_dims_outside_allowed_set_raises(self) -> None:
        with pytest.raises(ValueError, match=r"spatial_dims must be in \{0, 2, 3\}"):
            SandMotionCapability(
                pathway="invalid_4d",
                kind=SandMotionKind.DISCRETE_GRAINS,
                spatial_dims=4,
                grain_trajectories_3d=False,
                ball_spin_3d=False,
                physical=True,
                notes="4d is not supported",
            )

    def test_none_kind_requires_zero_spatial_dims_and_no_trajectories(self) -> None:
        with pytest.raises(ValueError, match="requires spatial_dims == 0"):
            SandMotionCapability(
                pathway="invalid_none_dims",
                kind=SandMotionKind.NONE,
                spatial_dims=2,
                grain_trajectories_3d=False,
                ball_spin_3d=False,
                physical=True,
                notes="none cannot have spatial dims",
            )

        with pytest.raises(ValueError, match="no trajectories"):
            SandMotionCapability(
                pathway="invalid_none_traj",
                kind=SandMotionKind.NONE,
                spatial_dims=0,
                grain_trajectories_3d=True,
                ball_spin_3d=False,
                physical=True,
                notes="none cannot have trajectories",
            )


class TestRegisterStructureAndIntegrity:
    """Register immutability and completeness."""

    def test_register_is_immutable(self) -> None:
        with pytest.raises(TypeError):
            SAND_MOTION_CAPABILITIES["custom"] = None  # type: ignore[index]

        with pytest.raises(TypeError):
            del SAND_MOTION_CAPABILITIES["F0"]  # type: ignore[misc]

    def test_every_fidelity_tier_member_is_registered(self) -> None:
        for tier in FidelityTier:
            assert tier.value in SAND_MOTION_CAPABILITIES
            cap = capability(tier.value)
            assert cap.pathway == tier.value
            # Both StrEnum member and raw string resolve to the same capability
            assert capability(tier) == cap

    def test_unknown_pathway_error_message_lists_known_ids(self) -> None:
        with pytest.raises(CapabilityError) as exc_info:
            capability("nonexistent_pathway_xyz")
        message = str(exc_info.value)
        assert "nonexistent_pathway_xyz" in message
        for pathway_id in SAND_MOTION_CAPABILITIES:
            assert pathway_id in message

    def test_capability_error_inheritance(self) -> None:
        err = CapabilityError("test message")
        assert isinstance(err, OutOfEnvelopeError)
        assert isinstance(err, ValueError)
        assert isinstance(err, SolverError)
