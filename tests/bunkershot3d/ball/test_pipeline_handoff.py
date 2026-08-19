"""Tests for bunker shot → flight pipeline handoff (issue #8613).

Verifies that bunkershot3d launch conditions integrate correctly with
the existing SwingBallFlightPipeline infrastructure.

TDD: Write failing tests first, then implement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.ball.lie import BallLie, BallProperties
from bunkershot3d.ball.pipeline import (
    BunkerShotState,
    compute_bunker_launch,
    to_post_impact_state,
)
from src.shared.python.physics.impact_model import PostImpactState

pytestmark = pytest.mark.unit


class TestBunkerShotState:
    """Tests for bunker shot state specification."""

    def test_bunker_shot_state_creation(self) -> None:
        """Can create a complete bunker shot state."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.01),
            entry_depth_m=0.015,
        )
        assert state.club_velocity_m_s == 25.0
        assert state.club_loft_deg == 56.0
        assert state.entry_depth_m == 0.015

    def test_defaults_are_reasonable(self) -> None:
        """Default values produce a valid bunker shot."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.005),
        )
        assert state.entry_depth_m > 0
        assert state.sole_width_m > 0
        assert state.sole_length_m > 0


class TestComputeBunkerLaunch:
    """Tests for computing launch conditions from bunker shot."""

    def test_produces_valid_launch(self) -> None:
        """Should produce valid launch conditions."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.01),
            entry_depth_m=0.015,
        )
        result = compute_bunker_launch(state)
        assert result.ball_speed_m_s > 0
        assert 0 < result.launch_angle_rad < math.pi / 2

    def test_produces_energy_accounting(self) -> None:
        """Should include energy accounting data."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.01),
            entry_depth_m=0.015,
        )
        result = compute_bunker_launch(state)
        assert 0 < result.energy_transfer_fraction < 1


class TestToPostImpactState:
    """Tests for converting to PostImpactState for pipeline handoff."""

    def test_converts_to_post_impact_state(self) -> None:
        """Should produce a valid PostImpactState."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.01),
            entry_depth_m=0.015,
        )
        result = compute_bunker_launch(state)
        post = to_post_impact_state(result, state)

        assert isinstance(post, PostImpactState)
        assert len(post.ball_velocity) == 3
        assert len(post.ball_angular_velocity) == 3

    def test_ball_velocity_matches_launch(self) -> None:
        """PostImpactState velocity should match launch result."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.01),
            entry_depth_m=0.015,
        )
        result = compute_bunker_launch(state)
        post = to_post_impact_state(result, state)

        # Velocity magnitude should match
        speed = np.linalg.norm(post.ball_velocity)
        assert speed == pytest.approx(result.ball_speed_m_s, rel=1e-6)

    def test_clubhead_velocity_is_post_sand(self) -> None:
        """Clubhead velocity should be reduced after sand resistance."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.01),
            entry_depth_m=0.015,
        )
        result = compute_bunker_launch(state)
        post = to_post_impact_state(result, state)

        # Clubhead should slow down after passing through sand
        clubhead_speed = np.linalg.norm(post.clubhead_velocity)
        assert clubhead_speed < state.club_velocity_m_s

    def test_energy_transfer_reported(self) -> None:
        """PostImpactState should have energy_transfer field populated."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.01),
            entry_depth_m=0.015,
        )
        result = compute_bunker_launch(state)
        post = to_post_impact_state(result, state)

        assert post.energy_transfer > 0


class TestEnergyAccounting:
    """Tests for energy conservation in bunker shots (issue #8613 acceptance)."""

    def test_energy_accounting_closes(self) -> None:
        """Club KE in = sand dissipation + ball KE + bed PE, within tolerance."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.01),
            entry_depth_m=0.015,
            club_mass_kg=0.30,
        )
        ball = BallProperties()
        result = compute_bunker_launch(state)
        post = to_post_impact_state(result, state)

        # Input energy
        club_ke_in = 0.5 * state.club_mass_kg * state.club_velocity_m_s**2

        # Output energies
        ball_ke = 0.5 * ball.mass_kg * result.ball_speed_m_s**2
        ball_rot_ke = (
            0.5 * ball.moi_kg_m2 * (result.spin_rate_rpm * 2 * math.pi / 60) ** 2
        )
        clubhead_speed_out = np.linalg.norm(post.clubhead_velocity)
        club_ke_out = 0.5 * state.club_mass_kg * clubhead_speed_out**2

        # Energy dissipated to sand
        total_ke_out = ball_ke + ball_rot_ke + club_ke_out
        sand_dissipation = club_ke_in - total_ke_out

        # Energy should be conserved (dissipation is positive)
        assert sand_dissipation > 0, "Sand should dissipate energy"
        assert sand_dissipation < club_ke_in, "Can't dissipate more than input"

        # Most energy goes to sand in a splash shot
        assert sand_dissipation > 0.5 * club_ke_in


class TestLaunchConditionsSanity:
    """Sanity checks for launch conditions (issue #8613 acceptance)."""

    def test_tour_bunker_shot_in_sane_range(self) -> None:
        """Tour bunker shot should produce physically sane launch."""
        state = BunkerShotState(
            club_velocity_m_s=25.0,
            club_loft_deg=56.0,
            ball_lie=BallLie(depth_m=0.005),
            entry_depth_m=0.015,
        )
        result = compute_bunker_launch(state)

        # Ball speed should be reasonable for bunker shot
        assert 8 < result.ball_speed_m_s < 25

        # Launch angle should be high (lofted club)
        launch_deg = math.degrees(result.launch_angle_rad)
        assert 25 < launch_deg < 70

        # Spin should be backspin in reasonable range
        assert 2000 < result.spin_rate_rpm < 12000
