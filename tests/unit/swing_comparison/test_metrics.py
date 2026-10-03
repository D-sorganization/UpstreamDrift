"""Unit tests for swing metrics calculation (Issue #11164)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.swing_comparison.events import SwingEvents
from src.shared.python.swing_comparison.metrics import (
    ClubMetrics,
    HandPathMetrics,
    KinematicSequenceMetrics,
    LeadArmMetrics,
    SegmentRotationMetrics,
    SwingMetrics,
    TempoMetrics,
    WristMetrics,
    compute_all_metrics,
    compute_club_metrics,
    compute_hand_path_metrics,
    compute_kinematic_sequence,
    compute_lead_arm_metrics,
    compute_segment_rotations,
    compute_tempo,
    compute_wrist_metrics,
)
from src.shared.python.swing_comparison.motion import (
    CAPTURE_A_MARKER_LABELS,
    SwingMotion,
    swing_motion_from_markers,
)


@pytest.mark.unit
class TestTempoMetrics:
    """Test suite for swing tempo metrics."""

    def test_tempo_analytic_3_to_1(self) -> None:
        """Verify backswing time, downswing time, and 3:1 tempo ratio."""
        events = SwingEvents(
            address_idx=10,
            address_time=0.10,
            top_idx=100,
            top_time=1.00,
            impact_idx=130,
            impact_time=1.30,
            finish_idx=170,
            finish_time=1.70,
        )
        tempo = compute_tempo(events)
        assert isinstance(tempo, TempoMetrics)
        assert np.isclose(tempo.backswing_time, 0.90)
        assert np.isclose(tempo.downswing_time, 0.30)
        assert np.isclose(tempo.tempo_ratio, 3.0)

    def test_tempo_zero_downswing_rejects(self) -> None:
        """Reject non-positive downswing time via DbC."""
        events = SwingEvents(
            address_idx=10,
            address_time=0.10,
            top_idx=100,
            top_time=1.00,
            impact_idx=100,
            impact_time=1.00,  # downswing = 0
            finish_idx=170,
            finish_time=1.70,
        )
        with pytest.raises((PreconditionError, ValueError)):
            compute_tempo(events)


@pytest.mark.unit
class TestSegmentRotationMetrics:
    """Test suite for pelvis yaw, thorax yaw, and X-factor."""

    def test_analytic_rotations_and_x_factor(self) -> None:
        """Analytic rotation test: known yaw rates for hips and shoulders."""
        n = 100
        dt = 0.01
        t = np.arange(n, dtype=np.float64) * dt

        # Pelvis yaw: theta_p(t) = 40 * sin(pi * t) (deg)
        # Thorax yaw: theta_t(t) = 80 * sin(pi * t) (deg)
        th_p = 40.0 * np.sin(np.pi * t)
        th_t = 80.0 * np.sin(np.pi * t)

        rad_p = np.radians(th_p)
        rad_t = np.radians(th_t)

        # Build hip markers (width 0.3m, rotating about origin in XY plane, Z=0.9)
        d_p = 0.30
        waist_l = np.column_stack(
            [d_p / 2 * np.cos(rad_p), d_p / 2 * np.sin(rad_p), np.full(n, 0.9)]
        )
        waist_r = np.column_stack(
            [-d_p / 2 * np.cos(rad_p), -d_p / 2 * np.sin(rad_p), np.full(n, 0.9)]
        )

        # Build shoulder markers (width 0.4m, rotating about origin in XY plane, Z=1.4)
        d_t = 0.40
        sh_l = np.column_stack(
            [d_t / 2 * np.cos(rad_t), d_t / 2 * np.sin(rad_t), np.full(n, 1.4)]
        )
        sh_r = np.column_stack(
            [-d_t / 2 * np.cos(rad_t), -d_t / 2 * np.sin(rad_t), np.full(n, 1.4)]
        )

        markers = {
            "WaistLeft": waist_l,
            "WaistRight": waist_r,
            "LShoulderBack": sh_l,
            "RShoulderBack": sh_r,
        }
        motion = SwingMotion(t=t, markers=markers)
        events = SwingEvents(
            address_idx=0,
            address_time=0.0,
            top_idx=50,  # t=0.5s -> sin(pi*0.5) = 1.0 (peak rotation)
            top_time=0.5,
            impact_idx=80,
            impact_time=0.8,
            finish_idx=99,
            finish_time=0.99,
        )

        res = compute_segment_rotations(motion, events)
        assert isinstance(res, SegmentRotationMetrics)

        # Check address values (t=0 -> angle=0)
        assert np.isclose(res.pelvis_yaw_address, 0.0, atol=1e-3)
        assert np.isclose(res.thorax_yaw_address, 0.0, atol=1e-3)
        assert np.isclose(res.x_factor_address, 0.0, atol=1e-3)

        # Check top of backswing values (t=0.5 -> peak angle)
        assert np.isclose(res.pelvis_yaw_top, 40.0, atol=1e-1)
        assert np.isclose(res.thorax_yaw_top, 80.0, atol=1e-1)
        assert np.isclose(res.x_factor_top, 40.0, atol=1e-1)

        # Check X-factor stretch: max absolute difference is 40 deg
        assert np.isclose(res.x_factor_stretch, 40.0, atol=1e-1)


@pytest.mark.unit
class TestKinematicSequenceMetrics:
    """Test suite for kinematic sequence peak speeds, times, and order."""

    def test_analytic_kinematic_sequence(self) -> None:
        """Verify peak speeds and sequence ordering for 4 segments."""
        n = 200
        dt = 0.01
        t = np.arange(n, dtype=np.float64) * dt

        # Create 4 gaussian speed pulses with known peak speeds and times:
        # 1. Pelvis at t=1.10s, peak = 350 deg/s
        # 2. Thorax at t=1.20s, peak = 600 deg/s
        # 3. Lead Arm at t=1.30s, peak = 1000 deg/s
        # 4. Club at t=1.38s, peak = 2200 deg/s
        def pulse(t_peak: float, w_peak: float) -> np.ndarray:
            return w_peak * np.exp(-((t - t_peak) ** 2) / (2 * 0.04**2))

        w_p = pulse(1.10, 350.0)
        w_t = pulse(1.20, 600.0)
        w_a = pulse(1.30, 1000.0)
        w_c = pulse(1.38, 2200.0)

        # Integrate angles
        theta_p = np.cumsum(w_p) * dt
        theta_t = np.cumsum(w_t) * dt
        theta_a = np.cumsum(w_a) * dt
        theta_c = np.cumsum(w_c) * dt

        rad_p = np.radians(theta_p)
        rad_t = np.radians(theta_t)
        rad_a = np.radians(theta_a)
        rad_c = np.radians(theta_c)

        # Markers
        waist_l = np.column_stack(
            [0.15 * np.cos(rad_p), 0.15 * np.sin(rad_p), np.full(n, 0.9)]
        )
        waist_r = np.column_stack(
            [-0.15 * np.cos(rad_p), -0.15 * np.sin(rad_p), np.full(n, 0.9)]
        )
        sh_l = np.column_stack(
            [0.20 * np.cos(rad_t), 0.20 * np.sin(rad_t), np.full(n, 1.4)]
        )
        sh_r = np.column_stack(
            [-0.20 * np.cos(rad_t), -0.20 * np.sin(rad_t), np.full(n, 1.4)]
        )

        # Arm: shoulder to wrist vector rotating at rad_a
        wrist_l = sh_l + np.column_stack(
            [0.55 * np.cos(rad_a), 0.55 * np.sin(rad_a), np.zeros(n)]
        )

        # Club: grip to clubhead rotating at rad_c
        grip = wrist_l.copy()
        club_head = grip + np.column_stack(
            [1.0 * np.cos(rad_c), 1.0 * np.sin(rad_c), np.zeros(n)]
        )

        markers = {
            "WaistLeft": waist_l,
            "WaistRight": waist_r,
            "LShoulderBack": sh_l,
            "RShoulderBack": sh_r,
            "LWristTop": wrist_l,
            "Marker_2:2:1": club_head,
            "Marker_3:3:1": grip,
        }
        motion = SwingMotion(t=t, markers=markers, club_head=club_head, grip=grip)
        events = SwingEvents(
            address_idx=0,
            address_time=0.0,
            top_idx=90,
            top_time=0.9,
            impact_idx=145,
            impact_time=1.45,
            finish_idx=190,
            finish_time=1.90,
        )

        res = compute_kinematic_sequence(motion, events)
        assert isinstance(res, KinematicSequenceMetrics)

        # Verify proximal-to-distal ordering
        assert res.order == ("pelvis", "thorax", "lead_arm", "club")
        assert res.is_proximal_to_distal is True

        # Verify peak speeds match within numerical differentiation precision
        assert np.isclose(res.pelvis.peak_speed, 350.0, rtol=0.05)
        assert np.isclose(res.thorax.peak_speed, 600.0, rtol=0.05)
        assert np.isclose(res.lead_arm.peak_speed, 1000.0, rtol=0.05)
        assert np.isclose(res.club.peak_speed, 2200.0, rtol=0.05)

        # Verify peak times
        assert np.isclose(res.pelvis.peak_time, 1.10, atol=0.02)
        assert np.isclose(res.thorax.peak_time, 1.20, atol=0.02)
        assert np.isclose(res.lead_arm.peak_time, 1.30, atol=0.02)
        assert np.isclose(res.club.peak_time, 1.38, atol=0.02)


@pytest.mark.unit
class TestLeadArmMetrics:
    """Test suite for lead elbow included angle and maximum flexion."""

    def test_planar_two_link_arm_analytic(self) -> None:
        """Planar two-link arm with known joint angle profile."""
        n = 100
        dt = 0.01
        t = np.arange(n, dtype=np.float64) * dt

        # Shoulder at (0, 0, 1.5)
        S = np.tile([0.0, 0.0, 1.5], (n, 1))
        # Elbow 0.3m along +Y: E = (0, 0.3, 1.5)
        E = np.tile([0.0, 0.3, 1.5], (n, 1))

        # Forearm angle from straight (flexion angle phi):
        # phi starts at 20 deg (included 160), reaches 70 deg at top (included 110), 10 deg at impact (included 170)
        phi = np.linspace(20.0, 70.0, n)
        phi_rad = np.radians(phi)

        # Wrist: W = E + 0.3 * (0, cos(phi), sin(phi)) in elbow frame
        W = E + 0.3 * np.column_stack([np.zeros(n), np.cos(phi_rad), np.sin(phi_rad)])

        markers = {
            "LShoulderTop": S,
            "LElbowOut": E,
            "LWristTop": W,
        }
        motion = SwingMotion(t=t, markers=markers)
        events = SwingEvents(
            address_idx=0,
            address_time=0.0,
            top_idx=50,
            top_time=0.5,
            impact_idx=80,
            impact_time=0.8,
            finish_idx=99,
            finish_time=0.99,
        )

        res = compute_lead_arm_metrics(motion, events)
        assert isinstance(res, LeadArmMetrics)

        # Address: phi=20 deg -> included angle = 180 - 20 = 160 deg
        assert np.isclose(res.elbow_angle_address, 160.0, atol=1e-3)
        # Top (idx=50): phi ≈ 45.25 deg -> included angle ≈ 134.75 deg
        expected_top = 180.0 - phi[50]
        assert np.isclose(res.elbow_angle_top, expected_top, atol=1e-3)
        # Impact (idx=80): phi ≈ 60.4 deg
        expected_impact = 180.0 - phi[80]
        assert np.isclose(res.elbow_angle_impact, expected_impact, atol=1e-3)

        # Max flexion in address->impact: phi reaches 70 at idx=80 if slice, or phi[80]
        min_included = np.min(res.elbow_angle_deg[:81])
        assert np.isclose(res.min_included_angle, min_included, atol=1e-3)
        assert np.isclose(res.max_flexion_deg, 180.0 - min_included, atol=1e-3)


@pytest.mark.unit
class TestWristMetrics:
    """Test suite for lead wrist hinge angle at top and impact."""

    def test_wrist_hinge_angle_analytic(self) -> None:
        """Wrist hinge angle with known forearm and shaft vectors."""
        n = 100
        dt = 0.01
        t = np.arange(n, dtype=np.float64) * dt

        # Forearm pointing along +Y: E at (0, 0, 1.0), W at (0, 0.3, 1.0)
        E = np.tile([0.0, 0.0, 1.0], (n, 1))
        W = np.tile([0.0, 0.3, 1.0], (n, 1))

        # Club shaft: at top, angle is 90 deg; at impact, angle is 10 deg
        # Shaft unit vector at angle alpha from forearm (+Y):
        # u = (sin(alpha), cos(alpha), 0)
        alpha = np.linspace(20.0, 90.0, n)
        alpha[70:] = 10.0  # at impact (idx=70) and beyond, 10 deg
        alpha[45] = 90.0  # top of backswing at idx=45: 90 deg

        alpha_rad = np.radians(alpha)
        grip = W.copy()
        head = grip + 1.0 * np.column_stack(
            [np.sin(alpha_rad), np.cos(alpha_rad), np.zeros(n)]
        )

        markers = {
            "LElbowOut": E,
            "LWristTop": W,
            "Marker_2:2:1": head,
            "Marker_3:3:1": grip,
        }
        motion = SwingMotion(t=t, markers=markers, club_head=head, grip=grip)
        events = SwingEvents(
            address_idx=0,
            address_time=0.0,
            top_idx=45,
            top_time=0.45,
            impact_idx=70,
            impact_time=0.70,
            finish_idx=95,
            finish_time=0.95,
        )

        res = compute_wrist_metrics(motion, events)
        assert isinstance(res, WristMetrics)
        assert np.isclose(res.hinge_angle_top, 90.0, atol=1e-3)
        assert np.isclose(res.hinge_angle_impact, 10.0, atol=1e-3)


@pytest.mark.unit
class TestHandPathMetrics:
    """Test suite for grip path length and max height."""

    def test_hand_path_semicircle_analytic(self) -> None:
        """Grip on semicircle of radius R=0.8m: arc length is pi*R, max height is R."""
        n = 101
        dt = 0.01
        t = np.arange(n, dtype=np.float64) * dt
        r = 0.8
        theta = np.linspace(0.0, np.pi, n)

        # XZ plane arc: x = -R*cos(theta), z = R*sin(theta) + 0.5
        grip_x = -r * np.cos(theta)
        grip_y = np.zeros(n)
        grip_z = r * np.sin(theta) + 0.5
        grip = np.column_stack([grip_x, grip_y, grip_z])

        motion = SwingMotion(t=t, markers={"grip": grip}, grip=grip)
        events = SwingEvents(
            address_idx=0,
            address_time=0.0,
            top_idx=50,  # apex at theta = pi/2 -> height = 0.8 + 0.5 = 1.3
            top_time=0.5,
            impact_idx=80,
            impact_time=0.8,
            finish_idx=100,
            finish_time=1.00,
        )

        res = compute_hand_path_metrics(motion, events)
        assert isinstance(res, HandPathMetrics)

        expected_total_len = np.pi * r
        assert np.isclose(res.path_length_total_m, expected_total_len, rtol=0.01)
        assert np.isclose(res.max_height_top_m, 1.3, atol=1e-3)
        assert np.isclose(res.height_address_m, 0.5, atol=1e-3)


@pytest.mark.unit
class TestClubMetrics:
    """Test suite for club head speed, shaft lean, and face angle."""

    def test_club_delivery_analytic(self) -> None:
        """Club delivery metrics with known analytic values at impact."""
        n = 100
        dt = 0.01
        t = np.arange(n, dtype=np.float64) * dt

        # Clubhead moving along +X at constant 45 m/s near impact
        v_impact = 45.0
        club_x = v_impact * t
        club_y = np.zeros(n)
        club_z = 0.05 * np.ones(n)
        club_head = np.column_stack([club_x, club_y, club_z])

        # Grip: forward shaft lean of 8 deg
        # dx = L * sin(8 deg), dz = L * cos(8 deg) with L = 1.0 m
        lean_deg = 8.0
        lean_rad = math.radians(lean_deg)
        dx = 1.0 * math.sin(lean_rad)
        dz = 1.0 * math.cos(lean_rad)

        grip = club_head + np.array([dx, 0.0, dz])

        # Face normal: open by 3.5 deg relative to target line (+X) in horizontal plane
        face_deg = 3.5
        face_rad = math.radians(face_deg)
        face_normal = np.tile([math.cos(face_rad), math.sin(face_rad), 0.0], (n, 1))

        motion = SwingMotion(
            t=t,
            markers={"club_head": club_head, "grip": grip},
            club_head=club_head,
            grip=grip,
            face_normal=face_normal,
        )
        events = SwingEvents(
            address_idx=0,
            address_time=0.0,
            top_idx=40,
            top_time=0.4,
            impact_idx=70,
            impact_time=0.7,
            finish_idx=95,
            finish_time=0.95,
        )

        res = compute_club_metrics(motion, events)
        assert isinstance(res, ClubMetrics)
        assert np.isclose(res.impact_club_head_speed_m_s, 45.0, atol=1e-2)
        assert np.isclose(res.impact_shaft_lean_deg, 8.0, atol=1e-2)
        assert res.impact_face_angle_deg is not None
        assert np.isclose(res.impact_face_angle_deg, 3.5, atol=1e-2)

    def test_club_delivery_without_face_normal(self) -> None:
        """When face_normal is omitted, impact_face_angle_deg is None."""
        n = 50
        dt = 0.01
        t = np.arange(n, dtype=np.float64) * dt
        head = np.zeros((n, 3))
        grip = head + np.array([0.1, 0.0, 1.0])

        motion = SwingMotion(
            t=t, markers={"head": head, "grip": grip}, club_head=head, grip=grip
        )
        events = SwingEvents(0, 0.0, 20, 0.2, 35, 0.35, 49, 0.49)
        res = compute_club_metrics(motion, events)
        assert res.impact_face_angle_deg is None


@pytest.mark.unit
class TestSwingMotionFromMarkers:
    """Test suite for building SwingMotion from marker dict using capture-A naming."""

    def test_builds_from_capture_a_markers(self) -> None:
        """Build SwingMotion from markers using standard capture-A labels."""
        n = 20
        t = np.arange(n, dtype=np.float64) * 0.01
        markers = {
            label: np.full((n, 3), np.nan, dtype=np.float64)
            for label in CAPTURE_A_MARKER_LABELS
        }

        # Place clubhead and grip
        markers["Marker_2:2:1"] = np.ones((n, 3))
        markers["Marker_3:3:1"] = 2.0 * np.ones((n, 3))

        motion = swing_motion_from_markers(t, markers)
        assert isinstance(motion, SwingMotion)
        assert motion.club_head is not None
        assert motion.grip is not None
        assert motion.shaft_axis is not None
        assert np.allclose(motion.club_head, 1.0)
        assert np.allclose(motion.grip, 2.0)
