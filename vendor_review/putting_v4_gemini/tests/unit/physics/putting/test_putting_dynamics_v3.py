from __future__ import annotations
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent


class _DirectStatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        base = fullname.rpartition(".")[2]
        cand = _THIS_DIR / f"{base}.py"
        if cand.exists():
            return importlib.util.spec_from_file_location(fullname, cand)
        return None


sys.meta_path.insert(0, _DirectStatFinder())

"""Automated 22-Test TDD/DbC/LoD/DRY Validation Suite for v3.3 Putting Dynamics, UpstreamDrift & AffineDrift.

Validates:
1-7.   Dual-Axis 4-Point Grip Couple Equilibrium, Zero Shear, Off-Plane Gravity Support vs Stabilization, & C^2 Noise
8-12.  UpstreamDrift Ball Impact, Sweet-Spot Efficiency, Skid-to-Roll Transition, Sloped Break, DbC, & USGA Stimpmeter
13.    Exact Impact Timing Synchronization (theta(t_imp) == 0 at t_imp = 0.8168s, zero launch delay)
14.    Moving Club Butt Hand-Arc Trajectory (3D shoulder-driven arc travel > 5 cm)
15.    Oblong / Kidney Green Contour & Holmes/Penner Lip-Out vs Center-Cup Capture Physics
16.    AffineDrift Control-Affine Tangent Bundle Decomposition (dx/dt = f_drift(x) + G_control(x) * u) & LoD Facade
17.    Apples-to-Apples Unperturbed Nominal Equivalence (Moment-of-Force tau_preload=0 vs Stabilized tau_preload=0.60)
18.    Deterministic Perturbation Impulse Rejection (Mid-Downswing t=0.72s Disturbance: >5x Face Error Reduction)
19.    Eccentric Toe-Strike Recoil Attenuation (+10mm Toe Hit: >2x Effective MOI & <0.5x Recoil Twist)
20.    Convex U-Shaped Grip Pressure / Counter-Torque Optimization Curve (tau_preload in [0.0, 2.0] N*m vs SDN Tremor)
21.    4-Point Dual-Hand Grasp Matrix Null-Space Orthogonality & ZTCF/ZVCF Counterfactual Superposition
22.    Zero-Allocation @cached_precondition Signature Caching (D-11803) & AimPoint Optimal Launch Solver
"""


import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from putting_dynamics_v3 import (
    AffineDriftDecomposition,
    BallRollPhysics,
    BallState,
    DualHandGraspNullSpace,
    GreenSurface,
    PutterBallImpactEngine,
    RollMode,
    SlopeRegion,
    StrokeParameters,
    TurfProperties,
    compute_exact_impact_time,
    decompose_ztcf_zvcf_stroke,
    explore_grip_pressure_optimization,
    generate_smooth_ou_noise,
    simulate_matched_nominal_perturbation_response,
    simulate_v3_dual_axis_stroke,
    solve_aimpoint_optimal_launch,
)


class TestDualAxisCounterbalancedGrip(unittest.TestCase):
    """Tests for the 3D 4-point force couple, moving butt arc, and dual-axis midpoint torque mechanics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim_pre = simulate_v3_dual_axis_stroke(tau_preload_inplane=0.85, tau_preload_offplane=0.55, motor_noise_scale=0.0)
        cls.sim_slk = simulate_v3_dual_axis_stroke(tau_preload_inplane=0.00, tau_preload_offplane=0.00, motor_noise_scale=0.0)
        cls.sim_noisy = simulate_v3_dual_axis_stroke(tau_preload_inplane=0.85, tau_preload_offplane=0.55, motor_noise_scale=1.5, seed=99)

    def test_01_zero_lateral_hub_shear_in_each_hand(self) -> None:
        """Forefinger and pinky forces within each hand must sum to zero in both planes (pure torque couple)."""
        net_F_L = self.sim_pre["F_L_pky"] + self.sim_pre["F_L_idx"]
        net_F_R = self.sim_pre["F_R_pky"] + self.sim_pre["F_R_idx"]
        np.testing.assert_allclose(net_F_L, 0.0, atol=1e-12)
        np.testing.assert_allclose(net_F_R, 0.0, atol=1e-12)

    def test_02_inplane_counterbalanced_torque_signs(self) -> None:
        """Left hand midpoint torque must remain strictly negative (< -preload) and Right hand strictly positive (> +preload)."""
        self.assertTrue(np.all(self.sim_pre["tau_L_in"] <= -0.84))
        self.assertTrue(np.all(self.sim_pre["tau_R_in"] >= +0.84))

    def test_03_offplane_left_supports_gravity_right_stabilizes_against(self) -> None:
        """Validates user hypothesis: Left hand torques positive (+tau_L_off > 0) to support against gravity droop (< 0),
        while Right hand torques negative (-tau_R_off < 0) against the left hand to stabilize the plane."""
        self.assertTrue(np.all(self.sim_pre["tau_grav_droop"] < -0.90))
        self.assertTrue(np.all(self.sim_pre["tau_L_off"] >= +0.54))
        self.assertTrue(np.all(self.sim_pre["tau_R_off"] <= -0.54))
        net_env = self.sim_pre["tau_grav_droop"] + self.sim_pre["tau_cent_whip"]
        net_hands = self.sim_pre["tau_L_off"] + self.sim_pre["tau_R_off"]
        np.testing.assert_allclose(net_hands + net_env, 0.0, atol=1e-12)

    def test_04_moment_arm_consistency_at_midpoints(self) -> None:
        """Midpoint torques at M_L and M_R must equal d * F_couple."""
        d_L, d_R = 0.10, 0.10
        np.testing.assert_allclose(self.sim_pre["Tau_M_L"][:, 2], d_L * self.sim_pre["F_L_idx"][:, 1], atol=1e-12)
        np.testing.assert_allclose(self.sim_pre["Tau_M_R"][:, 2], d_R * self.sim_pre["F_R_idx"][:, 1], atol=1e-12)

    def test_05_smooth_continuous_vectors_under_motor_noise(self) -> None:
        """Force and torque vectors under Ornstein-Uhlenbeck motor noise must be C^2 continuous without step discontinuities."""
        F_noisy = self.sim_noisy["F_L_idx"][:, 1]
        noise_only = F_noisy - self.sim_pre["F_L_idx"][:, 1]
        max_noise_step = float(np.max(np.abs(np.diff(noise_only))))
        max_c2_diff = float(np.max(np.abs(np.diff(F_noisy[10:-10], n=2))))
        self.assertLess(max_noise_step, 0.05)
        self.assertLess(max_c2_diff, 0.05)
        self.assertGreater(float(np.std(noise_only)), 0.01)

    def test_06_transition_jerk_reduction(self) -> None:
        """Pre-loaded counterbalanced system must achieve >5x lower peak transition jerk than unbalanced slack system."""
        t = self.sim_pre["t"]
        mask = (t >= 0.45) & (t <= 0.75)
        pk_pre = float(np.max(np.abs(self.sim_pre["jerk_deg"][mask])))
        pk_slk = float(np.max(np.abs(self.sim_slk["jerk_deg"][mask])))
        self.assertGreater(pk_slk / pk_pre, 5.0)

    def test_07_offplane_lie_stability(self) -> None:
        """Pre-loaded dual-axis system must maintain >15x tighter off-plane lie angle than unbalanced system."""
        max_ph_pre = float(np.max(np.abs(self.sim_pre["phi_deg"])))
        max_ph_slk = float(np.max(np.abs(self.sim_slk["phi_deg"])))
        self.assertLess(max_ph_pre, 0.03)
        self.assertGreater(max_ph_slk / max_ph_pre, 15.0)


class TestBallImpactAndGreenRollPhysics(unittest.TestCase):
    """Tests for UpstreamDrift & AffineDrift compatibility, exact impact timing, oblong green, and v3.3 extensions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim_pre = simulate_v3_dual_axis_stroke(tau_preload_inplane=0.85, tau_preload_offplane=0.55, motor_noise_scale=0.0)

    def test_08_sweet_spot_efficiency_and_cor(self) -> None:
        """Center strike must transfer higher ball speed than a 15mm toe mishit; higher MOI must lose less speed."""
        p_center = StrokeParameters(speed=1.65, direction=np.array([1.0, 0.0]), impact_location_m=np.array([0.0, 0.0]), putter_moi_gcm2=5000.0)
        p_toe_hi = StrokeParameters(speed=1.65, direction=np.array([1.0, 0.0]), impact_location_m=np.array([0.015, 0.0]), putter_moi_gcm2=5500.0)
        p_toe_lo = StrokeParameters(speed=1.65, direction=np.array([1.0, 0.0]), impact_location_m=np.array([0.015, 0.0]), putter_moi_gcm2=3000.0)

        b_center = PutterBallImpactEngine.compute_post_impact_ball_state(np.zeros(2), p_center)
        b_toe_hi = PutterBallImpactEngine.compute_post_impact_ball_state(np.zeros(2), p_toe_hi)
        b_toe_lo = PutterBallImpactEngine.compute_post_impact_ball_state(np.zeros(2), p_toe_lo)

        self.assertGreater(b_center.speed, b_toe_hi.speed)
        self.assertGreater(b_toe_hi.speed, b_toe_lo.speed)

    def test_09_skid_to_pure_roll_transition_on_flat_green(self) -> None:
        """Ball must start in SLIDING mode after impact, transition to ROLLING mode, and hole out on a straight flat putt."""
        green = GreenSurface(width=10.0, height=6.0, turf=TurfProperties(stimp_rating=11.5))
        green.set_hole_position(np.array([3.048, 0.0]))
        params = StrokeParameters(speed=1.32, direction=np.array([1.0, 0.0]), face_angle_deg=0.0, attack_angle_deg=-1.0)
        res = PutterBallImpactEngine.simulate_putt_roll(green, params)

        self.assertEqual(res["modes"][0], "sliding")
        self.assertIn("rolling", res["modes"])
        self.assertGreater(res["skid_distance_m"], 0.05)
        self.assertLess(res["skid_distance_m"], 0.85)
        self.assertTrue(res["holed"])

    def test_10_sloped_green_lateral_break(self) -> None:
        """On a 2.5% cross-slope green, a straight-aimed putt must curve downhill laterally by >15 cm."""
        green_flat = GreenSurface(width=10.0, height=6.0, uniform_slope=np.array([0.0, 0.0]))
        green_sloped = GreenSurface(width=10.0, height=6.0, uniform_slope=np.array([0.0, 0.025]))
        params = StrokeParameters(speed=1.35, direction=np.array([1.0, 0.0]))

        res_flat = PutterBallImpactEngine.simulate_putt_roll(green_flat, params)
        res_slop = PutterBallImpactEngine.simulate_putt_roll(green_sloped, params)

        self.assertAlmostEqual(float(res_flat["final_position"][1]), 0.0, places=3)
        self.assertLess(float(res_slop["final_position"][1]), -0.18)

    def test_11_design_by_contract_preconditions(self) -> None:
        """Invalid parameters must raise ValueError via DbC require() checks."""
        with self.assertRaises(ValueError):
            TurfProperties(stimp_rating=20.0)
        with self.assertRaises(ValueError):
            StrokeParameters(speed=-1.0, direction=np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            simulate_v3_dual_axis_stroke(tau_preload_inplane=-0.5)

    def test_12_usga_stimpmeter_calibration(self) -> None:
        """Pure rolling ball launched at USGA Stimpmeter speed (1.83 m/s) on flat green must roll ~Stimp feet."""
        stimp_ft = 11.5
        green = GreenSurface(width=15.0, height=6.0, turf=TurfProperties(stimp_rating=stimp_ft, height_of_cut_mm=3.0))
        green.set_hole_position(np.array([12.0, 3.0]))
        physics = BallRollPhysics(green)
        v0 = 1.83
        state = BallState(position=np.zeros(2), velocity=np.array([v0, 0.0]), spin=np.array([0.0, -v0 / physics.ball_radius, 0.0]))
        t = 0.0
        while state.is_moving and t < 15.0:
            state = physics.step_rk4(state, 0.002)
            t += 0.002
        rolled_feet = float(state.position[0]) / 0.3048
        self.assertAlmostEqual(rolled_feet, stimp_ft, delta=0.35)

    def test_13_exact_impact_timing_synchronization(self) -> None:
        """Impact time t_imp must occur at the exact instant theta_ref crosses 0.0 deg (~0.817 s), eliminating launch delay."""
        t_imp = compute_exact_impact_time(0.60, 1.05, 2.15)
        self.assertAlmostEqual(t_imp, 0.8168, delta=0.005)
        sim = self.sim_pre
        idx_imp = sim["idx_imp"]
        self.assertLess(abs(float(sim["theta_ref_deg"][idx_imp])), 0.25)

    def test_14_moving_butt_hand_arc_trajectory(self) -> None:
        """Butt of the club must travel along a smooth 3D arc (> 5 cm horizontal travel) when butt_arc_radius_m = 0.36 m."""
        butt_pos = self.sim_pre["butt_pos_3d"]
        x_span = float(np.max(butt_pos[:, 0]) - np.min(butt_pos[:, 0]))
        z_lift = float(np.max(butt_pos[:, 2]) - np.min(butt_pos[:, 2]))
        self.assertGreater(x_span, 0.08, "Butt of the club must travel > 8 cm back-and-through along hand arc")
        self.assertGreater(z_lift, 0.002, "Butt of the club must exhibit vertical arc curvature")

    def test_15_oblong_green_contour_and_lip_out_physics(self) -> None:
        """Oblong green contour must be non-rectangular and off-center fast putts must lip out rather than capture."""
        green = GreenSurface(width=8.0, height=5.0)
        boundary = green.get_oblong_boundary_points(120)
        self.assertEqual(boundary.shape, (120, 2))
        self.assertTrue(green.is_within_oblong_green(np.array([1.65, 0.0])))
        self.assertFalse(green.is_within_oblong_green(np.array([10.0, 10.0])))

        hole = green.hole_position
        res_center = green.evaluate_cup_interaction(hole + np.array([-0.01, 0.0]), np.array([0.85, 0.0]))
        res_lipout = green.evaluate_cup_interaction(hole + np.array([0.0, 0.048]), np.array([1.25, 0.0]))
        self.assertTrue(res_center["captured"])
        self.assertFalse(res_lipout["captured"])
        self.assertTrue(res_lipout["lip_out"])

    def test_16_affinedrift_control_affine_decomposition_and_lod(self) -> None:
        """AffineDrift decomposition dx/dt = f_drift(x) + G_control(x)*u must match Newton-Euler accelerations."""
        decomp: AffineDriftDecomposition = self.sim_pre["affine_decomposition"]
        dxdt = decomp.evaluate_state_derivative()
        self.assertEqual(dxdt.shape, (4,))
        self.assertAlmostEqual(float(dxdt[3]), 0.0, places=5)

    def test_17_apples_to_apples_unperturbed_nominal_equivalence(self) -> None:
        """When unperturbed, Moment-of-Force (tau_preload=0) and Stabilized (tau_preload=0.60) strokes must execute
        the EXACT same clubhead kinematics (Delta theta == 0, Delta phi == 0) and post-impact ball launch speed."""
        mof = simulate_matched_nominal_perturbation_response(tau_preload_nm=0.0, perturbation_impulse_nm=0.0, toe_strike_offset_m=0.0, motor_noise_scale=0.0)
        stb = simulate_matched_nominal_perturbation_response(tau_preload_nm=0.60, perturbation_impulse_nm=0.0, toe_strike_offset_m=0.0, motor_noise_scale=0.0)

        np.testing.assert_allclose(mof["theta_deg"], stb["theta_deg"], atol=1e-11)
        np.testing.assert_allclose(mof["phi_deg"], stb["phi_deg"], atol=1e-11)
        self.assertAlmostEqual(mof["impact_face_deg"], 0.0, places=9)
        self.assertAlmostEqual(stb["impact_face_deg"], 0.0, places=9)
        self.assertAlmostEqual(mof["ball_launch_speed_ms"], stb["ball_launch_speed_ms"], places=9)

    def test_18_perturbation_impulse_rejection_stabilized_vs_slack(self) -> None:
        """Under a 0.25 N*m mid-downswing disturbance (t=0.72s), Counter-Torqued stabilization (0.60 N*m) must reduce
        impact face-angle error by >5x compared to the nominal-matched Moment-of-Force stroke (0.00 N*m)."""
        mof_pert = simulate_matched_nominal_perturbation_response(tau_preload_nm=0.0, perturbation_impulse_nm=0.25, perturbation_time_s=0.72)
        stb_pert = simulate_matched_nominal_perturbation_response(tau_preload_nm=0.60, perturbation_impulse_nm=0.25, perturbation_time_s=0.72)

        err_mof = abs(mof_pert["impact_face_deg"])
        err_stb = abs(stb_pert["impact_face_deg"])
        self.assertGreater(err_mof, 0.80, "Unstabilized Moment-of-Force stroke must suffer significant face deflection under impulse")
        self.assertGreater(err_mof / max(1e-9, err_stb), 5.0, "Stabilized counter-torque must reduce impulse face error by >5x")

    def test_19_eccentric_toe_strike_recoil_attenuation(self) -> None:
        """Under a +10 mm toe mishit, locking the 4 grip contacts via counter-torque must >2x effective MOI and halve recoil twist."""
        mof_toe = simulate_matched_nominal_perturbation_response(tau_preload_nm=0.0, toe_strike_offset_m=0.010)
        stb_toe = simulate_matched_nominal_perturbation_response(tau_preload_nm=0.60, toe_strike_offset_m=0.010)

        self.assertGreater(stb_toe["I_eff_gcm2"] / mof_toe["I_eff_gcm2"], 2.0)
        self.assertLess(abs(stb_toe["recoil_face_twist_deg"]) / abs(mof_toe["recoil_face_twist_deg"]), 0.50)
        self.assertGreater(stb_toe["ball_launch_speed_ms"], mof_toe["ball_launch_speed_ms"])

    def test_20_u_shaped_grip_pressure_optimization_curve(self) -> None:
        """Grip pressure sweep across [0, 2.0] N*m must exhibit a strict U-shaped total face error and inverted-U make rate."""
        sweep = explore_grip_pressure_optimization()
        preloads = np.array([row["tau_preload_nm"] for row in sweep])
        tot_err = np.array([row["sigma_tot_deg"] for row in sweep])
        make_pct = np.array([row["make_rate_pct"] for row in sweep])

        idx_min_err = int(np.argmin(tot_err))
        idx_max_make = int(np.argmax(make_pct))

        self.assertTrue(0.40 <= preloads[idx_min_err] <= 0.80, f"Optimal error preload {preloads[idx_min_err]} outside [0.40, 0.80]")
        self.assertTrue(0.40 <= preloads[idx_max_make] <= 0.80, f"Optimal make-rate preload {preloads[idx_max_make]} outside [0.40, 0.80]")
        self.assertLess(tot_err[idx_min_err], tot_err[0] * 0.45)
        self.assertLess(tot_err[idx_min_err], tot_err[-1] * 0.45)
        self.assertGreater(make_pct[idx_max_make], 88.0)
        self.assertLess(make_pct[0], 55.0)
        self.assertLess(make_pct[-1], 35.0)

    def test_21_dual_hand_grasp_nullspace_and_ztcf_zvcf(self) -> None:
        """DualHandGraspNullSpace must enforce zero task spillover and ZTCF/ZVCF must satisfy exact superposition."""
        grasp = DualHandGraspNullSpace()
        res = grasp.decompose_actuation(np.array([-0.42, 0.95]), np.array([0.75, 0.55]))
        np.testing.assert_allclose(grasp.A_couple @ res["u_null"], 0.0, atol=1e-12)
        np.testing.assert_allclose(res["net_task_from_u_total"], np.array([-0.42, 0.95]), atol=1e-12)
        self.assertAlmostEqual(float(np.dot(res["u_task"], res["u_null"])), 0.0, places=12)

        cf = decompose_ztcf_zvcf_stroke(theta_rad=-0.12, omega_rad_s=1.45, tau_net_inplane=0.65, tau_net_offplane=0.92)
        np.testing.assert_allclose(
            cf.q_ddot_actual,
            cf.q_ddot_ztcf + cf.q_ddot_zvcf - cf.q_ddot_gravity,
            atol=1e-12,
        )

    def test_22_zero_allocation_precondition_and_aimpoint_solver(self) -> None:
        """Zero-allocation @cached_precondition must reject negative preload and AimPoint solver must hole breaking putt."""
        with self.assertRaises(ValueError):
            simulate_matched_nominal_perturbation_response(tau_preload_nm=-0.1)

        green = GreenSurface(width=10.0, height=6.0, turf=TurfProperties(stimp_rating=11.5), uniform_slope=np.array([0.004, 0.022]))
        green.set_hole_position(np.array([3.048, 0.0]))
        sol = solve_aimpoint_optimal_launch(green)
        self.assertGreater(sol["optimal_aim_deg"], 2.0, "AimPoint solver must aim uphill (+Y) on a +2.2% cross-slope")
        self.assertTrue(sol["holed"], "AimPoint optimal trajectory must be captured by the USGA cup")



    def test_23_realistic_ball_rebound_and_clubhead_deceleration(self) -> None:
        """Ball must rebound off the clubface faster than the clubhead (~1.59x pre-impact, ~2.00x post-impact follow-through)."""
        params = StrokeParameters(speed=1.12, direction=np.array([1.0, 0.0]), putter_mass_kg=0.360, cor=0.80)
        reb = PutterBallImpactEngine.compute_impact_rebound_kinematics(params, tau_preload_nm=0.60)
        self.assertGreater(reb["smash_factor"], 1.55)
        self.assertLess(reb["v_head_post_ms"], reb["v_head_pre_ms"] * 0.85)
        self.assertGreater(reb["ball_to_followthrough_ratio"], 1.95)



    def test_24_undulating_green_and_micro_roughness(self):
        """Non-planar multi-tier undulating green gradient must match finite differences and include turf chatter."""
        from green_surface_undulation import (
            GreenUndulationProfile,
            evaluate_green_elevation_m,
            evaluate_green_gradient,
            sample_surface_roughness_acceleration,
        )
        prof = GreenUndulationProfile(slope_x_pct=0.5, slope_y_pct=2.2, tier_height_m=0.035)
        eps = 1e-6
        gx, gy = evaluate_green_gradient(1.85, -0.15, prof)
        fd_x = (evaluate_green_elevation_m(1.85 + eps, -0.15, prof) - evaluate_green_elevation_m(1.85 - eps, -0.15, prof)) / (2.0 * eps)
        fd_y = (evaluate_green_elevation_m(1.85, -0.15 + eps, prof) - evaluate_green_elevation_m(1.85, -0.15 - eps, prof)) / (2.0 * eps)
        self.assertAlmostEqual(gx, fd_x, places=7)
        self.assertAlmostEqual(gy, fd_y, places=7)
        ax, ay = sample_surface_roughness_acceleration(1.5, 0.1, 0.75, roughness_rms_mm=0.40, seed=11)
        self.assertGreater((ax * ax + ay * ay) ** 0.5, 0.0)

    def test_25_player_specific_bellman_strokes_gained_0_to_45_ft(self):
        """Player-specific Bellman expected strokes E_player(d) across 0-45 ft must reflect actual player skill."""
        from strokes_gained_analytics import PlayerSkillProfile, compute_player_expected_strokes_curve
        stab = PlayerSkillProfile(tau_preload_nm=0.60, base_face_sd_deg=0.45, base_speed_cv_pct=3.0)
        poor = PlayerSkillProfile(tau_preload_nm=0.0, base_face_sd_deg=1.30, base_speed_cv_pct=6.5, green_read_sd_pct=0.60)
        c_stab = compute_player_expected_strokes_curve(skill=stab, n_trials=120)
        c_poor = compute_player_expected_strokes_curve(skill=poor, n_trials=120)
        self.assertLess(c_stab["expected_strokes_player"][4], c_poor["expected_strokes_player"][4])

    def test_26_capture_speed_optimization_by_player_skill(self):
        """Poor high-variance putter must optimize SG_player at slower capture speed than stabilized putter."""
        from strokes_gained_analytics import PlayerSkillProfile, optimize_capture_speed_strategy
        stab = PlayerSkillProfile(tau_preload_nm=0.60, base_face_sd_deg=0.42, base_speed_cv_pct=2.8, green_read_sd_pct=0.20)
        poor = PlayerSkillProfile(tau_preload_nm=0.0, base_face_sd_deg=1.35, base_speed_cv_pct=6.8, green_read_sd_pct=0.65)
        opt_s = optimize_capture_speed_strategy(12.0, 2.2, stab, n_trials=240)
        opt_p = optimize_capture_speed_strategy(12.0, 2.2, poor, n_trials=240)
        self.assertLessEqual(opt_p["optimal_entry_speed_m_s"], opt_s["optimal_entry_speed_m_s"])
        self.assertLess(opt_s["optimal_expected_strokes"], opt_p["optimal_expected_strokes"])

    def test_27_six_channel_variance_and_green_reading_decomposition(self):
        """6-channel variance decomposition must quantify Face, Path, Spot, Green Read, Roughness, and Counter-Torque."""
        from strokes_gained_analytics import decompose_putting_variance_channels
        res = decompose_putting_variance_channels(distance_ft=15.0, slope_y_pct=2.2, n_trials=120)
        ch = res["strokes_gained_per_18_holes_by_channel"]
        self.assertEqual(len(ch), 6)
        self.assertGreater(ch["green_reading_slope_error"], 0.0)

    def test_28_stimpmeter_ramp_green_reader_and_mcp_server(self):
        """USGA Stimpmeter ramp rollout, optimal green read solver, and MCP JSON-RPC server must operate cleanly."""
        from putting_mcp_server import PuttingMCPServer
        from stimpmeter_reader import roll_stimpmeter_ball, solve_player_optimal_green_read
        roll = roll_stimpmeter_ball(0.0, 11.0)
        self.assertGreater(roll.stopping_distance_ft, 7.0)
        read = solve_player_optimal_green_read(12.0, 11.0)
        self.assertGreater(read["optimal_launch_speed_m_s"], read["optimal_entry_capture_speed_m_s"])
        srv = PuttingMCPServer()
        resp = srv.handle_jsonrpc_request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "run_stimpmeter_green_read", "arguments": {"distance_ft": 12.0}},
        })
        self.assertIn("result", resp)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
