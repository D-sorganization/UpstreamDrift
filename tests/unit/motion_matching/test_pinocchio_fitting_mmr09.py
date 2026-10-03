"""Behavioral unit and integration tests for Pinocchio native fitting and cross-engine torque transfer (MMR-09, #11093).

Contracts verified:
1. An iron run using driver attachment identity or driver calibration must fail before promotion.
2. A solve/replay pair with different declared integrators must fail before promotion.
3. Named-state permutation must not silently change the physical pose (strict order/name invariant checking).
4. G1 must pass before G2/G3 promotion.
5. Force/closure/RoM checks are not just solver convergence.
6. Controls imported into MuJoCo/Simscape reproduce declared tolerances or reject with diagnostic.
7. Energy/virtual-work and velocity mapping tests cover quaternion and scalar joints.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import Horizon
from src.shared.python.motion_matching.named_state import (
    CaptureAttachmentDeclaration,
    NamedStateConformanceAdapter,
    NamedStateManifest,
    QuaternionVelocityMap,
    SmallBodyVirtualWorkOracle,
    SphericalJointVirtualWorkOracle,
)
from src.shared.python.motion_matching.pinocchio_g2_g3 import (
    CandidatePromotionRequest,
    CandidatePromotionVerdict,
    ClubKind,
    CrossEngineTransferMatrix,
    EngineTransferVerdict,
    IntegratorConfig,
    TorqueTransferTolerances,
    evaluate_candidate_promotion,
    evaluate_cross_engine_torque_transfer,
)

pytestmark = pytest.mark.unit


def _valid_driver_declaration() -> CaptureAttachmentDeclaration:
    return CaptureAttachmentDeclaration(
        club=ClubKind.DRIVER,
        document_id="anthro_driver",
        document_sha256="sha256_driver_doc_12345",
        attachment_calibration_hash="calib_driver_hash_abc",
        grip_frame_id="grip_driver_frame",
    )


def _valid_iron_declaration() -> CaptureAttachmentDeclaration:
    return CaptureAttachmentDeclaration(
        club=ClubKind.IRON,
        document_id="anthro_iron",
        document_sha256="sha256_iron_doc_67890",
        attachment_calibration_hash="calib_iron_hash_xyz",
        grip_frame_id="grip_iron_frame",
    )


def _clean_physical_audit() -> dict[str, float]:
    return {
        "max_normal_force_n": 1200.0,
        "max_normal_force_body_weights": 1.5,
        "max_penetration_m": 0.004,
        "closure_translation_error_max_m": 0.002,
        "closure_rotation_error_max_rad": 0.015,
        "weight_fraction_min": 0.35,
        "weight_fraction_max": 1.6,
        "peak_effort_n_m": 120.0,
    }


def _clean_metrics() -> dict[str, float]:
    return {
        "whole_marker_rmse_m": 0.020,
        "early_marker_rmse_m": 0.010,
        "terminal_marker_rmse_m": 0.030,
        "club_marker_rmse_m": 0.045,
        "pelvis_yaw_deg": 2.1,
    }


def test_iron_run_using_driver_attachment_fails_promotion() -> None:
    """An iron run using driver attachment identity or driver calibration must fail before promotion."""
    solve_cfg = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)
    replay_cfg = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)

    # 1. Iron run using driver capture declaration
    bad_req1 = CandidatePromotionRequest(
        club=ClubKind.IRON,
        target_horizon=Horizon.G1,
        solver_converged=True,
        solver_cost=15.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=_valid_driver_declaration(),  # Cross-club contamination!
        physical_audit=_clean_physical_audit(),
        metrics=_clean_metrics(),
        g1_accepted=True,
    )
    verdict1 = evaluate_candidate_promotion(bad_req1)
    assert isinstance(verdict1, CandidatePromotionVerdict)
    assert verdict1.promoted is False
    assert any(
        "cross-club" in r.lower() or "driver" in r.lower()
        for r in verdict1.rejection_reasons
    )

    # 2. Iron declaration but using driver calibration hash or driver document id
    contaminated_decl = CaptureAttachmentDeclaration(
        club=ClubKind.IRON,
        document_id="anthro_driver",  # Reused driver document!
        document_sha256="sha256_driver_doc_12345",
        attachment_calibration_hash="calib_driver_hash_abc",  # Reused driver calibration!
        grip_frame_id="grip_iron_frame",
    )
    bad_req2 = CandidatePromotionRequest(
        club=ClubKind.IRON,
        target_horizon=Horizon.G1,
        solver_converged=True,
        solver_cost=15.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=contaminated_decl,
        physical_audit=_clean_physical_audit(),
        metrics=_clean_metrics(),
        g1_accepted=True,
    )
    verdict2 = evaluate_candidate_promotion(bad_req2)
    assert verdict2.promoted is False
    assert any(
        "driver" in r.lower() or "calibration" in r.lower()
        for r in verdict2.rejection_reasons
    )


def test_solve_replay_integrator_mismatch_fails_promotion() -> None:
    """A solve/replay pair with different declared integrators must fail before promotion."""
    # Mismatched names
    req_diff_names = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G1,
        solver_converged=True,
        solver_cost=12.0,
        solve_integrator=IntegratorConfig(
            name="implicit_euler", rtol=1e-4, fixed_step=True
        ),
        replay_integrator=IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False),
        capture_declaration=_valid_driver_declaration(),
        physical_audit=_clean_physical_audit(),
        metrics=_clean_metrics(),
        g1_accepted=True,
    )
    verdict_names = evaluate_candidate_promotion(req_diff_names)
    assert verdict_names.promoted is False
    assert any("integrator" in r.lower() for r in verdict_names.rejection_reasons)

    # Mismatched rtol
    req_diff_rtol = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G1,
        solver_converged=True,
        solver_cost=12.0,
        solve_integrator=IntegratorConfig(name="rk45", rtol=1e-4, fixed_step=False),
        replay_integrator=IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False),
        capture_declaration=_valid_driver_declaration(),
        physical_audit=_clean_physical_audit(),
        metrics=_clean_metrics(),
        g1_accepted=True,
    )
    verdict_rtol = evaluate_candidate_promotion(req_diff_rtol)
    assert verdict_rtol.promoted is False
    assert any(
        "rtol" in r.lower() or "integrator" in r.lower()
        for r in verdict_rtol.rejection_reasons
    )


def test_named_state_permutation_invariance_and_order_checks() -> None:
    """Named-state permutation must not silently change the physical pose (strict order/name invariant checking)."""
    coord_names = (
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "hip_flex_r",
        "knee_ext_r",
        "ankle_flex_r",
    )
    vel_names = tuple(f"v_{c}" for c in coord_names)
    ctrl_names = tuple(f"u_{c}" for c in coord_names[3:])
    manifest = NamedStateManifest(
        schema_version="named-state-conformance/1.0.0",
        coordinate_names=coord_names,
        velocity_names=vel_names,
        control_names=ctrl_names,
        armature=dict.fromkeys(coord_names, 5e-3),
        interpolation=dict.fromkeys(coord_names, "pchip"),
    )
    adapter = NamedStateConformanceAdapter(manifest)

    # State dictionary with canonical key ordering
    dict_canonical = {
        "pelvis_tx": 0.05,
        "pelvis_ty": -0.10,
        "pelvis_tz": 0.88,
        "hip_flex_r": 0.35,
        "knee_ext_r": -0.45,
        "ankle_flex_r": 0.12,
    }
    # State dictionary with permuted key ordering
    dict_permuted = {
        "ankle_flex_r": 0.12,
        "pelvis_tz": 0.88,
        "hip_flex_r": 0.35,
        "pelvis_tx": 0.05,
        "knee_ext_r": -0.45,
        "pelvis_ty": -0.10,
    }

    # Canonical and permuted dicts MUST produce identical ordered vectors and kinematics
    vec_canonical = manifest.pack_q(dict_canonical)
    vec_permuted = manifest.pack_q(dict_permuted)
    np.testing.assert_allclose(vec_canonical, vec_permuted)

    kin_canonical = adapter.compute_kinematics(dict_canonical)
    kin_permuted = adapter.compute_kinematics(dict_permuted)
    np.testing.assert_allclose(
        kin_canonical["end_effector"], kin_permuted["end_effector"]
    )

    # Permuting vector entries directly without name mapping changes the physical pose
    scrambled_vec = np.array([dict_permuted[name] for name in reversed(coord_names)])
    scrambled_pose = float(
        np.sum(scrambled_vec * np.cos(np.arange(len(scrambled_vec))))
    )
    assert not np.isclose(scrambled_pose, kin_canonical["end_effector"][0])

    # Missing or unknown coordinates must fail closed
    with pytest.raises(ValueError, match="missing required coordinate"):
        manifest.pack_q({"pelvis_tx": 0.05})

    with pytest.raises(ValueError, match="unknown coordinate"):
        manifest.pack_q({**dict_canonical, "nonexistent_coord": 0.0})


def test_g1_must_pass_before_g2_g3_promotion() -> None:
    """G1 must pass before G2/G3 promotion."""
    solve_cfg = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)
    replay_cfg = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)

    # G2 promotion without G1 accepted must fail
    req_g2_no_g1 = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G2,
        solver_converged=True,
        solver_cost=25.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=_valid_driver_declaration(),
        physical_audit=_clean_physical_audit(),
        metrics=_clean_metrics(),
        g1_accepted=False,
    )
    verdict_g2 = evaluate_candidate_promotion(req_g2_no_g1)
    assert verdict_g2.promoted is False
    assert any("g1 must pass" in r.lower() for r in verdict_g2.rejection_reasons)

    # G3 promotion without G1 accepted must fail
    req_g3_no_g1 = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G3,
        solver_converged=True,
        solver_cost=45.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=_valid_driver_declaration(),
        physical_audit=_clean_physical_audit(),
        metrics=_clean_metrics(),
        g1_accepted=False,
    )
    verdict_g3 = evaluate_candidate_promotion(req_g3_no_g1)
    assert verdict_g3.promoted is False
    assert any("g1 must pass" in r.lower() for r in verdict_g3.rejection_reasons)

    # G2 promotion WITH G1 accepted passes all gates
    req_g2_with_g1 = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G2,
        solver_converged=True,
        solver_cost=25.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=_valid_driver_declaration(),
        physical_audit=_clean_physical_audit(),
        metrics=_clean_metrics(),
        g1_accepted=True,
    )
    verdict_g2_ok = evaluate_candidate_promotion(req_g2_with_g1)
    assert verdict_g2_ok.promoted is True
    assert len(verdict_g2_ok.rejection_reasons) == 0


def test_force_closure_rom_checks_are_not_just_solver_convergence() -> None:
    """Force/closure/RoM checks are not just solver convergence."""
    solve_cfg = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)
    replay_cfg = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)

    # 1. Penetration violation: 17.2 mm > 10 mm
    audit_bad_pen = _clean_physical_audit()
    audit_bad_pen["max_penetration_m"] = 0.0172
    req_bad_pen = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G1,
        solver_converged=True,
        solver_cost=10.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=_valid_driver_declaration(),
        physical_audit=audit_bad_pen,
        metrics=_clean_metrics(),
        g1_accepted=True,
    )
    v_pen = evaluate_candidate_promotion(req_bad_pen)
    assert v_pen.promoted is False
    assert any("penetration" in r.lower() for r in v_pen.rejection_reasons)

    # 2. Closure violation: 12.5 mm > 5 mm
    audit_bad_closure = _clean_physical_audit()
    audit_bad_closure["closure_translation_error_max_m"] = 0.0125
    req_bad_closure = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G1,
        solver_converged=True,
        solver_cost=10.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=_valid_driver_declaration(),
        physical_audit=audit_bad_closure,
        metrics=_clean_metrics(),
        g1_accepted=True,
    )
    v_closure = evaluate_candidate_promotion(req_bad_closure)
    assert v_closure.promoted is False
    assert any("closure" in r.lower() for r in v_closure.rejection_reasons)

    # 3. RoM violation: coordinate leaves human range by 15 degrees (> 0.5 deg tolerance)
    req_bad_rom = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G1,
        solver_converged=True,
        solver_cost=10.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=_valid_driver_declaration(),
        physical_audit=_clean_physical_audit(),
        metrics=_clean_metrics(),
        g1_accepted=True,
        rom_violations={"elbow_flex_r": 15.0},
    )
    v_rom = evaluate_candidate_promotion(req_bad_rom)
    assert v_rom.promoted is False
    assert any(
        "range of motion" in r.lower() or "rom" in r.lower()
        for r in v_rom.rejection_reasons
    )

    # 4. Normal force body weight violation: 4.2 BW > 3.0 BW
    audit_bad_force = _clean_physical_audit()
    audit_bad_force["max_normal_force_body_weights"] = 4.2
    req_bad_force = CandidatePromotionRequest(
        club=ClubKind.DRIVER,
        target_horizon=Horizon.G1,
        solver_converged=True,
        solver_cost=10.0,
        solve_integrator=solve_cfg,
        replay_integrator=replay_cfg,
        capture_declaration=_valid_driver_declaration(),
        physical_audit=audit_bad_force,
        metrics=_clean_metrics(),
        g1_accepted=True,
    )
    v_force = evaluate_candidate_promotion(req_bad_force)
    assert v_force.promoted is False
    assert any(
        "normal force" in r.lower() or "weight" in r.lower()
        for r in v_force.rejection_reasons
    )


def test_controls_imported_into_mujoco_simscape_reproduce_declared_tolerances() -> None:
    """Controls imported into MuJoCo/Simscape reproduce declared tolerances or reject with diagnostic."""
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MultiEngineTorqueAllocator,
        create_engine_force_adapter,
    )

    n_frames = 10
    nv = 44
    time_s = np.linspace(0.0, 0.1, n_frames)
    q = np.zeros((n_frames, nv))
    q[:, 2] = 0.85
    v = np.zeros((n_frames, nv))
    a = np.zeros((n_frames, nv))
    coord_names = tuple(f"coord_{i}" for i in range(nv))

    # Allocate source controls from Pinocchio plant
    pin_adapter = create_engine_force_adapter("pinocchio", allow_synthetic=True, nv=nv)
    pin_allocator = MultiEngineTorqueAllocator(pin_adapter)
    u = np.zeros((n_frames, 38))
    for i in range(n_frames):
        alloc = pin_allocator.allocate_frame(q[i], v[i], a[i])
        u[i] = alloc.tau_actuated

    matrix = evaluate_cross_engine_torque_transfer(
        source_engine="pinocchio",
        club=ClubKind.DRIVER,
        target_engines=("mujoco", "simscape"),
        time_s=time_s,
        q=q,
        v=v,
        a=a,
        controls_u=u,
        coordinate_names=coord_names,
        declared_armature_kg_m2=5e-3,
        delta_tau_root=np.zeros((n_frames, 6)),
        tracking_gains={"kp": 400.0, "kd": 40.0},
        tolerances=TorqueTransferTolerances(
            max_acceleration_parity_m_s2=0.05,
            engine_tolerances={"simscape": 1.0},
            engine_equilibrium_tolerances={"simscape": 10.0},
        ),
    )

    assert isinstance(matrix, CrossEngineTransferMatrix)
    assert matrix.source_engine == "pinocchio"
    assert "mujoco" in matrix.target_verdicts
    assert "simscape" in matrix.target_verdicts
    assert matrix.all_accepted is True
    assert matrix.target_verdicts["mujoco"].accepted is True
    assert matrix.target_verdicts["simscape"].accepted is True
    assert matrix.declared_armature_kg_m2 == pytest.approx(5e-3)
    assert matrix.residual_root_assistance_included is True


def test_controls_imported_into_target_engine_rejects_with_diagnostic_when_divergent() -> (
    None
):
    """Controls exceeding declared tolerance or incompatible topology must reject with diagnostic."""
    n_frames = 5
    nv = 44
    time_s = np.linspace(0.0, 0.05, n_frames)
    q = np.zeros((n_frames, nv))
    v = np.zeros((n_frames, nv))
    # Huge mismatched acceleration that cannot be reproduced within tight 1e-4 tolerance
    a = np.full((n_frames, nv), 50.0)
    u = np.zeros((n_frames, 38))
    coord_names = tuple(f"coord_{i}" for i in range(nv))

    matrix = evaluate_cross_engine_torque_transfer(
        source_engine="pinocchio",
        club=ClubKind.DRIVER,
        target_engines=("simscape",),
        time_s=time_s,
        q=q,
        v=v,
        a=a,
        controls_u=u,
        coordinate_names=coord_names,
        tolerances=TorqueTransferTolerances(max_acceleration_parity_m_s2=1e-4),
    )

    assert matrix.all_accepted is False
    simscape_v = matrix.target_verdicts["simscape"]
    assert isinstance(simscape_v, EngineTransferVerdict)
    assert simscape_v.accepted is False
    assert simscape_v.acceleration_parity_residual > 1e-4
    assert (
        "exceeds declared tolerance" in simscape_v.reason
        or "parity" in simscape_v.reason
    )


def test_energy_virtual_work_and_velocity_mapping_cover_quaternion_and_scalar_joints() -> (
    None
):
    """Energy/virtual-work and velocity mapping tests cover quaternion and scalar joints."""
    # --- 1. Scalar Joint Virtual Work Duality (SmallBodyVirtualWorkOracle) ---
    oracle_scalar = SmallBodyVirtualWorkOracle()
    q_map = {"shoulder_flex": 0.5, "elbow_flex": 0.8}
    dq_map = {"shoulder_flex": 0.02, "elbow_flex": -0.015}
    f_tip = np.array([12.5, -8.0])

    w_joint, w_cart = oracle_scalar.verify_virtual_work_duality(q_map, dq_map, f_tip)
    assert np.isclose(w_joint, w_cart, rtol=1e-12, atol=1e-12)

    # --- 2. Quaternion/Spherical Joint Virtual Work & Energy Conservation ---
    oracle_quat = SphericalJointVirtualWorkOracle()
    # Rotation about axis [1, 2, 3] by angle 0.6 rad
    axis = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
    angle = 0.6
    q_rot = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])

    omega = np.array([2.5, -1.8, 3.2])
    torque = np.array([15.0, -22.0, 8.5])
    dt = 0.01

    w_joint_quat, w_cart_quat = oracle_quat.verify_virtual_work_duality(
        q=q_rot, omega=omega, torque=torque, dt=dt
    )
    assert np.isclose(w_joint_quat, w_cart_quat, rtol=1e-12, atol=1e-12)

    # --- 3. Quaternion Velocity Mapping with Double Cover Invariance ---
    dt_step = 0.005
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w_target = np.array([1.5, -2.0, 0.8])
    # Exact integration to q1
    theta = float(np.linalg.norm(w_target)) * dt_step
    ax = w_target / np.linalg.norm(w_target)
    q_step = np.array([np.cos(theta / 2.0), *(ax * np.sin(theta / 2.0))])

    # Extracted angular velocity must match target
    w_extracted = QuaternionVelocityMap.quaternion_to_angular_velocity(
        q0, q_step, dt_step
    )
    np.testing.assert_allclose(w_extracted, w_target, rtol=1e-6, atol=1e-6)

    # Double-cover antipodal alignment: -q_step must yield identical angular velocity
    w_antipodal = QuaternionVelocityMap.quaternion_to_angular_velocity(
        q0, -q_step, dt_step
    )
    np.testing.assert_allclose(w_antipodal, w_target, rtol=1e-6, atol=1e-6)
