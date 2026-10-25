"""Full-motion lift target extraction and muscle solve adapters (OSL-8, #13045)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    resolve_base_model,
)
from src.shared.python.lifting.bundle_targets import LiftTargets, lift_targets
from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES


def _retarget_lift_motion(lift: str, sol: Any) -> Any:
    """Retarget sagittal solution to 3D full body motion."""
    from src.shared.python.lifting.back_bar import BACK_SQUAT, BackBarGeometry
    from src.shared.python.lifting.back_rack import RACKS
    from src.shared.python.lifting.barbell_bushing import Barbell
    from src.shared.python.lifting.bundles import (
        LIFT_SPEC,
        lift_hands_to_full_body,
        lift_to_full_body,
        load_seed_pose,
    )
    from src.shared.python.lifting.loaded_balance import LoadBalance

    lift_spec = LIFT_SPEC.read_bytes()
    if lift == "squat":
        geom = BackBarGeometry.from_standard(
            BACK_SQUAT, height_m=1.78, rack=RACKS["high_bar"]
        )
        load = LoadBalance(Barbell(float(sol.source["bar_mass_kg"])))
        return lift_to_full_body(
            lift_spec, sol, load_seed_pose(), geometry=geom, load=load
        )
    return lift_hands_to_full_body(lift_spec, sol, load_seed_pose(), lift)


def _compute_full_motion_kinematics(
    kin: Any,
    frame: Any,
    attachments: dict[str, Any],
    lift_motion: Any,
    in_pelvis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute landmark positions, barbell endpoints, and pelvis rotations."""
    from src.shared.python.lifting.bundle_targets import IK_LANDMARKS, PELVIS_BODY

    order = tuple(attachments)
    full_points = np.array([kin.marker_positions(q) for q in lift_motion.q])
    full_positions = frame.point(full_points[:, [order.index(x) for x in IK_LANDMARKS]])

    # LANDMARK_ATTACHMENTS maps "bar" to ("Hub", _ORIGIN) for kinematic topology,
    # but the barbell's real position comes from the retargeted lift motion (OSL-8, #13045).
    if hasattr(lift_motion, "bar_m") and lift_motion.bar_m is not None:
        full_positions[:, IK_LANDMARKS.index("bar")] = frame.point(lift_motion.bar_m)

    bar_pts = full_positions[:, IK_LANDMARKS.index("bar")]
    ends_1 = bar_pts + np.array([0.0, 0.0, -1.10])
    ends_2 = bar_pts + np.array([0.0, 0.0, 1.10])
    bar_endpoints = np.concatenate([ends_1, ends_2], axis=0)

    full_pelvis_rot = np.array(
        [
            frame.rotation
            @ kin.body_poses(q, [PELVIS_BODY])[PELVIS_BODY][0]
            @ in_pelvis
            for q in lift_motion.q
        ]
    )
    return full_positions, bar_endpoints, full_pelvis_rot


def solve_full_lift_motion_targets(
    lift: str,
    path: Path,
    stride: int = 1,
) -> tuple[LiftTargets, tuple[float, float], np.ndarray]:
    """Extract full-motion LiftTargets from bundle sagittal retargeting provenance.

    Returns (full_targets, (t_start, t_end), bar_endpoints_m).
    """
    from src.shared.python.lifting.bundle_targets import (
        _foot_stations,
        _kinematics,
        _lift_frame,
        bundle_pelvis_rotation,
        landmark_attachments,
        lift_targets,
    )
    from src.shared.python.lifting.sagittal_retarget import SagittalSolution
    from src.shared.python.motion_matching.same_input import InputBundle

    bundle = InputBundle.load(path)
    win_dict = bundle.provenance.get("window", {})
    t_start = float(win_dict.get("start_s", 0.0))
    t_dur = float(win_dict.get("duration_s", 0.2))
    window_s = (t_start, t_start + t_dur)

    targets_short = lift_targets(lift, path, stride=5)
    spec = json.loads(bundle.spec_bytes)
    stations = _foot_stations(spec)
    attachments = landmark_attachments(spec)
    kin, body_mass = _kinematics(bundle.spec_bytes, attachments)
    frame = _lift_frame(kin, bundle.reference_q[0])

    sol = SagittalSolution.from_record(bundle.provenance["sagittal_solution"])
    lift_motion = _retarget_lift_motion(lift, sol)
    in_pelvis = bundle_pelvis_rotation(spec)
    full_positions, bar_endpoints, full_pelvis_rot = _compute_full_motion_kinematics(
        kin, frame, attachments, lift_motion, in_pelvis
    )

    indices = tuple(range(0, len(lift_motion.time_s), stride))
    time_s = lift_motion.time_s[list(indices)]
    full_positions = full_positions[list(indices)]
    full_pelvis_rot = full_pelvis_rot[list(indices)]

    full_targets = LiftTargets(
        lift=lift,
        path=path,
        sha256=targets_short.sha256,
        indices=indices,
        time_s=time_s,
        positions_m=full_positions,
        foot_stations_m=stations,
        grf_force_n=targets_short.grf_force_n,
        grf_moment_nm=targets_short.grf_moment_nm,
        hand_force_n=targets_short.hand_force_n,
        hand_torque_nm=targets_short.hand_torque_nm,
        grip_m=targets_short.grip_m,
        pelvis_rotation=full_pelvis_rot,
        source_coordinates=targets_short.source_coordinates,
        body_mass_kg=body_mass,
        bar_mass_kg=targets_short.bar_mass_kg,
        height_m=targets_short.height_m,
        frame=frame,
        tibial_torsion=targets_short.tibial_torsion,
        bench_force_n=targets_short.bench_force_n,
        bench_point_m=targets_short.bench_point_m,
        provenance=targets_short.provenance,
    )
    return full_targets, window_s, bar_endpoints


def _truncate_targets(targets: LiftTargets, max_frames: int | None) -> LiftTargets:
    """Truncate targets to max_frames along the time dimension if requested."""
    if max_frames is None or max_frames >= len(targets.time_s):
        return targets
    return LiftTargets(
        lift=targets.lift,
        path=targets.path,
        sha256=targets.sha256,
        indices=targets.indices[:max_frames],
        time_s=targets.time_s[:max_frames],
        positions_m=targets.positions_m[:max_frames],
        foot_stations_m=targets.foot_stations_m,
        grf_force_n=targets.grf_force_n,
        grf_moment_nm=targets.grf_moment_nm,
        hand_force_n=targets.hand_force_n,
        hand_torque_nm=targets.hand_torque_nm,
        grip_m=targets.grip_m,
        pelvis_rotation=targets.pelvis_rotation[:max_frames],
        source_coordinates=targets.source_coordinates,
        body_mass_kg=targets.body_mass_kg,
        bar_mass_kg=targets.bar_mass_kg,
        height_m=targets.height_m,
        frame=targets.frame,
        tibial_torsion=targets.tibial_torsion,
        bench_force_n=targets.bench_force_n,
        bench_point_m=targets.bench_point_m,
        provenance=targets.provenance,
    )


def _solve_window_activations(
    lift: str,
    path: Path,
    stride: int,
) -> tuple[LiftTargets, Any, Any, np.ndarray, list[str], dict[str, float]]:
    """Solve short-window static optimization for muscle activations."""
    from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
    from src.engines.physics_engines.opensim.python.msk_lift_arm_reach import (
        reaching_lift_build,
    )
    from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
        MuscleBasis,
        _make_muscles_rigid_tendon,
        _solve_lift_frames,
        leg_coordinates,
    )

    window_targets = lift_targets(lift, path, stride=stride)
    base_model = resolve_base_model()
    scales = ikid.bundle_segment_scales(window_targets, base_model)
    build, reach = reaching_lift_build(lift, window_targets, base_model, scales)
    pelvis_short = ikid.pelvis_landmark_targets(build, window_targets, base_model)
    ik_short = ikid.run_inverse_kinematics(build, window_targets, pelvis_short)
    inverse_short = ikid.run_inverse_dynamics(build, window_targets, ik_short)

    _make_muscles_rigid_tendon(build.model)
    basis = MuscleBasis(build.model, ik_short.coordinates)
    leg_coords = leg_coordinates()
    leg_indices = [ik_short.coordinates.index(c) for c in leg_coords]
    solve_short = _solve_lift_frames(
        basis, ik_short, inverse_short, leg_indices, window_targets.time_s
    )
    return (
        window_targets,
        base_model,
        scales,
        solve_short.activations,
        basis.muscle_names,
        reach.coordinates,
    )


def solve_full_lift_motion_and_muscles(
    lift: str,
    stride: int = 5,
    max_ik_frames: int | None = None,
) -> tuple[
    LiftTargets,
    Any,
    Any,
    LiftTargets,
    np.ndarray,
    list[str],
    tuple[float, float],
    np.ndarray,
]:
    """Solve full-motion IK and OSL-4 window static optimization muscle activations.

    Returns (full_targets, build, full_ik, window_targets, activations, muscle_names, window_s, bar_endpoints).
    """
    from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        build_lift_model,
    )

    path = SAME_INPUT_LIFT_BUNDLES.get(lift)
    if path is None or not path.is_file():
        raise ValueError(f"no same-input bundle found for lift {lift!r}")

    # 1. Short-window solve for activations & wrenches
    (
        window_targets,
        base_model,
        scales,
        activations,
        muscle_names,
        arm_coords,
    ) = _solve_window_activations(lift, path, stride)

    # 2. Full-motion targets & IK
    full_targets, window_s, bar_endpoints = solve_full_lift_motion_targets(
        lift, path, stride=stride
    )
    full_targets = _truncate_targets(full_targets, max_ik_frames)

    build_full = build_lift_model(
        lift,
        body_mass_kg=full_targets.body_mass_kg,
        height_m=full_targets.height_m,
        bar_mass_kg=full_targets.bar_mass_kg,
        base_model_path=base_model,
        segment_scales=scales,
        arm_coordinates=arm_coords,
        tibial_torsion=full_targets.tibial_torsion,
    )
    pelvis_full = ikid.pelvis_landmark_targets(build_full, full_targets, base_model)
    full_ik = ikid.run_inverse_kinematics(build_full, full_targets, pelvis_full)

    return (
        full_targets,
        build_full,
        full_ik,
        window_targets,
        activations,
        muscle_names,
        window_s,
        bar_endpoints,
    )


def solve_lift_motion_and_muscles(
    lift: str,
    stride: int = 5,
) -> tuple[LiftTargets, Any, Any, np.ndarray, list[str]]:
    """Solve IK, ID, and static optimization muscle activations for a barbell lift.

    Returns (targets, build, ik, activations, muscle_names).
    """
    from src.engines.physics_engines.opensim.python.msk_lift_muscles import (
        MuscleBasis,
        _make_muscles_rigid_tendon,
        _run_lift_ik_id,
        _solve_lift_frames,
        leg_coordinates,
    )

    path = SAME_INPUT_LIFT_BUNDLES.get(lift)
    if path is None or not path.is_file():
        raise ValueError(f"no same-input bundle found for lift {lift!r}")

    targets, build, ik, inverse = _run_lift_ik_id(lift, stride)

    _make_muscles_rigid_tendon(build.model)
    basis = MuscleBasis(build.model, ik.coordinates)
    leg_coords = leg_coordinates()
    leg_indices = [ik.coordinates.index(c) for c in leg_coords]
    solve = _solve_lift_frames(basis, ik, inverse, leg_indices, targets.time_s)

    return targets, build, ik, solve.activations, basis.muscle_names
