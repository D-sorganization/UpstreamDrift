"""Lift trajectories as same-input bundles (LIFT-5, #11745).

Pipeline for one lift:

1. :func:`solve_sagittal_squat` runs the Tools Movement Optimizer
   (``movement_optimizer``, sagittal shank/thigh/trunk chain with bar path)
   through its public API and returns a :class:`SagittalSolution`.
2. :func:`lift_to_full_body` solves a standing pose of the full-body spec,
   retargets the planar solution onto foot, knee, hip, shoulder and bar
   landmarks (:mod:`.sagittal_retarget`) and tracks them with the shared
   MuJoCo marker IK (:class:`FullBodyMarkerKinematics`), feet held flat on
   the ground.
3. :func:`build_lift_bundle` tracks a window of that joint trajectory in
   MuJoCo with the pipeline controller (:func:`generate_reference_bundle`) and
   packages it as an :class:`InputBundle` with provenance.

Model limitations are recorded in every bundle's provenance
(:data:`LIMITATIONS`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SagittalFrame,
    SagittalSolution,
    retarget_landmarks,
)
from src.shared.python.motion_matching.same_input import (
    InputBundle,
    VectorPlant,
    generate_reference_bundle,
)

Array = NDArray[np.float64]

_REPO = Path(__file__).resolve().parents[4]
_GROUND_SUPPORT = _REPO / "docs/development/full_body_models/evidence/ground_support"
LIFT_SPEC = _GROUND_SUPPORT / "full_body_spec_hipcal_scaled.json"
SEED_RECORD = _GROUND_SUPPORT / "dynamics_record.npz"
BACK_SQUAT_BUNDLE = (
    _REPO / "docs/development/lifting/evidence/back_squat/back_squat_bundle.npz"
)

BAR_FRAME = "Hub"
LIMITATIONS = (
    "The only spec the same-input harness replays in every engine is the golf "
    "full-body spec, so the lifter keeps that spec's arms and dual-grip club; "
    "the arm coordinates are held at the seed pose, which keeps the grip "
    "closure exact. There is no barbell body or load: the 'bar' is the spec's "
    "Hub frame (shoulder-girdle centre), matching the optimizer's back-squat "
    "bar at the shoulder joint (zero squat_bar_depth/squat_bar_height). The "
    "bar mass enters only the planar optimizer, not the full-body dynamics. "
    "Barbell body and hand contact are LIFT-4 (#11744)."
)
_ORIGIN = (0.0, 0.0, 0.0)
LANDMARK_ATTACHMENTS: dict[str, tuple[str, tuple[float, float, float]]] = {
    "ankle_l": ("talus_l", _ORIGIN),
    "ankle_r": ("talus_r", _ORIGIN),
    "knee_l": ("tibia_l", _ORIGIN),
    "knee_r": ("tibia_r", _ORIGIN),
    "hip_l": ("femur_l", _ORIGIN),
    "hip_r": ("femur_r", _ORIGIN),
    "shoulder_l": ("LS", _ORIGIN),
    "shoulder_r": ("RS", _ORIGIN),
    "bar": (BAR_FRAME, _ORIGIN),
}
STANCE_SPHERES: tuple[str, ...] = ("heel_l", "forefoot_l", "heel_r", "forefoot_r")
UPPER_LIMB_COORDINATES: tuple[str, ...] = (
    "LEInput",
    "LFInput",
    "LScapInputX",
    "LScapInputY",
    "LSInputX",
    "LSInputY",
    "LSInputZ",
    "LWInputX",
    "LWInputY",
    "REInput",
    "RFInput",
    "RScapInputX",
    "RScapInputY",
    "RSInputX",
    "RSInputY",
    "RSInputZ",
    "RWInputX",
    "RWInputY",
)
TOE_COORDINATES: tuple[str, ...] = ("mtp_angle_l", "mtp_angle_r")
POSTURE_COORDINATES: tuple[str, ...] = (
    "hip_adduction_l",
    "hip_adduction_r",
    "hip_rotation_l",
    "hip_rotation_r",
    "SpineInputY",
    "TorsoInput",
)


@dataclass(frozen=True)
class LiftIKOptions:
    """Marker-IK settings; recorded verbatim in the bundle provenance."""

    iterations: int = 100
    standing_iterations: int = 300
    prior_weight: float = 1e-4
    posture_weight: float = 1e-2

    def __post_init__(self) -> None:
        if min(self.iterations, self.standing_iterations) < 1:
            raise ValueError("IK iterations must be positive")
        if not (self.prior_weight >= 0.0 and self.posture_weight >= 0.0):
            raise ValueError("IK weights must be non-negative")


@dataclass(frozen=True)
class FullBodyLift:
    """Full-body joint trajectory tracking the retargeted landmarks.

    ``q`` is (N, nv) in spec coordinate order; ``targets`` (N, M, 3) and
    ``residual_m`` (N, M) follow :data:`LANDMARK_LABELS`; ``bar_m`` (N, 3) is
    the achieved bar landmark.  ``closure_m`` is the grip-weld position error
    and ``lowest_sphere_m`` the lowest foot-sphere height per frame.
    """

    time_s: Array
    q: Array
    targets: Array
    residual_m: Array
    bar_m: Array
    closure_m: Array
    lowest_sphere_m: Array
    q_standing: Array
    frame: SagittalFrame
    options: LiftIKOptions

    def max_residual_by_landmark(self) -> dict[str, float]:
        """Largest landmark error over all frames, metres."""
        peaks = self.residual_m.max(axis=0)
        return {label: float(peaks[k]) for k, label in enumerate(LANDMARK_LABELS)}

    def summary(self) -> dict[str, Any]:
        """Provenance record of the IK solve."""
        return {
            "solver": "FullBodyMarkerKinematics (MuJoCo), Levenberg-Marquardt",
            "landmarks": {k: v[0] for k, v in LANDMARK_ATTACHMENTS.items()},
            "locked": list(UPPER_LIMB_COORDINATES + TOE_COORDINATES),
            "flat_feet": list(STANCE_SPHERES),
            "options": asdict(self.options),
            "frames": int(self.time_s.size),
            "max_residual_m": self.max_residual_by_landmark(),
            "max_rms_m": float(np.sqrt((self.residual_m**2).mean(axis=1)).max()),
            "max_closure_m": float(self.closure_m.max()),
            "min_sphere_height_m": float(self.lowest_sphere_m.min()),
            "sagittal_forward": list(self.frame.forward),
            "q_standing": self.q_standing.tolist(),
        }


def load_seed_pose(path: Path = SEED_RECORD) -> Array:
    """First recorded pose of the committed ground-support run (arms on grip)."""
    with np.load(path, allow_pickle=False) as record:
        return np.asarray(record["q"][0], dtype=float)


def solve_sagittal_squat(
    *,
    body_mass_kg: float = 80.0,
    height_m: float = 1.78,
    bar_mass_kg: float = 100.0,
    duration_s: float = 3.0,
    n_eval: int = 151,
    n_waypoints: int = 12,
) -> SagittalSolution:
    """Back squat (stand, bottom, stand) from the Tools Movement Optimizer.

    Raises:
        ImportError: When ``movement_optimizer`` (vendored Tools) is absent.
        RuntimeError: When the optimizer reports failure.
    """
    import movement_optimizer
    from movement_optimizer.models import BodyModel, make_full_squat_config
    from movement_optimizer.trajectory import TrajectoryOptimizer

    body = BodyModel(body_mass=body_mass_kg, height=height_m)
    dyn, q_start, q_end, q_bounds, q_via = make_full_squat_config(body, bar_mass_kg)
    optimizer = TrajectoryOptimizer(
        body,
        dyn,
        "full_squat",
        bar_mass_kg,
        q_start,
        q_end,
        q_bounds,
        q_via=q_via,
        duration=duration_s,
        n_waypoints=n_waypoints,
        n_eval=n_eval,
    )
    result = optimizer.optimize()
    if not result.success:
        raise RuntimeError("movement optimizer did not converge for full_squat")
    return SagittalSolution(
        time_s=np.asarray(result.t, dtype=float),
        segment_angles_rad=np.asarray(result.q, dtype=float),
        bar_m=np.asarray(result.bar, dtype=float),
        source={
            "package": "movement_optimizer",
            "version": movement_optimizer.__version__,
            "exercise": "full_squat",
            "body_mass_kg": body_mass_kg,
            "height_m": height_m,
            "bar_mass_kg": bar_mass_kg,
            "duration_s": duration_s,
            "n_eval": n_eval,
            "n_waypoints": n_waypoints,
            "success": bool(result.success),
            "cost": float(result.cost),
            "segment_lengths_m": [float(x) for x in dyn.L_eff],
        },
    )


def _foot_frame(kin: Any, q: Array, ground: Any) -> SagittalFrame:
    """Sagittal frame of a pose: forward is the mean heel-to-forefoot line."""
    points = kin.sphere_ground_points(q, ground)
    forward = np.zeros(3)
    for side in "lr":
        forward += points[f"forefoot_{side}"] - points[f"heel_{side}"]
    up = np.asarray(ground.normal, dtype=float)
    forward = forward - (forward @ up) * up
    fx, fy, fz = (float(x) for x in forward)
    ux, uy, uz = (float(x) for x in up)
    return SagittalFrame(forward=(fx, fy, fz), up=(ux, uy, uz))


def _standing_targets(points: Mapping[str, Array], frame: SagittalFrame) -> Array:
    """Upright targets with the model's own segment lengths (bar untargeted)."""
    up, left = np.asarray(frame.up), frame.left
    base = 0.5 * (points["ankle_l"] + points["ankle_r"])
    hip_mid = 0.5 * (points["hip_l"] + points["hip_r"])
    shoulder_mid = 0.5 * (points["shoulder_l"] + points["shoulder_r"])
    trunk = float(np.linalg.norm(shoulder_mid - hip_mid))
    target: dict[str, Array] = {"bar": np.full(3, np.nan)}
    tops = []
    for s in "lr":
        shank = float(np.linalg.norm(points[f"knee_{s}"] - points[f"ankle_{s}"]))
        thigh = float(np.linalg.norm(points[f"hip_{s}"] - points[f"knee_{s}"]))
        ankle = base + float((points[f"hip_{s}"] - base) @ left) * left
        target[f"ankle_{s}"] = ankle
        target[f"knee_{s}"] = ankle + shank * up
        target[f"hip_{s}"] = ankle + (shank + thigh) * up
        tops.append(target[f"hip_{s}"])
    top = 0.5 * (tops[0] + tops[1]) + trunk * up
    for s in "lr":
        lateral = float((points[f"shoulder_{s}"] - hip_mid) @ left)
        target[f"shoulder_{s}"] = top + lateral * left
    return np.array([target[label] for label in LANDMARK_LABELS])


def _fit_arrays(kin: Any, fits: list[Any]) -> tuple[Array, Array, Array, Array]:
    residual = np.array([[f.per_marker_m[k] for k in LANDMARK_LABELS] for f in fits])
    bar_index = LANDMARK_LABELS.index("bar")
    bar = np.array([kin.marker_positions(f.q)[bar_index] for f in fits])
    closure = np.array([f.closure_error_m for f in fits])
    lowest = np.array([f.lowest_sphere_height_m for f in fits])
    return residual, bar, closure, lowest


def lift_to_full_body(
    spec_bytes: bytes,
    solution: SagittalSolution,
    seed_q: Array,
    *,
    options: LiftIKOptions | None = None,
) -> FullBodyLift:
    """Full-body joint trajectory for ``solution`` by marker IK in MuJoCo.

    Args:
        spec_bytes: Full-body spec (``full-body-v1``) document bytes.
        solution: Planar lift solution.
        seed_q: Seed pose in spec order; its upper-limb coordinates are held.
        options: IK settings (defaults: :class:`LiftIKOptions`).

    Postconditions: ``q`` has one row per solution sample; upper-limb and
    toe coordinates are constant.
    """
    if not isinstance(spec_bytes, bytes | bytearray):
        raise TypeError("spec_bytes must be the raw full-body spec bytes")
    order = tuple(json.loads(spec_bytes)["coordinate_order"])
    seed = np.asarray(seed_q, dtype=float)
    if seed.shape != (len(order),) or not np.isfinite(seed).all():
        raise ValueError(f"seed_q must be {len(order)} finite values in spec order")
    opts = options or LiftIKOptions()

    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane

    kin = FullBodyMarkerKinematics(
        NativeMujocoFullBodyModel(bytes(spec_bytes)), LANDMARK_ATTACHMENTS
    )
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    locked = {n: float(seed[order.index(n)]) for n in UPPER_LIMB_COORDINATES}
    posture = dict.fromkeys(POSTURE_COORDINATES, (0.0, opts.posture_weight))
    seed_points = dict(zip(LANDMARK_LABELS, kin.marker_positions(seed), strict=True))
    upright = _standing_targets(seed_points, _foot_frame(kin, seed, ground))
    standing = kin.solve_pose(
        upright,
        np.isfinite(upright).all(axis=1),
        seed,
        locked=locked,
        flat_feet=STANCE_SPHERES,
        iterations=opts.standing_iterations,
        ground=ground,
        prior_weight=opts.prior_weight,
        posture_prior=posture,
    ).q
    frame = _foot_frame(kin, standing, ground)
    points = dict(zip(LANDMARK_LABELS, kin.marker_positions(standing), strict=True))
    targets = retarget_landmarks(solution, points, frame)
    locked |= {n: float(standing[order.index(n)]) for n in TOE_COORDINATES}
    q, fits = kin.solve_trajectory(
        targets,
        np.ones(targets.shape[:2], dtype=bool),
        standing,
        locked=locked,
        flat_feet_per_frame=[STANCE_SPHERES] * len(targets),
        iterations=opts.iterations,
        ground=ground,
        prior_weight=opts.prior_weight,
        posture_prior=posture,
    )
    residual, bar, closure, lowest = _fit_arrays(kin, fits)
    return FullBodyLift(
        time_s=solution.time_s.copy(),
        q=np.asarray(q, dtype=float),
        targets=targets,
        residual_m=residual,
        bar_m=bar,
        closure_m=closure,
        lowest_sphere_m=lowest,
        q_standing=standing,
        frame=frame,
        options=opts,
    )


def bar_path(plant: VectorPlant, q_rows: Array) -> Array:
    """World bar-landmark (``Hub`` frame) positions for each state row."""
    rows = np.asarray(q_rows, dtype=float)
    return np.array([plant.kinematic_frames(q)[BAR_FRAME][:3, 3] for q in rows])


def _window_indices(time_s: Array, start_s: float, window_s: float) -> int:
    if not (np.isfinite(window_s) and window_s > 0.0):
        raise ValueError("window_s must be positive and finite")
    if not (time_s[0] <= start_s and start_s + window_s <= time_s[-1] + 1e-12):
        raise ValueError(
            f"window [{start_s}, {start_s + window_s}] s lies outside the lift "
            f"[{time_s[0]}, {time_s[-1]}] s"
        )
    index = int(np.searchsorted(time_s, start_s - 1e-12))
    if abs(time_s[index] - start_s) > 1e-9:
        raise ValueError("window start must coincide with a solution sample time")
    return index


def build_lift_bundle(
    spec_bytes: bytes,
    solution: SagittalSolution,
    lift: FullBodyLift | None,
    *,
    lift_name: str,
    window_start_s: float,
    window_s: float,
    provenance: Mapping[str, Any] | None = None,
) -> InputBundle:
    """Same-input bundle for a window of the full-body lift.

    The window starts at a solution sample (``window_start_s``) and spans
    ``window_s``; MuJoCo tracks the joint trajectory from there with the
    pipeline controller held per step.  The provenance records the planar
    solution, the IK summary, the window's bar path and ``provenance``.

    Raises:
        ValueError: For a window outside the lift or a missing ``lift``.
    """
    start = _window_indices(solution.time_s, float(window_start_s), float(window_s))
    if lift is None or lift.q.shape[0] != solution.time_s.size:
        raise ValueError("lift must be the full-body trajectory of solution")
    if not lift_name:
        raise ValueError("lift_name must be non-empty")
    times = solution.time_s[start:]
    record = {
        "lift": lift_name,
        "spec_sha256": hashlib.sha256(bytes(spec_bytes)).hexdigest(),
        "optimizer": dict(solution.source),
        "sagittal_solution": solution.to_record(),
        "ik": lift.summary(),
        "ik_bar_path_m": lift.bar_m.tolist(),
        "window": {"start_s": float(times[0]), "duration_s": float(window_s)},
        "bar_frame": BAR_FRAME,
        "limitations": LIMITATIONS,
        **dict(provenance or {}),
    }
    bundle = generate_reference_bundle(
        bytes(spec_bytes),
        times,
        lift.q[start:],
        duration_s=float(window_s),
        provenance=record,
    )
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    window_bar = bar_path(plant, bundle.reference_q)
    return InputBundle(
        spec_bytes=bundle.spec_bytes,
        coordinate_order=bundle.coordinate_order,
        dt_s=bundle.dt_s,
        q0=bundle.q0,
        v0=bundle.v0,
        efforts=bundle.efforts,
        reference_q=bundle.reference_q,
        reference_v=bundle.reference_v,
        reference_engine=bundle.reference_engine,
        provenance={**bundle.provenance, "window_bar_path_m": window_bar.tolist()},
    )
