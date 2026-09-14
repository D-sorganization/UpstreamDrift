"""Ground-supported full-body pipeline on the MuJoCo full-body model (GS-0 to GS-5).

Stages, each receipted in ``receipt.json`` beside this script:

0. Functional hip calibration: both hip joints of the v2 specification are
   relocated to the functional centres estimated from the knee markers in the
   pelvis frame (``full_body_spec_hipcal.json``, its own hash).
1. Ground calibration from the lowest toe markers of the tour capture mapped
   into the native world (``(x, y, z) -> (x, -z, y)``), and per-frame stance
   detection (which contact spheres are on the ground) from marker heights
   relative to address.
2. Address pose: 41-coordinate IK on frame 0 from several leg seeds with the
   25 qualified upper-body markers, the 8 lower-limb markers (anatomical seed
   offsets), the grip closure and the stance spheres pinned to the plane.
3. Lower-limb marker calibration with the shared alternating algorithm on a
   decimated frame set with the stance pins; then a grid search over femur
   and tibia length scales (``segment_scaling``) judged by the same pinned IK;
   recalibration on the scaled document (``full_body_spec_hipcal_scaled.json``).
4. Full 654-frame IK, zero-phase low-pass, and a consistency re-solve that
   keeps the smoothed reference on the ground and the grip closed.
5. Forward dynamics with the shared contact law: computed-torque tracking of
   the reference with an unactuated root, the feet carrying the golfer.
   Marker RMS of the simulated motion, ground reaction, centre of pressure
   and joint torques are reported; playback GIFs are rendered.

Every number here is a milestone of a stated candidate, not acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter  # noqa: E402
from src.engines.physics_engines.mujoco.python import full_body_simulation as fs  # noqa: E402
from src.engines.physics_engines.mujoco.python.full_body_markers import (  # noqa: E402
    FullBodyMarkerKinematics,
)
from src.engines.physics_engines.mujoco.python.full_body_model import (  # noqa: E402
    NativeMujocoFullBodyModel,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_calibration import (  # noqa: E402
    calibrate_marker_offsets,
    static_marker_offsets,
)
from src.shared.python.motion_matching import posture_metrics as post  # noqa: E402
from src.shared.python.motion_matching.anthropometric_candidate import (  # noqa: E402
    anthropometric_candidate,
)
from src.shared.python.motion_matching.contact_law import GroundPlane  # noqa: E402
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    canonical_sha256,
    validate_full_body_spec,
)
from src.shared.python.motion_matching.ground_support import (  # noqa: E402
    calibrate_ground_height,
    capture_to_native_world,
)
from src.shared.python.motion_matching.hip_calibration import (  # noqa: E402
    apply_hip_calibration,
    functional_hip_calibration,
)
from src.shared.python.motion_matching.segment_scaling import scale_segments  # noqa: E402
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    MARKER_SEGMENTS,
    TourCapture,
    load_tour_capture,
)

HERE = Path(__file__).resolve().parent
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"
BUILD_RECEIPT = ROOT / "docs/development/full_body_models/build_receipt_v2.json"
UPPER_SPEC = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
CANDIDATE = (
    ROOT
    / "docs/development/full_body_models/evidence/native_candidates/returned81_candidate.json"
)
C3D = ROOT / "data/C3D_TA_Driver.c3d"
OUT = HERE  # overridden by --out
UP_AXIS, FORWARD_AXIS, RIGHT_AXIS = (
    np.array([0.0, 0.0, 1.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
)
TOE_STANDOFF_M = 0.03  # marker centre above the sole (shoe upper plus marker radius)
# The v2 document carries heel and metatarsal-head spheres only (15 cm of
# support); the golfer's centre of mass at address lies 3 to 6 cm ahead of
# the metatarsal heads, so a toe sphere per foot (rigid with the calcaneus,
# bottom level with the others) extends the support polygon to the toe tips.
TOE_SPHERES = {
    "toe_r": ("calcn_r", (0.23, -0.010, 0.0), 0.025),
    "toe_l": ("calcn_l", (0.23, -0.010, 0.0), 0.025),
}
LEG_SEEDS = {  # OpenSim body frames: x forward, y up the segment, z to the right
    "RKneeOut": ("femur_r", (0.0, -0.40, 0.06)),
    "LKneeOut": ("femur_l", (0.0, -0.39, -0.06)),
    "RAnkleOut": ("tibia_r", (-0.01, -0.44, 0.055)),
    "LAnkleOut": ("tibia_l", (-0.01, -0.43, -0.055)),
    "RToeIn": ("calcn_r", (0.19, 0.03, -0.03)),
    "RToeOut": ("calcn_r", (0.17, 0.03, 0.06)),
    "LToeIn": ("calcn_l", (0.19, 0.03, 0.03)),
    "LToeOut": ("calcn_l", (0.17, 0.03, -0.06)),
}
LEG_LABELS = tuple(LEG_SEEDS)
# Rajagopal 2016 coordinate ranges (degrees) for the retained lower-limb joints;
# knee flexion is negative in that model. The IK projects onto these boxes
# widened by BOUND_WIDENING: the hip zero orientation is not anatomically
# calibrated yet, so a constant offset would otherwise consume the range and
# pin joints on their bounds (whole-swing RMS 54 mm at 1x, 28 mm at 2x).
LOWER_LIMB_RANGES_DEG = {
    "hip_flexion": (-30.0, 120.0),
    "hip_adduction": (-50.0, 30.0),
    "hip_rotation": (-40.0, 40.0),
    "knee_angle": (-120.0, 10.0),
    "ankle_angle": (-40.0, 30.0),
    "subtalar_angle": (-20.0, 20.0),
    "mtp_angle": (-30.0, 30.0),
}
BOUND_WIDENING = 2.0
# Address IK is solved from several leg seeds (degrees) and the best marker fit
# kept, so a local minimum with a joint pinned on its bound is not accepted.
ADDRESS_SEEDS_DEG = [
    {"hip_flexion": flexion, "knee_angle": knee, "hip_rotation": rotation}
    for flexion, knee in ((20.0, -20.0), (45.0, -25.0), (65.0, -30.0))
    for rotation in (-20.0, 0.0, 20.0)
]
STANCE_TOLERANCE_M = 0.02  # marker within this height of its address height: on ground
REFERENCE_CUTOFF_HZ = 12.0  # zero-phase low-pass on the IK reference before tracking
CONSISTENCY_PRIOR = 0.1  # weight pulling the re-solve toward the smoothed reference
CALIBRATION_STRIDE = 6
STATIC_FRAMES = 24  # first 0.067 s of address treated as the static trial
# The native chain has no neck: the head turns relative to the thorax during a
# swing, so its six markers get a low weight in the fit (errors still reported).
HEAD_MARKER_WEIGHT = 0.1
TRAJECTORY_RESTARTS = 4  # perturbed re-solves for frames above the threshold
TRAJECTORY_RESTART_THRESHOLD_M = 0.03
# Neutral address for the static trial: scapulae undepressed, spine nearly straight.
NEUTRAL_LOCKS = {f"{s}ScapInput{a}": 0.0 for s in "LR" for a in "XY"}
NEUTRAL_BOUNDS_DEG = {"SpineInputX": (-5.0, 5.0), "SpineInputY": (-10.0, 10.0)}
CALIBRATION_ITERATIONS = 3
CALIBRATION_PRIOR_FRAMES = 40.0  # anatomical seed weight in frame-equivalents
SCALE_GRID = (0.94, 0.97, 1.00)  # femur and tibia length scales searched
DT_S = 1e-3
OMEGA_RAD_S = 30.0
BALANCE = (60.0, 15.0)
RATE_HZ = 360.0
PLAYBACK_STRIDE = 6  # 60 Hz playback keeps the GIFs near 2 MB
PRIOR = 1e-3


def segment_rms(
    labels: tuple[str, ...], errors: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    out: dict[str, float] = {}
    for segment, members in MARKER_SEGMENTS.items():
        cols = [labels.index(m) for m in members if m in labels]
        if not cols:
            continue
        e, v = errors[:, cols], valid[:, cols]
        out[segment] = float(np.sqrt(np.mean(e[v] ** 2))) if v.any() else float("nan")
    return out


def stance_spheres(
    points: np.ndarray, valid: np.ndarray, labels: tuple[str, ...]
) -> list[tuple[str, ...]]:
    """Per frame, the contact spheres judged to be on the ground from marker heights.

    Both feet are flat at address (frame 0). A heel sphere stays in stance
    while that side's ankle marker sits within ``STANCE_TOLERANCE_M`` of its
    address height; a forefoot sphere while both toe markers do. Invalid
    samples count as lifted.
    """
    heights = np.where(valid, points[:, :, 2], np.nan)
    low = np.where(
        np.isfinite(heights), heights - heights[0] < STANCE_TOLERANCE_M, False
    )

    def column(label: str) -> np.ndarray:
        return low[:, labels.index(label)]

    out: list[tuple[str, ...]] = []
    for k in range(points.shape[0]):
        pinned: list[str] = []
        for side in ("r", "l"):
            prefix = side.upper()
            if column(f"{prefix}AnkleOut")[k]:
                pinned.append(f"heel_{side}")
            if column(f"{prefix}ToeIn")[k] and column(f"{prefix}ToeOut")[k]:
                pinned.append(f"forefoot_{side}")
                pinned.append(f"toe_{side}")
        out.append(tuple(pinned))
    return out


CONTACT_STIFFNESS_N_M = 2.0e5  # placeholder 5e4 sank 20 to 50 mm under swing loads


def add_toe_spheres(document: dict) -> dict:
    """Return a copy of ``document`` with toe spheres and the stiffer contact law."""
    contact = dict(document["contact"])
    contact["parameters"] = {
        **contact["parameters"],
        "stiffness_n_m": CONTACT_STIFFNESS_N_M,
    }
    names = {sphere["name"] for sphere in contact["spheres"]}
    extra = [
        {"name": name, "body": body, "position_m": list(position), "radius_m": radius}
        for name, (body, position, radius) in TOE_SPHERES.items()
        if name not in names
    ]
    contact["spheres"] = list(contact["spheres"]) + extra
    out = dict(document)
    out["contact"] = contact
    out["provenance"] = str(document.get("provenance", "")) + (
        " | toe contact spheres added on the calcanei (support polygon to the toe"
        f" tips); contact stiffness {CONTACT_STIFFNESS_N_M:.0e} N/m"
    )
    return out


def smooth_reference(q: np.ndarray, rate_hz: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass of every coordinate (edge-padded)."""
    b, a = butter(4, cutoff_hz / (0.5 * rate_hz))
    return filtfilt(b, a, q, axis=0, padlen=min(60, q.shape[0] - 1))


def marker_errors(
    kin: FullBodyMarkerKinematics, q: np.ndarray, points: np.ndarray
) -> np.ndarray:
    return np.array(
        [
            np.linalg.norm(kin.marker_positions(row) - target, axis=1)
            for row, target in zip(q, points, strict=True)
        ]
    )


def scaled_offsets(offsets: dict, femur: float, tibia: float) -> dict:
    """Leg marker offsets carried onto scaled femur and tibia bodies."""
    out = {}
    for label, (body, offset) in offsets.items():
        factor = (
            femur
            if body.startswith("femur")
            else tibia
            if body.startswith("tibia")
            else 1.0
        )
        out[label] = (body, tuple(float(v) for v in factor * np.asarray(offset)))
    return out


def render_playback(
    spec_bytes: bytes,
    names: tuple[str, ...],
    q: np.ndarray,
    lookat: np.ndarray,
    path: Path,
) -> None:
    xml, _ = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    addresses = [model.joint(n).qposadr[0] for n in names]
    renderer = mujoco.Renderer(model, 240, 320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -12.0
    frames_out = []
    for k in range(0, q.shape[0], PLAYBACK_STRIDE):
        data.qpos[addresses] = q[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames_out.append(renderer.render().copy())
    imageio.mimsave(path, frames_out, duration=1000 * PLAYBACK_STRIDE / RATE_HZ, loop=0)


def document_bounds(document: dict) -> dict[str, tuple[float, float]]:
    """Radian bounds declared by an anthropometric document (empty otherwise)."""
    return {
        name: (np.radians(lo), np.radians(hi))
        for name, (lo, hi) in document.get("coordinate_ranges_deg", {}).items()
    }


def document_seed(document: dict, kin: FullBodyMarkerKinematics) -> np.ndarray:
    """Start pose: the document's address seed (anthropometric geometry) or
    the qualified candidate's coordinates (native geometry)."""
    q = np.zeros(len(kin.coordinate_order))
    candidate = json.loads(CANDIDATE.read_text())
    seed = document.get("address_seed_deg")
    for name, value in zip(candidate["coordinate_names"], candidate["q0"], strict=True):
        if seed is None or name.startswith(("Translation", "HipInput")):
            q[kin.coordinate_order.index(name)] = value
    for name, value in (seed or {}).items():
        q[kin.coordinate_order.index(name)] = np.radians(value)
    return q


class Lane:
    """Capture, ground, stance and bounds shared by every stage."""

    def __init__(self, labels: tuple[str, ...]) -> None:
        self.labels = labels
        capture = load_tour_capture(C3D).subset(labels)
        self.points = capture_to_native_world(capture.points_m)
        self.valid = capture.valid.copy()
        self.times = np.asarray(capture.time_s, dtype=float)
        self.frames = int(capture.frames)
        self.ground_cal = calibrate_ground_height(
            self.points, self.valid, labels, LEG_LABELS[4:], standoff_m=TOE_STANDOFF_M
        )
        self.ground = GroundPlane(
            normal=(0.0, 0.0, 1.0), height_m=self.ground_cal.height_m
        )
        self.stance = stance_spheres(self.points, self.valid, labels)
        self.bounds = {
            f"{joint}_{side}": (
                np.radians(lo * BOUND_WIDENING),
                np.radians(hi * BOUND_WIDENING),
            )
            for joint, (lo, hi) in LOWER_LIMB_RANGES_DEG.items()
            for side in ("r", "l")
        }
        self.calibration_frames = list(range(0, self.frames, CALIBRATION_STRIDE))
        self.marker_weights = {
            label: HEAD_MARKER_WEIGHT
            for label in MARKER_SEGMENTS["head"]
            if label in labels
        }

    def kinematics(
        self, spec_bytes: bytes, attachments: dict
    ) -> tuple[NativeMujocoFullBodyModel, FullBodyMarkerKinematics]:
        adapter = NativeMujocoFullBodyModel(spec_bytes)
        if abs(adapter.ground_plane.height_m - self.ground.height_m) > 1e-3:
            raise ValueError("Spec ground height disagrees with the toe calibration")
        ordered = {label: attachments[label] for label in self.labels}
        return adapter, FullBodyMarkerKinematics(adapter, ordered)

    def best_address(
        self,
        kin: FullBodyMarkerKinematics,
        base: np.ndarray,
        *,
        neutral: bool = False,
    ):
        """Best address fit over the leg seeds; ``neutral`` locks the scapulae
        and bounds the spine to a straight-torso address."""
        bounds = dict(self.bounds)
        locked = None
        if neutral:
            locked = dict(NEUTRAL_LOCKS)
            bounds |= {
                name: (np.radians(lo), np.radians(hi))
                for name, (lo, hi) in NEUTRAL_BOUNDS_DEG.items()
            }
        best = None
        for seed in ADDRESS_SEEDS_DEG:
            start = base.copy()
            for joint, value in seed.items():
                for side in ("r", "l"):
                    start[kin.coordinate_order.index(f"{joint}_{side}")] = np.radians(
                        value
                    )
            fit = kin.solve_pose(
                self.points[0],
                self.valid[0],
                start,
                ground=self.ground,
                prior_weight=PRIOR,
                iterations=100,
                flat_feet=self.stance[0],
                bounds=bounds,
                locked=locked,
                marker_weights=self.marker_weights,
            )
            if best is None or fit.marker_rms_m < best.marker_rms_m:
                best = fit
        assert best is not None
        return best

    def static_offsets(
        self, kin: FullBodyMarkerKinematics, q: np.ndarray, seeds: dict
    ) -> dict:
        """Static-trial placement of every seed marker at pose ``q`` over the
        first ``STATIC_FRAMES`` frames (bodies as in ``seeds``)."""
        frames = list(range(STATIC_FRAMES))
        seen = {
            label
            for label in seeds
            if self.valid[frames, self.labels.index(label)].any()
        }
        cols = [self.labels.index(m) for m in seen]
        capture = TourCapture(
            time_s=self.times[frames],
            labels=tuple(seen),
            points_m=self.points[np.ix_(frames, cols)],
            valid=self.valid[np.ix_(frames, cols)],
        )
        bodies = {label: seeds[label][0] for label in seen}
        poses = kin.body_poses(q, sorted(set(bodies.values())))
        placed = static_marker_offsets(capture, bodies, [poses] * len(frames))
        # Markers absent from the static frames keep their anatomical seed.
        return {label: placed.get(label, seed) for label, seed in seeds.items()}

    def trajectory(
        self, kin: FullBodyMarkerKinematics, q_start: np.ndarray, frames=None
    ):
        return kin.solve_trajectory(
            self.points,
            self.valid,
            q_start,
            ground=self.ground,
            frames=frames,
            prior_weight=PRIOR,
            flat_feet_per_frame=self.stance,
            plant_stance=True,
            bounds=self.bounds,
            marker_weights=self.marker_weights,
            restarts=TRAJECTORY_RESTARTS,
            restart_threshold_m=TRAJECTORY_RESTART_THRESHOLD_M,
        )

    def calibrate_legs(
        self, spec_bytes: bytes, upper: dict, seeds: dict, q_start: np.ndarray
    ):
        """Alternating marker calibration of ``seeds`` (others fixed) with stance pins."""
        frames = self.calibration_frames
        calibrated = tuple(seeds)
        cols = [self.labels.index(m) for m in calibrated]
        leg_capture = TourCapture(
            time_s=self.times[frames],
            labels=calibrated,
            points_m=self.points[np.ix_(frames, cols)],
            valid=self.valid[np.ix_(frames, cols)],
        )
        leg_bodies = {label: body for label, (body, _) in seeds.items()}
        adapter, kin0 = self.kinematics(spec_bytes, {**upper, **seeds})
        state = {"kin": kin0}

        def pose_fn(q: np.ndarray) -> dict:
            return state["kin"].body_poses(q, sorted(set(leg_bodies.values())))

        def ik_fn(offsets: dict, cap: TourCapture) -> np.ndarray:
            merged = {**upper, **{k: (b, tuple(o)) for k, (b, o) in offsets.items()}}
            state["kin"] = FullBodyMarkerKinematics(
                adapter, {label: merged[label] for label in self.labels}
            )
            return self.trajectory(state["kin"], q_start, frames=frames)[0]

        result = calibrate_marker_offsets(
            leg_capture,
            leg_bodies,
            pose_fn,
            ik_fn,
            initial_q=q_start,
            iterations=CALIBRATION_ITERATIONS,
            prior_offsets={label: offset for label, (_, offset) in seeds.items()},
            prior_weight=CALIBRATION_PRIOR_FRAMES,
        )
        offsets = {
            label: (body, tuple(offset))
            for label, (body, offset) in result.offsets.items()
        }
        return offsets, result

    def pinned_rms(
        self, spec_bytes: bytes, attachments: dict, q_start: np.ndarray
    ) -> float:
        _, kin = self.kinematics(spec_bytes, attachments)
        frames = self.calibration_frames
        q, _ = self.trajectory(kin, q_start, frames=frames)
        errors = marker_errors(kin, q, self.points[frames])
        return float(np.sqrt(np.mean(errors[self.valid[frames]] ** 2)))


def posture_summary(kin: FullBodyMarkerKinematics, q: np.ndarray) -> dict:
    """Spine bend and clavicle-link angles of the model at one pose."""
    adapter = kin.adapter
    m, d = adapter.model, adapter.data
    kin.marker_positions(q)

    def site(frame: str) -> np.ndarray:
        return d.site_xpos[m.site(adapter.metadata["frame_sites"][frame]).id].copy()

    hips = (
        d.xanchor[m.joint("hip_flexion_r").id] + d.xanchor[m.joint("hip_flexion_l").id]
    ) / 2
    spine, hub = site("Spine"), site("Hub")
    bend = post.spine_bend(spine - hips, hub - spine, UP_AXIS, FORWARD_AXIS, RIGHT_AXIS)
    links = {}
    for side in ("L", "R"):
        v = site(f"{side}S") - hub
        links[side] = float(np.degrees(np.arcsin(-v[2] / np.linalg.norm(v))))
    return {
        "spine_bend_deg": bend.__dict__,
        "clavicle_link_below_horizontal_deg": links,
        "hips_to_shoulder_centre_m": float(
            np.linalg.norm((site("LS") + site("RS")) / 2 - hips)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument(
        "--spec",
        type=Path,
        default=SPEC,
        help="full-body document to start from (default: the qualified v2)",
    )
    parser.add_argument(
        "--skip-hip-calibration",
        action="store_true",
        help="keep the document's hip joints (anthropometric geometry places them)",
    )
    parser.add_argument(
        "--anthropometric",
        nargs=2,
        type=float,
        metavar=("STATURE_M", "MASS_KG"),
        help="build the de Leva candidate (unqualified) before calibration",
    )
    parser.add_argument(
        "--recalibrate-upper",
        action="store_true",
        help="calibrate the 25 upper-body offsets too (qualified offsets as prior)",
    )
    parser.add_argument(
        "--static-seeds",
        action="store_true",
        help="replace the seed offsets by a neutral-spine static-trial placement",
    )
    args = parser.parse_args()
    global OUT
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    hipcal_path = OUT / "full_body_spec_hipcal.json"
    scaled_path = OUT / "full_body_spec_hipcal_scaled.json"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("ground_support")
    t_start = time.perf_counter()
    base_spec = json.loads(args.spec.read_text())
    unqualified = "unqualified" in str(base_spec.get("upper_body_qualification", ""))
    upper_base = json.loads(UPPER_SPEC.read_text())
    candidate = json.loads(CANDIDATE.read_text())
    upper = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in base_spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    labels = tuple({**upper, **LEG_SEEDS})
    lane = Lane(labels)
    lane.bounds |= document_bounds(base_spec)

    # 0. functional hip calibration
    waist_offsets = {
        label: base_spec["marker_attachments"][label]["offset_m"]
        for label in MARKER_SEGMENTS["pelvis"]
    }
    hip_cal = functional_hip_calibration(lane.points, lane.valid, labels, waist_offsets)
    if args.skip_hip_calibration:
        hip_spec = add_toe_spheres(base_spec)
    else:
        alignment_old = json.loads(BUILD_RECEIPT.read_text())["pelvis_alignment"][
            "hip_from_opensim_pelvis"
        ]
        hip_spec = add_toe_spheres(
            apply_hip_calibration(base_spec, hip_cal, alignment_old)
        )
    if args.anthropometric:
        stature_m, mass_kg = args.anthropometric
        hip_spec = anthropometric_candidate(
            hip_spec, stature_m=stature_m, mass_kg=mass_kg
        )
        qualification_note = (
            f"anthropometric candidate ({stature_m:.3f} m, {mass_kg:.1f} kg), "
            "unqualified: upper-body lengths, masses and inertias changed"
        )
    elif unqualified:
        qualification_note = str(base_spec["upper_body_qualification"])
    else:
        validate_full_body_spec(hip_spec, upper_base)
        qualification_note = "qualified upper body with functional hips"
    hipcal_path.write_text(json.dumps(hip_spec, indent=2, sort_keys=True) + "\n")
    hip_bytes = hipcal_path.read_bytes()
    if args.recalibrate_upper:
        seeds_all = {**upper, **LEG_SEEDS}
        fixed: dict = {}
    else:
        seeds_all = dict(LEG_SEEDS)
        fixed = dict(upper)
    hip_report = {
        "spec_sha256": canonical_sha256(hip_spec),
        "centre_r_hip_frame_m": hip_cal.centre_r,
        "centre_l_hip_frame_m": hip_cal.centre_l,
        "radius_r_m": hip_cal.radius_r_m,
        "radius_l_m": hip_cal.radius_l_m,
        "sphere_sd_r_m": hip_cal.residual_sd_r_m,
        "sphere_sd_l_m": hip_cal.residual_sd_l_m,
        "frames": hip_cal.frames,
        "pelvis_axes_in_hip_frame": hip_cal.pelvis_axes,
        "waist_fit_max_residual_m": hip_cal.waist_fit_max_residual_m,
    }
    stance_fraction = {
        name: float(np.mean([name in s for s in lane.stance]))
        for name in ("heel_r", "forefoot_r", "toe_r", "heel_l", "forefoot_l", "toe_l")
    }

    # 2. address pose with seed leg offsets
    adapter, kin = lane.kinematics(hip_bytes, {**upper, **LEG_SEEDS})
    q_seed = document_seed(base_spec, kin)
    address = lane.best_address(kin, q_seed)
    address_report = {
        "seed_offsets": {
            "marker_rms_m": address.marker_rms_m,
            "segment_rms_m": segment_rms(
                labels,
                marker_errors(kin, address.q[None, :], lane.points[:1]),
                lane.valid[:1],
            ),
        },
        "stance_spheres": lane.stance[0],
    }
    if args.static_seeds:
        neutral = lane.best_address(kin, q_seed, neutral=True)
        seeds_all = lane.static_offsets(kin, neutral.q, seeds_all)
        adapter, kin = lane.kinematics(hip_bytes, {**fixed, **seeds_all})
        address = lane.best_address(kin, neutral.q)
        address_report["static_trial"] = {
            "frames": STATIC_FRAMES,
            "neutral_fit_rms_m": neutral.marker_rms_m,
            "neutral_posture": posture_summary(kin, neutral.q),
            "marker_rms_m": address.marker_rms_m,
            "posture": posture_summary(kin, address.q),
        }
        log.info(
            "static trial: neutral fit %.1f mm, address with static seeds %.1f mm",
            neutral.marker_rms_m * 1e3,
            address.marker_rms_m * 1e3,
        )

    # 3. leg calibration, segment scale search, recalibration
    offsets, calibration = lane.calibrate_legs(hip_bytes, fixed, seeds_all, address.q)
    scale_table = []
    best = (float("inf"), 1.0, 1.0, hip_spec)
    for femur in SCALE_GRID:
        for tibia in SCALE_GRID:
            scales = {f"femur_{s}": femur for s in "rl"} | {
                f"tibia_{s}": tibia for s in "rl"
            }
            doc = scale_segments(hip_spec, scales)
            doc_bytes = json.dumps(doc, sort_keys=True).encode()
            rms = lane.pinned_rms(
                doc_bytes, {**fixed, **scaled_offsets(offsets, femur, tibia)}, address.q
            )
            scale_table.append({"femur": femur, "tibia": tibia, "pinned_rms_m": rms})
            log.info(
                "scale femur %.2f tibia %.2f: pinned RMS %.1f mm",
                femur,
                tibia,
                rms * 1e3,
            )
            if rms < best[0]:
                best = (rms, femur, tibia, doc)
    _, femur_scale, tibia_scale, scaled_spec = best
    if not args.anthropometric and not unqualified:
        validate_full_body_spec(scaled_spec, upper_base)
    scaled_path.write_text(json.dumps(scaled_spec, indent=2, sort_keys=True) + "\n")
    spec_bytes = scaled_path.read_bytes()
    offsets, calibration2 = lane.calibrate_legs(
        spec_bytes, fixed, scaled_offsets(offsets, femur_scale, tibia_scale), address.q
    )
    attachments = {**fixed, **offsets}
    adapter, kin = lane.kinematics(spec_bytes, attachments)
    sim = fs.FullBodySimulator(adapter)
    address2 = lane.best_address(kin, address.q)
    address_report["calibrated"] = {
        "marker_rms_m": address2.marker_rms_m,
        "segment_rms_m": segment_rms(
            labels,
            marker_errors(kin, address2.q[None, :], lane.points[:1]),
            lane.valid[:1],
        ),
        "closure_error_m": address2.closure_error_m,
        "lowest_sphere_height_m": address2.lowest_sphere_height_m,
        "support_offset_m": kin.support_offset(address2.q, lane.ground),
        "leg_angles_deg": {
            name: float(np.degrees(address2.q[kin.coordinate_order.index(name)]))
            for name in kin.coordinate_order[adapter.upper_body_coordinates :]
        },
        "posture": posture_summary(kin, address2.q),
    }

    # 4. full IK, smoothing, consistency re-solve
    q_ik, fits = lane.trajectory(kin, address2.q)
    errors = marker_errors(kin, q_ik, lane.points)
    q_smooth = smooth_reference(q_ik, RATE_HZ, REFERENCE_CUTOFF_HZ)
    q_ref, ref_fits = kin.solve_trajectory(
        lane.points,
        lane.valid,
        q_smooth[0],
        ground=lane.ground,
        prior_weight=CONSISTENCY_PRIOR,
        iterations=30,
        flat_feet_per_frame=lane.stance,
        plant_stance=True,
        prior_trajectory=q_smooth,
        bounds=lane.bounds,
    )
    ref_errors = marker_errors(kin, q_ref, lane.points)
    heights = np.array([min(kin.sphere_heights(q, lane.ground).values()) for q in q_ik])
    ref_heights = np.array(
        [min(kin.sphere_heights(q, lane.ground).values()) for q in q_ref]
    )
    ik_report = {
        "frames": lane.frames,
        "marker_rms_m": float(np.sqrt(np.mean(errors[lane.valid] ** 2))),
        "segment_rms_m": segment_rms(labels, errors, lane.valid),
        "closure_error_max_m": float(max(f.closure_error_m for f in fits)),
        "lowest_sphere_height_min_m": float(heights.min()),
        "lowest_sphere_height_max_m": float(heights.max()),
        "reference": {
            "cutoff_hz": REFERENCE_CUTOFF_HZ,
            "consistency_prior": CONSISTENCY_PRIOR,
            "marker_rms_m": float(np.sqrt(np.mean(ref_errors[lane.valid] ** 2))),
            "segment_rms_m": segment_rms(labels, ref_errors, lane.valid),
            "closure_error_max_m": float(max(f.closure_error_m for f in ref_fits)),
            "lowest_sphere_height_min_m": float(ref_heights.min()),
            "lowest_sphere_height_max_m": float(ref_heights.max()),
            "max_deviation_from_smoothed_rad": float(
                np.abs(q_ref[:, 6:] - q_smooth[:, 6:]).max()
            ),
            "max_joint_speed_rad_s": float(
                np.abs(np.gradient(q_ref[:, 6:], lane.times, axis=0)).max()
            ),
            "stance_sphere_drift_max_m": float(
                max(
                    np.linalg.norm(
                        kin.sphere_ground_points(q_ref[k], lane.ground)[name]
                        - kin.sphere_ground_points(q_ref[0], lane.ground)[name]
                    )
                    for k in range(lane.frames)
                    for name in lane.stance[k]
                    if all(name in lane.stance[j] for j in range(k + 1))
                )
            ),
        },
        "calibration": {
            "stride": CALIBRATION_STRIDE,
            "frames": len(lane.calibration_frames),
            "prior_frames": CALIBRATION_PRIOR_FRAMES,
            "prior_offsets_m": {k: list(v[1]) for k, v in LEG_SEEDS.items()},
            "rms_per_iteration_m": list(calibration.rms_per_iteration_m),
            "rms_per_iteration_after_scaling_m": list(calibration2.rms_per_iteration_m),
            "per_marker_rms_m": calibration2.per_marker_rms_m,
            "offsets_m": {
                k: {"body": b, "offset_m": list(o)} for k, (b, o) in offsets.items()
            },
        },
        "segment_scaling": {
            "grid": SCALE_GRID,
            "table": scale_table,
            "femur_scale": femur_scale,
            "tibia_scale": tibia_scale,
            "spec_sha256": canonical_sha256(scaled_spec),
        },
        "joint_ranges_deg": LOWER_LIMB_RANGES_DEG,
        "bound_widening": BOUND_WIDENING,
        "leg_angle_ranges_deg": {
            name: [
                float(np.degrees(q_ref[:, kin.coordinate_order.index(name)].min())),
                float(np.degrees(q_ref[:, kin.coordinate_order.index(name)].max())),
            ]
            for name in kin.coordinate_order[adapter.upper_body_coordinates :]
        },
    }
    np.savez(
        OUT / "ik_trajectory.npz",
        time_s=lane.times,
        q=q_ik,
        q_ref=q_ref,
        errors_m=errors,
        ref_errors_m=ref_errors,
        valid=lane.valid,
    )

    # 5. forward dynamics tracking
    q0 = fs.preload_feet(sim, q_ref[0])
    v0 = np.gradient(q_ref, lane.times, axis=0)[0]
    controller = fs.tracking_controller(
        sim, lane.times, q_ref, omega_rad_s=OMEGA_RAD_S, zeta=1.0, balance=BALANCE
    )
    record = sim.run(
        q0,
        v0,
        controller,
        duration_s=float(lane.times[-1]),
        dt_s=DT_S,
        record_every=int(round(1.0 / (RATE_HZ * DT_S))),
    )
    sim_q = np.array(
        [
            [np.interp(t, record.time_s, record.q[:, k]) for k in range(sim.nv)]
            for t in lane.times
        ]
    )
    sim_errors = marker_errors(kin, sim_q, lane.points)
    dynamics_report = {
        "duration_s": float(lane.times[-1]),
        "dt_s": DT_S,
        "controller": {
            "type": "computed torque tracking, unactuated root",
            "omega_rad_s": OMEGA_RAD_S,
            "zeta": 1.0,
            "balance": BALANCE,
        },
        "marker_rms_m": float(np.sqrt(np.mean(sim_errors[lane.valid] ** 2))),
        "segment_rms_m": segment_rms(labels, sim_errors, lane.valid),
        "joint_tracking_rms_rad": float(
            np.sqrt(np.mean((sim_q[:, 6:] - q_ref[:, 6:]) ** 2))
        ),
        "root_tracking_rms_m": float(
            np.sqrt(np.mean((sim_q[:, :3] - q_ref[:, :3]) ** 2))
        ),
        "weight_fraction": {
            "min": float(record.weight_fraction.min()),
            "max": float(record.weight_fraction.max()),
            "mean": float(record.weight_fraction.mean()),
        },
        "inside_support_polygon_fraction": float(record.inside_support_polygon.mean()),
        "root_error_timeline_m": {
            f"{t:.2f}": float(
                np.linalg.norm(
                    sim_q[int(round(t * RATE_HZ)), :3]
                    - q_ref[int(round(t * RATE_HZ)), :3]
                )
            )
            for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75)
        },
        "backswing_to_1s": {
            "root_error_max_m": float(
                np.linalg.norm(sim_q[:361, :3] - q_ref[:361, :3], axis=1).max()
            ),
            "marker_rms_m": float(
                np.sqrt(np.mean(sim_errors[:361][lane.valid[:361]] ** 2))
            ),
            "weight_fraction_min": float(
                record.weight_fraction[record.time_s <= 1.0].min()
            ),
            "weight_fraction_max": float(
                record.weight_fraction[record.time_s <= 1.0].max()
            ),
        },
        "peak_joint_torque_n_m": float(np.abs(record.tau).max()),
        "lowest_sphere_height_min_m": float(record.lowest_sphere_height_m.min()),
        "lowest_sphere_height_max_m": float(record.lowest_sphere_height_m.max()),
    }
    np.savez(
        OUT / "dynamics_record.npz",
        time_s=record.time_s,
        q=record.q,
        v=record.v,
        tau=record.tau,
        normal_force_n=record.normal_force_n,
        weight_fraction=record.weight_fraction,
        cop_m=record.centre_of_pressure_m,
        inside=record.inside_support_polygon,
        lowest_sphere_height_m=record.lowest_sphere_height_m,
        sim_errors_m=sim_errors,
    )
    lookat = np.nanmean(lane.points[0], axis=0)
    names = tuple(kin.coordinate_order)
    render_playback(spec_bytes, names, q_ref, lookat, OUT / "ik_playback.gif")
    render_playback(spec_bytes, names, sim_q, lookat, OUT / "tracking_playback.gif")

    receipt = {
        "base_spec_sha256": canonical_sha256(base_spec),
        "base_spec_file": args.spec.name,
        "spec_file": scaled_path.name,
        "hipcal_spec_file": hipcal_path.name,
        "recalibrate_upper": bool(args.recalibrate_upper),
        "anthropometric": list(args.anthropometric) if args.anthropometric else None,
        "posture_top_of_backswing": posture_summary(kin, q_ref[int(0.83 * RATE_HZ)]),
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "hip_calibration": hip_report,
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "capture_sha256": hashlib.sha256(C3D.read_bytes()).hexdigest(),
        "labels": labels,
        "ground": {
            "height_m": lane.ground_cal.height_m,
            "lowest_toe_marker_m": lane.ground_cal.lowest_marker_height_m,
            "standoff_m": lane.ground_cal.standoff_m,
            "policy": lane.ground_cal.policy,
            "stance_tolerance_m": STANCE_TOLERANCE_M,
            "stance_rule": "heights relative to address; both feet flat at frame 0",
            "toe_spheres": {
                k: {"body": b, "position_m": list(p), "radius_m": r}
                for k, (b, p, r) in TOE_SPHERES.items()
            },
            "stance_fraction": stance_fraction,
        },
        "address": address_report,
        "ik": ik_report,
        "dynamics": dynamics_report,
        "elapsed_s": time.perf_counter() - t_start,
        "qualification": (
            "kinematic IK and computed-torque tracking milestone on the MuJoCo "
            f"full-body model ({qualification_note}); not a fit, not acceptance"
        ),
    }
    (OUT / "receipt.json").write_text(
        json.dumps(receipt, indent=2, default=float) + "\n"
    )
    log.info(
        json.dumps(
            {k: receipt[k] for k in ("address", "dynamics")}, indent=1, default=float
        )
    )
    log.info(
        "ik %s",
        json.dumps(
            {
                k: ik_report[k]
                for k in (
                    "marker_rms_m",
                    "segment_rms_m",
                    "reference",
                    "segment_scaling",
                    "leg_angle_ranges_deg",
                )
            },
            indent=1,
            default=float,
        ),
    )
    log.info(
        "calibration rms %s -> scaled %s",
        calibration.rms_per_iteration_m,
        calibration2.rms_per_iteration_m,
    )


if __name__ == "__main__":
    main()
