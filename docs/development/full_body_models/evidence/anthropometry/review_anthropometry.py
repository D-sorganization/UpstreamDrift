"""Anthropometry and posture review of the full-body model against the capture.

Writes ``receipt.json`` and ``REVIEW.md`` beside this script with:

1. Golfer dimensions read straight from the tour capture (marker distances,
   functional hip radii, stature estimate through de Leva proportions).
2. Model dimensions and inertial parameters: the native upper body (Simscape
   qualified geometry and solids) and the Rajagopal legs as used by the
   ground-support driver, per body mass and the total.
3. de Leva expectations for the estimated stature at candidate body masses.
4. Posture: pelvis and trunk tilt and the spine bend, forward and lateral,
   from the markers (model-free) and from the model along the ground-support
   reference, at address and through the swing.

Numbers only; no model is edited here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python.full_body_model import (  # noqa: E402
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching import anthropometry as anth  # noqa: E402
from src.shared.python.motion_matching import posture_metrics as post  # noqa: E402
from src.shared.python.motion_matching.ground_support import (  # noqa: E402
    capture_to_native_world,
)
from src.shared.python.motion_matching.hip_calibration import (  # noqa: E402
    functional_hip_calibration,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    MARKER_SEGMENTS,
    load_tour_capture,
    tracked_labels,
)

HERE = Path(__file__).resolve().parent
GS = ROOT / "docs/development/full_body_models/evidence/ground_support"
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
C3D = ROOT / "data/C3D_TA_Driver.c3d"
UP, FORWARD, RIGHT = (
    np.array([0.0, 0, 1]),
    np.array([-1.0, 0, 0]),
    np.array([0, 1.0, 0]),
)
CANDIDATE_MASSES_KG = (73.0, 80.0, 85.0)
KNEE_MARKER_LATERAL_M = 0.055  # lateral epicondyle to knee centre
ANKLE_MARKER_LATERAL_M = 0.045


def marker_geometry(
    points: np.ndarray, valid: np.ndarray, labels: tuple[str, ...]
) -> dict:
    def col(label: str) -> int:
        return labels.index(label)

    def dist(a: str, b: str) -> dict:
        both = valid[:, col(a)] & valid[:, col(b)]
        d = np.linalg.norm(points[both, col(a)] - points[both, col(b)], axis=1)
        return {
            "mean_m": float(d.mean()),
            "sd_m": float(d.std()),
            "frames": int(both.sum()),
        }

    pairs = {
        "LShoulderTop_LElbowOut": dist("LShoulderTop", "LElbowOut"),
        "LElbowOut_LWristTop": dist("LElbowOut", "LWristTop"),
        "RShoulderBack_RElbowOut": dist("RShoulderBack", "RElbowOut"),
        "RElbowOut_RWristTop": dist("RElbowOut", "RWristTop"),
        "LShoulderTop_RShoulderTop": dist("LShoulderTop", "RShoulderTop"),
        "LShoulderBack_RShoulderBack": dist("LShoulderBack", "RShoulderBack"),
        "WaistLeft_WaistRight": dist("WaistLeft", "WaistRight"),
        "LKneeOut_LAnkleOut": dist("LKneeOut", "LAnkleOut"),
        "RKneeOut_RAnkleOut": dist("RKneeOut", "RAnkleOut"),
        "LAnkleOut_LToeIn": dist("LAnkleOut", "LToeIn"),
        "RAnkleOut_RToeIn": dist("RAnkleOut", "RToeIn"),
        "HeadTop_BackTop": dist("HeadTop", "BackTop"),
    }
    head = np.where(valid[:, col("HeadTop")], points[:, col("HeadTop"), 2], np.nan)
    return {
        "marker_distances": pairs,
        "head_top_max_height_m": float(np.nanmax(head)),
        "head_top_at_address_m": float(head[0]),
    }


def model_geometry(spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text())
    adapter = NativeMujocoFullBodyModel(spec_path.read_bytes())
    m, d = adapter.model, adapter.data
    adapter.frame_poses(dict.fromkeys(adapter.coordinate_order, 0.0))

    def anchor(joint: str) -> np.ndarray:
        return d.xanchor[m.joint(joint).id].copy()

    def site(frame: str) -> np.ndarray:
        return d.site_xpos[m.site(adapter.metadata["frame_sites"][frame]).id].copy()

    lengths = {
        "hub_to_shoulder_L": float(np.linalg.norm(site("LS") - site("Hub"))),
        "shoulder_to_shoulder": float(np.linalg.norm(site("LS") - site("RS"))),
        "upper_arm_L": float(np.linalg.norm(site("LE") - site("LS"))),
        "upper_arm_R": float(np.linalg.norm(site("RE") - site("RS"))),
        "elbow_to_wrist_L": float(np.linalg.norm(site("LW") - site("LE"))),
        "elbow_to_wrist_R": float(np.linalg.norm(site("RW") - site("RE"))),
        "spine_joint_to_hub": float(np.linalg.norm(site("Hub") - site("Spine"))),
        "hip_centres_to_spine_joint": float(
            np.linalg.norm(
                site("Spine") - (anchor("hip_flexion_r") + anchor("hip_flexion_l")) / 2
            )
        ),
        "hip_to_hip": float(
            np.linalg.norm(anchor("hip_flexion_r") - anchor("hip_flexion_l"))
        ),
        "femur_R": float(
            np.linalg.norm(anchor("knee_angle_r") - anchor("hip_flexion_r"))
        ),
        "femur_L": float(
            np.linalg.norm(anchor("knee_angle_l") - anchor("hip_flexion_l"))
        ),
        "tibia_R": float(
            np.linalg.norm(anchor("ankle_angle_r") - anchor("knee_angle_r"))
        ),
        "tibia_L": float(
            np.linalg.norm(anchor("ankle_angle_l") - anchor("knee_angle_l"))
        ),
        "wrist_to_clubhead": float(np.linalg.norm(site("Clubhead") - site("LW"))),
    }
    masses = {}
    for body in spec["bodies"]:
        total = sum(s["mass_kg"] for s in body["solids"])
        if total > 0:
            masses[body["name"].rsplit("/", 1)[-1]] = {
                "mass_kg": float(total),
                "solids": {
                    s["name"].rsplit("/", 1)[-1]: {
                        "mass_kg": s["mass_kg"],
                        "inertia_diag_kg_m2": np.diag(
                            np.array(s["inertia_com_kg_m2"])
                        ).tolist(),
                    }
                    for s in body["solids"]
                    if s["mass_kg"] > 0
                },
            }
    upper = {k: v for k, v in masses.items() if not k.endswith(("_r", "_l"))}
    legs = {k: v for k, v in masses.items() if k.endswith(("_r", "_l"))}
    return {
        "spec_file": spec_path.name,
        "lengths_m": lengths,
        "body_masses": masses,
        "total_mass_kg": float(sum(v["mass_kg"] for v in masses.values())),
        "upper_body_mass_kg": float(sum(v["mass_kg"] for v in upper.values())),
        "legs_mass_kg": float(sum(v["mass_kg"] for v in legs.values())),
        "trunk_head_mass_kg": float(
            sum(
                masses[k]["mass_kg"]
                for k in (
                    "LowerTorso",
                    "UpperTorsoBase",
                    "COMRod",
                    "HubtoLS",
                    "HubtoRS",
                )
                if k in masses
            )
        ),
        "arms_mass_kg": float(
            sum(
                v["mass_kg"]
                for k, v in masses.items()
                if "Arm" in k or "Forearm" in k or "Spherical" in k
            )
        ),
        "club_and_hands_mass_kg": float(
            masses.get("Clubface Vector", {}).get("mass_kg", 0.0)
        ),
    }


def posture_review(
    points: np.ndarray, valid: np.ndarray, labels: tuple[str, ...], spec_path: Path
) -> dict:
    def col(label: str) -> int:
        return labels.index(label)

    hip_cal = functional_hip_calibration(
        points,
        valid,
        labels,
        {
            name: json.loads(spec_path.read_text())["marker_attachments"][name][
                "offset_m"
            ]
            for name in MARKER_SEGMENTS["pelvis"]
        },
    )
    adapter = NativeMujocoFullBodyModel(spec_path.read_bytes())
    m, d = adapter.model, adapter.data
    names = adapter.coordinate_order
    ik = np.load(GS / "ik_trajectory.npz")
    q_ref, times = ik["q_ref"], ik["time_s"]
    rows = []
    for f in (0, 90, 180, 270, 360, 450):
        waist = np.array(
            [
                points[f, col(name)]
                for name in MARKER_SEGMENTS["pelvis"]
                if valid[f, col(name)]
            ]
        )
        if waist.shape[0] < 3:
            continue
        back = np.array(
            [points[f, col(name)] for name in ("BackTop", "BackLeft", "BackRight")]
        )
        shoulders = [
            points[f, col(name)]
            for name in ("LShoulderTop", "RShoulderTop")
            if valid[f, col(name)]
        ]
        top = np.mean(shoulders, axis=0) if shoulders else back.mean(axis=0)
        trunk_axis = top - waist.mean(axis=0)
        pelvis_normal = post.plane_normal(waist, UP)
        marker_bend = post.spine_bend(pelvis_normal, trunk_axis, UP, FORWARD, RIGHT)
        back_roll = np.degrees(
            np.arctan2(
                points[f, col("BackLeft"), 2] - points[f, col("BackRight"), 2],
                np.linalg.norm(
                    (points[f, col("BackLeft")] - points[f, col("BackRight")])[:2]
                ),
            )
        )
        adapter.frame_poses(dict(zip(names, q_ref[f].tolist(), strict=True)))
        hips = (
            d.xanchor[m.joint("hip_flexion_r").id]
            + d.xanchor[m.joint("hip_flexion_l").id]
        ) / 2
        spine = d.site_xpos[m.site(adapter.metadata["frame_sites"]["Spine"]).id]
        hub = d.site_xpos[m.site(adapter.metadata["frame_sites"]["Hub"]).id]
        model_bend = post.spine_bend(spine - hips, hub - spine, UP, FORWARD, RIGHT)
        rows.append(
            {
                "time_s": float(times[f]),
                "markers": {
                    "pelvis_normal_tilt": post.segment_tilt(
                        pelvis_normal, UP, FORWARD, RIGHT
                    ).__dict__,
                    "trunk_axis_tilt": post.segment_tilt(
                        trunk_axis, UP, FORWARD, RIGHT
                    ).__dict__,
                    "spine_bend": marker_bend.__dict__,
                    "upper_back_roll_deg_left_high": float(back_roll),
                },
                "model": {
                    "pelvis_block_tilt": post.segment_tilt(
                        spine - hips, UP, FORWARD, RIGHT
                    ).__dict__,
                    "rod_tilt": post.segment_tilt(
                        hub - spine, UP, FORWARD, RIGHT
                    ).__dict__,
                    "spine_bend": model_bend.__dict__,
                    "spine_input_deg": {
                        n: float(np.degrees(q_ref[f, names.index(n)]))
                        for n in ("SpineInputX", "SpineInputY", "TorsoInput")
                    },
                },
            }
        )
    return {
        "hip_line_length_m": float(
            np.linalg.norm(np.array(hip_cal.centre_r) - np.array(hip_cal.centre_l))
        ),
        "functional_hip_radius_m": {"R": hip_cal.radius_r_m, "L": hip_cal.radius_l_m},
        "frames": rows,
    }


def main() -> None:
    labels = tracked_labels()
    capture = load_tour_capture(C3D).subset(labels)
    points = capture_to_native_world(capture.points_m)
    valid = capture.valid
    spec_path = GS / json.loads((GS / "receipt.json").read_text())["spec_file"]
    geometry = marker_geometry(points, valid, labels)
    posture = posture_review(points, valid, labels, spec_path)
    model = model_geometry(spec_path)
    femur_from_hip = {
        side: float(np.sqrt(max(r**2 - KNEE_MARKER_LATERAL_M**2, 0.0)))
        for side, r in posture["functional_hip_radius_m"].items()
    }
    shank = {
        side: float(
            np.sqrt(
                max(
                    geometry["marker_distances"][f"{side}KneeOut_{side}AnkleOut"][
                        "mean_m"
                    ]
                    ** 2
                    - (KNEE_MARKER_LATERAL_M - ANKLE_MARKER_LATERAL_M) ** 2,
                    0.0,
                )
            )
        )
        for side in ("L", "R")
    }
    forearm = np.mean(
        [
            geometry["marker_distances"]["LElbowOut_LWristTop"]["mean_m"],
            geometry["marker_distances"]["RElbowOut_RWristTop"]["mean_m"],
        ]
    )
    stature = anth.stature_from_segment_lengths(
        {
            "thigh": float(np.mean(list(femur_from_hip.values()))),
            "shank": float(np.mean(list(shank.values()))),
            "forearm": float(forearm),
        }
    )
    expectations = {
        f"{mass:.0f}kg": {
            name: params.__dict__
            for name, params in anth.whole_body(stature, mass).items()
        }
        for mass in CANDIDATE_MASSES_KG
    }
    receipt = {
        "capture": geometry,
        "capture_derived": {
            "femur_from_functional_hip_m": femur_from_hip,
            "shank_from_markers_m": shank,
            "forearm_from_markers_m": float(forearm),
            "stature_estimate_m": stature,
            "stature_from_head_top_max_m": geometry["head_top_max_height_m"] + 0.02,
            "note": "stature from de Leva proportions of thigh, shank and forearm; the golfer never stands fully upright in this capture",
        },
        "model": model,
        "de_leva_expectations": expectations,
        "posture": posture,
    }
    (HERE / "receipt.json").write_text(
        json.dumps(receipt, indent=2, default=float) + "\n"
    )


if __name__ == "__main__":
    main()
