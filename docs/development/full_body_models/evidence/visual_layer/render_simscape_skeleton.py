"""Draw the Simscape-qualified skeleton at the native candidate's start pose.

The full-body spec's upper body is the port of the Simscape ``GolfSwing3D_Kinetic``
model (native frame parity receipt ``native_frame_parity_r2025b.json``). This
script places the 27 upper-body coordinates at the returned81 candidate's q0
(the Simscape qualified start), reads every joint anchor from MuJoCo forward
kinematics, and draws the joint-to-joint stick skeleton with labels from three
views, with the frame-0 capture markers overlaid. It also writes a posture
table (segment lengths, spine bend, rod tilt, shoulder drop below the hub) so
the picture can be discussed in numbers. Kinematic only; no dynamics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter  # noqa: E402
from src.shared.python.motion_matching.ground_support import (  # noqa: E402
    capture_to_native_world,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    load_tour_capture,
)

NATIVE = Path(
    "C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native/docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81"
)
HERE = Path(__file__).resolve().parent
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"
C3D = ROOT / "data/C3D_TA_Driver.c3d"

# Simscape joint names -> the spec body whose own joint sits at that point.
JOINTS = {
    "hip/spine/torso": "UpperTorsoBase",  # hip, torso and spine joints coincide
    "hub (scapulae)": "HubtoLS",
    "LS": "LUpperArm",
    "RS": "RUpperArm",
    "LE": "Spherical Solid",
    "RE": "Spherical Solid1",
    "LF": "LLowerForearm",
    "RF": "RLowerForearm",
    "LW": "Clubface Vector",
    "RW": "RHandStandoff",
    "hip_r": "femur_r",
    "hip_l": "femur_l",
    "knee_r": "tibia_r",
    "knee_l": "tibia_l",
    "ankle_r": "talus_r",
    "ankle_l": "talus_l",
}
LINKS = [
    ("hip/spine/torso", "hub (scapulae)"),
    ("hub (scapulae)", "LS"),
    ("hub (scapulae)", "RS"),
    ("LS", "LE"),
    ("RS", "RE"),
    ("LE", "LF"),
    ("RE", "RF"),
    ("LF", "LW"),
    ("RF", "RW"),
    ("LW", "clubhead"),
    ("hip/spine/torso", "hip_r"),
    ("hip/spine/torso", "hip_l"),
    ("hip_r", "knee_r"),
    ("hip_l", "knee_l"),
    ("knee_r", "ankle_r"),
    ("knee_l", "ankle_l"),
]
FRAME_SITES: dict[str, str] = {}
UPPER_BODY_BODIES = {
    "LowerTorso", "UpperTorsoBase", "COMRod", "HubtoLS", "HubtoRS",
    "LUpperArm", "RUpperArm", "Spherical Solid", "Spherical Solid1",
    "LLowerForearm", "RLowerForearm", "Clubface Vector", "RHandStandoff",
}  # fmt: skip


def joint_points(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    anchors: dict[str, np.ndarray] = {}
    for j in range(model.njnt):
        body = model.body(model.jnt_bodyid[j]).name.rsplit("/", 1)[-1]
        anchors.setdefault(body, data.xanchor[j].copy())
    points = {label: anchors[body] for label, body in JOINTS.items()}
    # The spec's "Clubhead" frame is exported as a site; use it directly.
    site = model.site(FRAME_SITES["Clubhead"]).id
    points["clubhead"] = data.site_xpos[site].copy()
    return points


def posture(points: dict[str, np.ndarray], q: dict[str, float]) -> dict:
    def length(a: str, b: str) -> float:
        return float(np.linalg.norm(points[a] - points[b]))

    rod = points["hub (scapulae)"] - points["hip/spine/torso"]
    tilt = float(np.degrees(np.arccos(rod[2] / np.linalg.norm(rod))))
    return {
        "source": "returned81 candidate q0 = Simscape qualified start; legs at zero",
        "simscape_geometry_in_inches": {"UpperArmLength": 14.5, "LowerArmLength": 12.0},
        "segment_lengths_m": {
            "hub_to_shoulder": length("hub (scapulae)", "LS"),
            "shoulder_to_shoulder": length("LS", "RS"),
            "upper_arm_L": length("LS", "LE"),
            "upper_arm_R": length("RS", "RE"),
            "elbow_to_wrist_L": length("LE", "LF") + length("LF", "LW"),
            "elbow_to_wrist_R": length("RE", "RF") + length("RF", "RW"),
            "spine_rod_hip_to_hub": length("hip/spine/torso", "hub (scapulae)"),
            "wrist_to_clubhead": length("LW", "clubhead"),
        },
        "start_pose_deg": {
            "spine_bend_pelvis_to_rod_X": np.degrees(q["SpineInputX"]),
            "spine_bend_pelvis_to_rod_Y": np.degrees(q["SpineInputY"]),
            "torso_rotation_Z": np.degrees(q["TorsoInput"]),
            "rod_tilt_from_vertical": tilt,
            "LScap_X": np.degrees(q["LScapInputX"]),
            "LScap_Y": np.degrees(q["LScapInputY"]),
            "RScap_X": np.degrees(q["RScapInputX"]),
            "RScap_Y": np.degrees(q["RScapInputY"]),
        },
        "shoulder_drop_below_hub_m": {
            "L": float(points["hub (scapulae)"][2] - points["LS"][2]),
            "R": float(points["hub (scapulae)"][2] - points["RS"][2]),
        },
    }


def draw(
    points: dict[str, np.ndarray], markers: np.ndarray, path: Path, source: str
) -> None:
    fig = plt.figure(figsize=(15, 5.5))
    views = {
        "front (down the line)": (0, -90),
        "side (face on)": (0, 0),
        "isometric": (22, -45),
    }
    for k, (title, (elev, azim)) in enumerate(views.items(), start=1):
        ax = fig.add_subplot(1, 3, k, projection="3d")
        for a, b in LINKS:
            p, r = points[a], points[b]
            colour = (
                "0.35"
                if b == "clubhead"
                else (
                    "tab:blue"
                    if a in ("LS", "RS", "LE", "RE", "LF", "RF")
                    else "tab:red"
                )
            )
            ax.plot(
                [p[0], r[0]],
                [p[1], r[1]],
                [p[2], r[2]],
                "-",
                color=colour,
                lw=3 if b != "clubhead" else 2,
            )
        for label, p in points.items():
            ax.scatter(*p, s=25, color="k", depthshade=False)
            if k == 3:
                ax.text(p[0], p[1], p[2], f" {label}", fontsize=7)
        valid = np.isfinite(markers).all(axis=1)
        ax.scatter(
            *markers[valid].T,
            s=10,
            color="tab:green",
            alpha=0.8,
            depthshade=False,
            label="capture markers, frame 0",
        )
        centre = np.nanmean(markers, axis=0)
        for axis, c in zip("xyz", centre, strict=True):
            getattr(ax, f"set_{axis}lim")(c - 1.0, c + 1.0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        if k == 1:
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Skeleton through the exported joint frames: {source}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--trajectory", type=Path, help="driver ik_trajectory.npz")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    suffix = f"_{args.suffix}" if args.suffix else ""
    xml, meta = exporter.export_full_body_mjcf(args.spec.read_bytes(), visual=True)
    FRAME_SITES.update(meta["frame_sites"])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if args.trajectory is None:
        candidate = json.loads((NATIVE / "returned-candidate.json").read_text())
        q = dict(zip(candidate["coordinate_names"], candidate["q0"], strict=True))
        markers = np.load(NATIVE / "returned-replay.npz")["target_m"][0]
        source = "returned81 candidate q0 = Simscape qualified start; legs at zero"
    else:
        order = json.loads(args.spec.read_text())["coordinate_order"]
        q = dict(zip(order, np.load(args.trajectory)["q"][0], strict=True))
        capture = load_tour_capture(C3D)
        markers = capture_to_native_world(capture.points_m)[0]
        source = f"frame 0 of {args.trajectory.name} on {args.spec.name}"
    for name, value in q.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    points = joint_points(model, data)
    draw(points, markers, HERE / f"simscape_skeleton_address{suffix}.png", source)
    report = posture(points, q)
    report["source"] = source
    report["joint_points_m"] = {
        k: [round(float(v), 4) for v in p] for k, p in points.items()
    }
    (HERE / f"address_posture{suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
