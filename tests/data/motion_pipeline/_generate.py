"""Generate deterministic synthetic golden fixtures for motion_pipeline tests.

Run from repo root:

    python3 tests/data/motion_pipeline/_generate.py

This script regenerates every fixture under ``tests/data/motion_pipeline/golden/``.
Output is deterministic (modulo float rounding) so re-running produces the same
files. Each fixture is intentionally tiny (<= 50 KB) — the goal is to exercise
the source-format adapters, not to provide real biomechanical captures.

Part of issue #4571 gap-fill.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Deterministic config -------------------------------------------------------
NUM_FRAMES = 30
FPS = 60.0
DT = 1.0 / FPS
NP_RNG = np.random.default_rng(seed=4571)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def _swing_signal(num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Smooth deterministic 1D oscillation in [-1, 1] over `num_frames` frames."""
    t = np.linspace(0.0, 2.0 * math.pi, num_frames, dtype=np.float64)
    return np.sin(t)


# ---------------------------------------------------------------------------
# BVH
# ---------------------------------------------------------------------------
def write_bvh(path: Path) -> None:
    """5-joint linear chain, 30 frames, gentle swing on root + child rotations."""
    sig = _swing_signal()
    # 6 root channels (XYZ pos + ZXY rot) + 3 channels per of 4 children = 18 total
    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "  OFFSET 0.00 0.00 0.00",
        "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
        "  JOINT Spine",
        "  {",
        "    OFFSET 0.00 10.00 0.00",
        "    CHANNELS 3 Zrotation Xrotation Yrotation",
        "    JOINT Chest",
        "    {",
        "      OFFSET 0.00 15.00 0.00",
        "      CHANNELS 3 Zrotation Xrotation Yrotation",
        "      JOINT Neck",
        "      {",
        "        OFFSET 0.00 20.00 0.00",
        "        CHANNELS 3 Zrotation Xrotation Yrotation",
        "        JOINT Head",
        "        {",
        "          OFFSET 0.00 10.00 0.00",
        "          CHANNELS 3 Zrotation Xrotation Yrotation",
        "          End Site",
        "          {",
        "            OFFSET 0.00 5.00 0.00",
        "          }",
        "        }",
        "      }",
        "    }",
        "  }",
        "}",
        "MOTION",
        f"Frames: {NUM_FRAMES}",
        f"Frame Time: {DT:.7f}",
    ]
    for i in range(NUM_FRAMES):
        s = float(sig[i])
        # 18 channels: 6 root + 4*3 children
        vals = [
            0.0,
            0.0,
            0.0,  # root pos
            10.0 * s,
            0.0,
            0.0,  # root rot
            5.0 * s,
            0.0,
            0.0,  # spine
            5.0 * s,
            0.0,
            0.0,  # chest
            3.0 * s,
            0.0,
            0.0,  # neck
            2.0 * s,
            0.0,
            0.0,  # head
        ]
        lines.append(" ".join(f"{v:.6f}" for v in vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# TRC (OpenSim marker file)
# ---------------------------------------------------------------------------
def write_trc(path: Path) -> None:
    """6-marker, 30-frame TRC at 60 Hz."""
    markers = ["RASI", "LASI", "RPSI", "LPSI", "RKNE", "LKNE"]
    sig = _swing_signal()
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{FPS}\t{FPS}\t{NUM_FRAMES}\t{len(markers)}\tm\t{FPS}\t1\t{NUM_FRAMES}",
        "Frame#\tTime\t" + "\t\t\t".join(markers) + "\t\t\t",
        "\t\t"
        + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(markers))),
        "",
    ]
    rows = []
    for i in range(NUM_FRAMES):
        t = i * DT
        s = float(sig[i])
        coords = []
        for m_idx, _ in enumerate(markers):
            x = 0.1 * (m_idx + 1) + 0.01 * s
            y = 1.0 + 0.05 * s
            z = 0.05 * m_idx
            coords.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        rows.append(f"{i + 1}\t{t:.6f}\t" + "\t".join(coords))
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# MOT (OpenSim joint-angle file)
# ---------------------------------------------------------------------------
def _write_storage(path: Path, header_name: str, col_names: list[str]) -> None:
    sig = _swing_signal()
    n_cols = len(col_names) - 1  # minus 'time'
    header = [
        f"{header_name}",
        "version=1",
        f"nRows={NUM_FRAMES}",
        f"nColumns={len(col_names)}",
        "inDegrees=yes",
        "endheader",
        "\t".join(col_names),
    ]
    rows = []
    for i in range(NUM_FRAMES):
        t = i * DT
        s = float(sig[i])
        vals = [f"{t:.6f}"] + [f"{(j + 1) * 5.0 * s:.6f}" for j in range(n_cols)]
        rows.append("\t".join(vals))
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def write_mot(path: Path) -> None:
    cols = [
        "time",
        "hip_flexion_r",
        "knee_angle_r",
        "ankle_angle_r",
        "hip_flexion_l",
        "knee_angle_l",
    ]
    _write_storage(path, "Coordinates", cols)


def write_sto(path: Path) -> None:
    cols = [
        "time",
        "/jointset/hip_r/hip_flexion_r/value",
        "/jointset/knee_r/knee_angle_r/value",
        "/jointset/ankle_r/ankle_angle_r/value",
        "/jointset/hip_l/hip_flexion_l/value",
        "/jointset/knee_l/knee_angle_l/value",
    ]
    _write_storage(path, "States", cols)


# ---------------------------------------------------------------------------
# OpenPose BODY_25
# ---------------------------------------------------------------------------
BODY_25_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]


def write_openpose(path: Path) -> None:
    """OpenPose BODY_25 array form, multi-frame.

    Stored as a list of single-frame `{version, people:[{pose_keypoints_2d}]}`
    objects, which is the de-facto multi-frame form used by adapters that
    process sequential JSON dumps.
    """
    sig = _swing_signal()
    frames = []
    for i in range(NUM_FRAMES):
        s = float(sig[i])
        flat = []
        for k in range(25):
            x = 100.0 + 5.0 * k + 2.0 * s
            y = 200.0 + 3.0 * k + 1.5 * s
            c = 0.9
            flat.extend([round(x, 4), round(y, 4), c])
        frames.append(
            {
                "version": 1.3,
                "frame_index": i,
                "people": [{"pose_keypoints_2d": flat}],
            }
        )
    path.write_text(json.dumps(frames, separators=(",", ":")), encoding="utf-8")


# ---------------------------------------------------------------------------
# AlphaPose COCO-17
# ---------------------------------------------------------------------------
COCO_17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def write_alphapose(path: Path) -> None:
    """AlphaPose multi-frame COCO-17 in the typical list-of-detections form."""
    sig = _swing_signal()
    out = []
    for i in range(NUM_FRAMES):
        s = float(sig[i])
        flat = []
        for k in range(17):
            x = 50.0 + 4.0 * k + 1.5 * s
            y = 80.0 + 2.5 * k + s
            c = 0.85
            flat.extend([round(x, 4), round(y, 4), c])
        out.append(
            {
                "image_id": f"frame_{i:04d}.jpg",
                "category_id": 1,
                "keypoints": flat,
                "score": 0.95,
                "idx": [0],
            }
        )
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")


# ---------------------------------------------------------------------------
# HRNet COCO-17 single-person
# ---------------------------------------------------------------------------
def write_hrnet(path: Path) -> None:
    sig = _swing_signal()
    frames = []
    for i in range(NUM_FRAMES):
        s = float(sig[i])
        kp = []
        for k in range(17):
            x = 60.0 + 3.5 * k + s
            y = 90.0 + 2.0 * k + 0.5 * s
            c = 0.88
            kp.append([round(x, 4), round(y, 4), c])
        frames.append({"frame": i, "keypoints": kp})
    payload = {
        "model": "hrnet_w32",
        "schema": "COCO_17",
        "fps": FPS,
        "frames": frames,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV (generic frame, time, x_<j>, y_<j>, z_<j>)
# ---------------------------------------------------------------------------
def write_csv(path: Path) -> None:
    joints = ["hip", "knee", "ankle", "shoulder", "elbow"]
    sig = _swing_signal()
    cols = ["frame", "time"]
    for j in joints:
        cols.extend([f"x_{j}", f"y_{j}", f"z_{j}"])
    rows = [",".join(cols)]
    for i in range(NUM_FRAMES):
        t = i * DT
        s = float(sig[i])
        vals = [str(i), f"{t:.6f}"]
        for k, _ in enumerate(joints):
            x = 0.1 * (k + 1) + 0.02 * s
            y = 0.5 + 0.01 * s
            z = 0.05 * k
            vals.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        rows.append(",".join(vals))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# MediaPipe Pose 33-landmark dump
# ---------------------------------------------------------------------------
def write_mediapipe(path: Path) -> None:
    sig = _swing_signal()
    frames = []
    for i in range(NUM_FRAMES):
        s = float(sig[i])
        landmarks = []
        for k in range(33):
            x = 0.5 + 0.01 * k + 0.005 * s  # normalized [0,1]
            y = 0.4 + 0.005 * k + 0.003 * s
            z = -0.1 + 0.002 * k
            landmarks.append(
                [
                    round(x, 4),
                    round(y, 4),
                    round(z, 4),
                    0.95,
                    0.99,
                ]
            )
        frames.append(
            {
                "frame_index": i,
                "timestamp": round(i * DT, 6),
                "landmarks": landmarks,
            }
        )
    payload = {
        "schema": "MediaPipe_33",
        "image_width": 1920,
        "image_height": 1080,
        "fps": FPS,
        "frames": frames,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


# ---------------------------------------------------------------------------
# C3D (binary, optional ezc3d dependency)
# ---------------------------------------------------------------------------
def write_c3d(path: Path) -> bool:
    """Write a minimal C3D using ezc3d if available. Returns True if written."""
    try:
        import ezc3d  # type: ignore  # noqa: F401
    except ImportError:
        return False

    c3d = ezc3d.c3d()
    n_markers = 6
    c3d["parameters"]["POINT"]["RATE"]["value"] = [FPS]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = [
        f"M{i + 1}" for i in range(n_markers)
    ]
    sig = _swing_signal()
    pts = np.zeros((4, n_markers, NUM_FRAMES), dtype=np.float64)
    for i in range(NUM_FRAMES):
        s = float(sig[i])
        for m in range(n_markers):
            pts[0, m, i] = 100.0 * (m + 1) + 10.0 * s  # mm
            pts[1, m, i] = 1000.0 + 50.0 * s
            pts[2, m, i] = 50.0 * m
            pts[3, m, i] = 1.0  # residual
    c3d["data"]["points"] = pts
    c3d.write(str(path))
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
FIXTURES: list[tuple[str, callable]] = [
    ("sample.bvh", write_bvh),
    ("sample.trc", write_trc),
    ("sample.mot", write_mot),
    ("sample.sto", write_sto),
    ("openpose_keypoints.json", write_openpose),
    ("alphapose.json", write_alphapose),
    ("hrnet.json", write_hrnet),
    ("sample.csv", write_csv),
    ("mediapipe.json", write_mediapipe),
]


def main() -> None:
    for name, fn in FIXTURES:
        out = GOLDEN_DIR / name
        fn(out)
        size = out.stat().st_size
        print(f"  wrote {name:30s} {size:>6d} bytes")
        if size > 50_000:
            raise SystemExit(f"Fixture {name} is {size} bytes, exceeds 50 KB cap")

    c3d_path = GOLDEN_DIR / "sample.c3d"
    wrote = write_c3d(c3d_path)
    if wrote:
        size = c3d_path.stat().st_size
        print(f"  wrote {'sample.c3d':30s} {size:>6d} bytes")
        if size > 50_000:
            raise SystemExit(f"sample.c3d is {size} bytes, exceeds 50 KB cap")
    else:
        print("  SKIP sample.c3d (ezc3d not installed)")


if __name__ == "__main__":
    main()
