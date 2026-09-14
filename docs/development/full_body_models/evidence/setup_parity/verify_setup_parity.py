"""Same address pose in MuJoCo, Drake and Pinocchio from one document (AN-1).

The full-body document is the single source of the model; every engine builds
from it and takes a pose as ``{coordinate name: value}`` in the document's
``coordinate_order`` (``frame_poses`` on each adapter), so a pose translates
between engines by name, never by index. This script takes the fitted
address (frame 0 of the ground-support IK trajectory), evaluates every spec
frame in MuJoCo locally and in Drake and Pinocchio on ControlTower (their
environments live there, as for FB-3), and writes ``receipt_<run>.json`` with the
largest position and rotation deviation per engine. Gate: positions within 1e-5 m
and rotations within 2e-5 rad of MuJoCo (URDF text precision), plus the masses and
the whole-body centre of mass reported for the record.

    python verify_setup_parity.py            # local MuJoCo + remote engines
    python verify_setup_parity.py --engine drake --local   # on the remote host
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

GROUND_SUPPORT = ROOT / "docs/development/full_body_models/evidence/ground_support"
RUN = "anthro_driver"  # overridden by --run (a ground-support output folder)
SPEC = GROUND_SUPPORT / RUN / "full_body_spec_hipcal_scaled.json"
TRAJECTORY = GROUND_SUPPORT / RUN / "ik_trajectory.npz"
POSE = HERE / "address_pose.json"
REMOTE_BASE = "/home/dieterolson/fb3_pinocchio_test"
REMOTE_PYTHON = {
    "drake": "/home/dieterolson/drake-native-10022/bin/python",
    "pinocchio": "/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python",
}
SYNC = [
    "src/engines/physics_engines/drake/python/full_body_model.py",
    "src/engines/physics_engines/drake/python/full_body_urdf.py",
    "src/engines/physics_engines/pinocchio/python/native_model.py",
    "src/shared/python/motion_matching/full_body_spec.py",
    "src/shared/python/motion_matching/contact_law.py",
    "src/shared/python/motion_matching/marker_projection.py",
    "docs/development/full_body_models/evidence/setup_parity/verify_setup_parity.py",
    "docs/development/full_body_models/evidence/setup_parity/address_pose.json",
    "docs/development/full_body_models/evidence/setup_parity/spec.json",
]
POSITION_TOL_M = 1e-5  # URDF text precision bounds Drake at a few micrometres
ROTATION_TOL_RAD = 2e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pose() -> dict[str, float]:
    spec = json.loads(SPEC.read_text())
    q0 = np.load(TRAJECTORY)["q"][0]
    pose = {
        name: float(v) for name, v in zip(spec["coordinate_order"], q0, strict=True)
    }
    POSE.write_text(json.dumps(pose, indent=2) + "\n")
    (HERE / "spec.json").write_text(SPEC.read_text())
    return pose


def frame_poses(engine: str, spec: dict, pose: dict[str, float]) -> dict[str, list]:
    if engine == "mujoco":
        from src.engines.physics_engines.mujoco.python.full_body_model import (
            NativeMujocoFullBodyModel,
        )

        adapter = NativeMujocoFullBodyModel(json.dumps(spec).encode())
        poses = adapter.frame_poses(pose)
    elif engine == "drake":
        from src.engines.physics_engines.drake.python.full_body_model import (
            FullBodyDrakeModel,
        )

        poses = FullBodyDrakeModel(spec).frame_poses(pose)
    elif engine == "pinocchio":
        from src.engines.physics_engines.pinocchio.python.native_model import (
            FullBodyPinocchioModel,
        )

        poses = FullBodyPinocchioModel(spec).frame_poses(pose)
    else:
        raise ValueError(f"Unknown engine {engine}")
    return {name: np.asarray(m, dtype=float).tolist() for name, m in poses.items()}


def run_remote(engine: str) -> dict[str, list]:
    for rel in SYNC:
        local = ROOT / rel
        remote = f"{REMOTE_BASE}/{rel}"
        mk = subprocess.run(
            [
                "ssh",
                "controltower",
                "wsl",
                "-d",
                "ControlTower-Runner",
                "mkdir",
                "-p",
                str(Path(remote).parent.as_posix()),
            ],
            capture_output=True,
        )
        if mk.returncode != 0:
            raise RuntimeError(mk.stderr.decode(errors="ignore"))
        tee = subprocess.run(
            ["ssh", "controltower", "wsl", "-d", "ControlTower-Runner", "tee", remote],
            input=local.read_bytes(),
            capture_output=True,
        )
        if tee.returncode != 0:
            raise RuntimeError(f"sync {rel}: {tee.stderr.decode(errors='ignore')}")
    script = (
        f"cd {REMOTE_BASE}\n{REMOTE_PYTHON[engine]} "
        f"docs/development/full_body_models/evidence/setup_parity/verify_setup_parity.py "
        f"--engine {engine} --local\n"
    ).encode()
    res = subprocess.run(
        ["ssh", "controltower", "wsl", "-d", "ControlTower-Runner", "bash"],
        input=script,
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"{engine} remote failed:\n{res.stderr.decode(errors='ignore')}\n{res.stdout.decode(errors='ignore')}"
        )
    out = res.stdout.decode(errors="ignore")
    return json.loads(out[out.index("{") :])


def deviation(reference: dict[str, list], other: dict[str, list]) -> dict:
    worst_pos, worst_rot, worst_frame = 0.0, 0.0, ""
    for name, ref in reference.items():
        if name not in other:
            raise ValueError(f"Frame {name} missing")
        a, b = np.asarray(ref), np.asarray(other[name])
        pos = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
        rel = a[:3, :3].T @ b[:3, :3]
        rot = float(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0)))
        if max(pos, rot) > max(worst_pos, worst_rot):
            worst_frame = name
        worst_pos, worst_rot = max(worst_pos, pos), max(worst_rot, rot)
    return {
        "max_position_error_m": worst_pos,
        "max_rotation_error_rad": worst_rot,
        "worst_frame": worst_frame,
        "frames": len(reference),
        "passed": worst_pos <= POSITION_TOL_M and worst_rot <= ROTATION_TOL_RAD,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="mujoco")
    parser.add_argument("--run", default=RUN, help="ground-support output folder")
    parser.add_argument(
        "--local", action="store_true", help="evaluate one engine here and print JSON"
    )
    args = parser.parse_args()
    global SPEC, TRAJECTORY
    SPEC = GROUND_SUPPORT / args.run / "full_body_spec_hipcal_scaled.json"
    TRAJECTORY = GROUND_SUPPORT / args.run / "ik_trajectory.npz"
    receipt_path = HERE / f"receipt_{args.run}.json"
    if args.local:
        spec = json.loads((HERE / "spec.json").read_text())
        pose = json.loads(POSE.read_text())
        sys.stdout.write(json.dumps(frame_poses(args.engine, spec, pose)) + "\n")
        return
    pose = write_pose()
    spec = json.loads(SPEC.read_text())
    reference = frame_poses("mujoco", spec, pose)
    results = {"mujoco": {"frames": len(reference), "passed": True, "reference": True}}
    for engine in ("drake", "pinocchio"):
        try:
            results[engine] = deviation(reference, run_remote(engine))
        except (RuntimeError, ValueError) as exc:  # noqa: BLE001 - recorded in the receipt
            results[engine] = {"passed": False, "error": str(exc)[:2000]}
    masses = sum(s["mass_kg"] for b in spec["bodies"] for s in b["solids"])
    receipt = {
        "issue": "#10099",
        "run": args.run,
        "club": spec.get("club"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "spec": str(SPEC.relative_to(ROOT)),
        "spec_sha256": sha256(SPEC),
        "trajectory_sha256": sha256(TRAJECTORY),
        "pose": "frame 0 of the ground-support IK trajectory, by coordinate name",
        "coordinates": len(pose),
        "total_mass_kg": masses,
        "tolerances": {"position_m": POSITION_TOL_M, "rotation_rad": ROTATION_TOL_RAD},
        "engines": results,
        "translation_rule": "a pose is a mapping coordinate name -> value in the document's coordinate_order; every adapter's frame_poses takes that mapping",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    sys.stdout.write(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
