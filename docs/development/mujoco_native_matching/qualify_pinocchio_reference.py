"""Compare MuJoCo rigid dynamics to a separately saved native Pinocchio run."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from src.engines.physics_engines.mujoco.python.native_model import NativeMujocoModel
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "candidate", "reference", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_bytes()), names, hashlib.sha256(raw).hexdigest()
    )
    meta = json.loads((args.reference / "report.json").read_bytes())
    if (
        meta["candidate_sha256"] != candidate.sha256
        or meta["model_sha256"] != candidate.document["model_sha256"]
        or meta["coordinate_order"] != names
        or meta["engine"] != "pinocchio"
    ):
        raise ValueError("Pinocchio reference identities differ")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = np.load(args.reference / "reference.npz")
    clock, state = reference["time_s"], reference["state"]
    engine = NativeMujocoModel(raw)
    frames, accelerations = [], []
    n = len(names)
    for time in meta["pulse_times_s"]:
        selected = np.flatnonzero(clock == time)
        if len(selected) != 1:
            raise ValueError("Pulse time missing or ambiguous")
        row = state[selected[0]]
        q, v = (
            dict(zip(names, row[:n], strict=True)),
            dict(zip(names, row[n:], strict=True)),
        )
        poses = engine.frame_poses(q)
        if set(poses) != {frame["name"] for frame in spec["frames"]}:
            raise ValueError("Native frame inventory missing")
        frames.append([poses[name] for name in sorted(poses)])
        cases = []
        for pulse in np.vstack((np.zeros(n), np.eye(n))):
            result = engine.accelerations(q, v, dict(zip(names, pulse, strict=True)))
            cases.append([result[name] for name in names])
        accelerations.append(cases)
    replay = replay_candidate(
        raw,
        candidate,
        clock,
        model_factory=lambda _: NativeMujocoModel(raw),
        rtol=1e-11,
        atol=1e-13,
        max_step=0.00025,
    )
    delta = np.asarray(accelerations) - reference["accelerations"]
    report = {
        "execution_mode": "MuJoCo M/bias/J/Jdot explicit rigid KKT; stock mj_step unqualified",
        "integrator": "existing shared native_replay and DOP853; rtol=1e-11 atol=1e-13 max_step=0.00025",
        "model_sha256": engine.model_sha256,
        "candidate_sha256": candidate.sha256,
        "mujoco_version": mujoco.__version__,
        "horizon_s": float(clock[-1]),
        "coordinate_order": names,
        "frame_names": sorted(poses),
        "pulse_times_s": meta["pulse_times_s"],
        "pulse_count": int(np.prod(delta.shape[:2])),
        "pulse_max_abs": float(np.max(np.abs(delta))),
        "pulse_max_scaled": float(
            np.max(np.abs(delta) / (1 + np.abs(reference["accelerations"])))
        ),
        "frame_max_abs": float(
            np.max(np.abs(np.asarray(frames) - reference["frames"]))
        ),
        "q_max_abs": float(
            np.max(np.abs(replay.integration.state[:, :n] - state[:, :n]))
        ),
        "qd_max_abs": float(
            np.max(np.abs(replay.integration.state[:, n:] - state[:, n:]))
        ),
        "marker_max_abs_m": float(
            np.max(np.abs(replay.markers_m - reference["markers"]))
        ),
        "closure_pose_max_abs": replay.closure_pose_max_abs,
        "closure_rate_max_abs": replay.closure_velocity_max_abs,
        "elapsed_s": replay.integration.elapsed_s,
    }
    report["passed"] = bool(
        report["pulse_max_scaled"] < 1e-7
        and report["frame_max_abs"] < 1e-10
        and report["q_max_abs"] < 1e-5
        and report["qd_max_abs"] < 0.01
        and report["marker_max_abs_m"] < 1e-6
    )
    paths = [
        args.model,
        args.candidate,
        args.reference / "report.json",
        args.reference / "reference.npz",
        Path(__file__),
        Path("src/engines/physics_engines/mujoco/python/native_model.py"),
        Path("src/engines/physics_engines/mujoco/python/native_mjcf.py"),
        Path("src/engines/physics_engines/pinocchio/python/native_replay.py"),
    ]
    report["input_and_source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        args.output / "trajectory.npz",
        time_s=clock,
        state=replay.integration.state,
        markers=replay.markers_m,
        accelerations=accelerations,
        frames=frames,
    )
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
