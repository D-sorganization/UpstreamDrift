"""Source-bound R2025b pulse, moving-state and continuous native audit."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from src.engines.physics_engines.mujoco.python.native_model import NativeMujocoModel
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.marker_projection import project_markers
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("model", "candidate", "pulses", "reference", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    model = NativeMujocoModel(raw)
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_bytes()),
        spec["coordinate_order"],
        model.model_sha256,
    )
    inputs = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.model, args.candidate, args.reference)
    }
    source = {
        str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).resolve(),
            Path("src/engines/physics_engines/mujoco/python/native_model.py").resolve(),
            Path("src/engines/physics_engines/mujoco/python/native_mjcf.py").resolve(),
            Path(
                "src/engines/physics_engines/pinocchio/python/native_replay.py"
            ).resolve(),
        )
    }

    def compare(row: dict) -> dict:
        names = row["coordinate_names"]
        state = [
            dict(zip(names, row[key], strict=True))
            for key in ("q", "qd", "primitive_efforts")
        ]
        actual = model.accelerations(*state)
        expected = np.asarray(row["qdd"])
        delta = np.asarray([actual[name] for name in names]) - expected
        return {
            "maximum_absolute_acceleration_error": float(np.max(np.abs(delta))),
            "maximum_scaled_acceleration_error": float(
                np.max(np.abs(delta) / np.maximum(1, np.abs(expected)))
            ),
        }

    pulses = []
    for path in sorted(args.pulses.glob("case-*.json")):
        row = json.loads(path.read_bytes())
        if row["release"] != "2025b":
            raise ValueError("Pulse reference must use R2025b")
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        pulses.append({"file": path.name, **compare(row)})
    if len(pulses) != 28:
        raise ValueError("Require all 27 unit pulses plus zero")
    reference = json.loads(args.reference.read_bytes())
    if reference["release"] != "2025b":
        raise ValueError("Moving reference must use R2025b")
    if set(reference["frame_names"]) != {frame["name"] for frame in spec["frames"]}:
        raise ValueError("Reference must cover every native marker frame")
    if (
        reference["coordinate_names"] != candidate.document["coordinate_names"]
        or not np.array_equal(
            reference["native_coefficients"], candidate.document["coefficients"]
        )
        or not np.array_equal(reference["q"][0], candidate.document["q0"])
        or not np.array_equal(reference["qd"][0], candidate.document["qd0"])
    ):
        raise ValueError(
            "Continuous parity requires identical inputs and initial state"
        )
    moving = []
    for i, time in enumerate(reference["time_s"]):
        row = {
            key: reference[key][i] for key in ("q", "qd", "qdd", "primitive_efforts")
        }
        row["coordinate_names"] = reference["coordinate_names"]
        result = compare(row)
        poses = model.frame_poses(
            dict(zip(row["coordinate_names"], row["q"], strict=True))
        )
        result["maximum_frame_matrix_error"] = float(
            max(
                np.max(np.abs(poses[name] - pose))
                for name, pose in zip(
                    reference["frame_names"], reference["poses"][i], strict=True
                )
            )
        )
        moving.append({"time_s": time, **result})
    time = np.linspace(0, candidate.document["duration_s"], 801)
    replay = replay_candidate(
        raw,
        candidate,
        time,
        model_factory=lambda _: NativeMujocoModel(raw),
        rtol=1e-11,
        atol=1e-13,
        max_step=0.00025,
    )
    trajectory = reference["trajectory"]
    # Interpolate recorded R2025b positions/rates; never reset the forward state.
    spline = CubicHermiteSpline(trajectory["time_s"], trajectory["q"], trajectory["qd"])
    q_reference, v_reference = spline(time), spline(time, 1)
    body = candidate.document["marker_bodies"]
    offsets = candidate.document["marker_offsets_m"]
    markers = np.asarray(
        [
            project_markers(
                model.frame_poses(dict(zip(spec["coordinate_order"], q, strict=True))),
                body,
                offsets,
            )
            for q in q_reference
        ]
    )
    marker_delta = replay.markers_m - markers
    n = len(spec["coordinate_order"])
    continuous = {
        "horizon_s": float(time[-1]),
        "sample_count": len(time),
        "q_max_abs": float(
            np.max(np.abs(replay.integration.state[:, :n] - q_reference))
        ),
        "qd_max_abs": float(
            np.max(np.abs(replay.integration.state[:, n:] - v_reference))
        ),
        "marker_max_abs_m": float(np.max(np.abs(marker_delta))),
        "marker_rms_m": float(np.sqrt(np.mean(marker_delta**2))),
        "elapsed_s": replay.integration.elapsed_s,
        "closure_pose_max_abs": replay.closure_pose_max_abs,
        "closure_velocity_max_abs": replay.closure_velocity_max_abs,
    }
    passed = (
        max(row["maximum_scaled_acceleration_error"] for row in pulses + moving) < 1e-7
        and max(row["maximum_frame_matrix_error"] for row in moving) < 1e-10
        and continuous["marker_max_abs_m"] < 5e-6
        and continuous["q_max_abs"] < 5e-5
        and continuous["qd_max_abs"] < 0.05
    )
    report = {
        "execution_mode": "MuJoCo M/bias/site Jacobian/Jdot with explicit rigid KKT closure; stock mj_step unqualified",
        "integrator": "shared continuous_forward.integrate_forward DOP853; rtol=1e-11 atol=1e-13 max_step=0.00025; no resets or feedback",
        "marker_reference": "MuJoCo FK at interpolated R2025b states; independent native FK checked at all 16 frames at six states",
        "qualification": "native rigid adapter baseline through 0.8 seconds only; stock mj_step and swing fitting unqualified",
        "passed": bool(passed),
        "mujoco_version": mujoco.__version__,
        "python": platform.python_version(),
        "inputs": inputs,
        "source_sha256": source,
        "model_sha256": model.model_sha256,
        "candidate_sha256": replay.candidate_sha256,
        "mass_kg": float(sum(model.model.body_mass)),
        "coordinate_count": model.model.nv,
        "pulses": pulses,
        "moving": moving,
        "continuous": continuous,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "native.xml").write_text(model.xml)
    (args.output / "metadata.json").write_text(
        json.dumps(model.metadata, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output / "continuous.npz",
        time_s=time,
        state=replay.integration.state,
        reference_q=q_reference,
        reference_qd=v_reference,
        markers_m=replay.markers_m,
        reference_markers_m=markers,
    )
    sys.stdout.write(json.dumps({"passed": passed, "continuous": continuous}) + "\n")


if __name__ == "__main__":
    main()
