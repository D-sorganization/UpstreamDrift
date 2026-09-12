"""Verify world-force authority against native COM replay, without fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "baseline", "candidate", "poses", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    model_hash = hashlib.sha256(raw).hexdigest()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    engine = NativePinocchioModel(spec)
    mass = float(pin.computeTotalMass(engine.model))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Expected finite positive native mass")

    def read_candidate(path: Path) -> NativeReplayCandidate:
        return NativeReplayCandidate.from_document(
            json.loads(path.read_text()), names, model_hash
        )

    baseline = read_candidate(args.baseline)
    candidate = read_candidate(args.candidate)
    duration = baseline.document["duration_s"]
    if candidate.document["duration_s"] != duration:
        raise ValueError("Candidate clocks differ")
    poses = json.loads(args.poses.read_text())
    if poses["input_sha256"]["model"] != model_hash:
        raise ValueError("Static pose model differs")
    times = np.array([0.0] + [row["time_s"] for row in poses["poses"]])
    if times[-1] != duration:
        raise ValueError("Pose clock does not cover the full candidate")

    def com(q: np.ndarray) -> np.ndarray:
        native = engine.configuration(dict(zip(names, map(float, q), strict=True)))
        return np.array(pin.centerOfMass(engine.model, engine.data, native), copy=True)

    def replay(value: NativeReplayCandidate) -> tuple[np.ndarray, dict]:
        result = replay_candidate(
            raw, value, times, rtol=1e-11, atol=1e-13, max_step=0.00025
        )
        return np.array(
            [com(state[: len(names)]) for state in result.integration.state]
        ), {
            "candidate_sha256": value.sha256,
            "closure_pose_max_abs": result.closure_pose_max_abs,
            "closure_velocity_max_abs": result.closure_velocity_max_abs,
        }

    base_com, base_receipt = replay(baseline)
    fit_com, fit_receipt = replay(candidate)
    controls = np.zeros((len(names), 7))
    controls[:3, 4:] = 2.0
    perturbed = increment_native_bernstein(
        baseline, controls, basis_duration_s=duration
    )
    perturb_com, perturb_receipt = replay(perturbed)

    def expected_delta(value: NativeReplayCandidate) -> np.ndarray:
        delta = (
            np.asarray(value.document["coefficients"])[:3]
            - np.asarray(baseline.document["coefficients"])[:3]
        )
        # Highest-power-first polynomial; zero initial position/velocity increment.
        return np.array(
            [np.polyval(np.polyint(np.polyint(row)), times) / mass for row in delta]
        ).T

    force_error = float(
        np.max(np.abs(perturb_com - base_com - expected_delta(perturbed)))
    )
    fit_error = float(np.max(np.abs(fit_com - base_com - expected_delta(candidate))))
    terminal_bound = 2.0 * duration**2 * sum(7 - k for k in (4, 5, 6)) / (56.0 * mass)
    # Independent closed-form Bernstein integral versus converted power polynomial.
    np.testing.assert_allclose(
        expected_delta(perturbed)[-1], terminal_bound, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        perturb_com - base_com, expected_delta(perturbed), rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        fit_com - base_com, expected_delta(candidate), rtol=0, atol=1e-7
    )
    static_com = np.array(
        [com(np.asarray(row["coordinates"])) for row in poses["poses"]]
    )
    report = {
        "qualification": "native force-authority diagnosis; static COM is not measured COM or a necessary fit target",
        "pinocchio_version": pin.__version__,
        "input_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ("model", "baseline", "candidate", "poses")
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mass_kg": mass,
        "time_s": times.tolist(),
        "baseline": base_receipt,
        "candidate": fit_receipt,
        "perturbation": perturb_receipt,
        "perturbation_root_force_controls_N": controls[:3].tolist(),
        "baseline_com_m": base_com.tolist(),
        "candidate_com_m": fit_com.tolist(),
        "perturbed_com_delta_m": (perturb_com - base_com).tolist(),
        "analytic_perturbed_com_delta_m": expected_delta(perturbed).tolist(),
        "perturbation_max_com_balance_error_m": force_error,
        "candidate_max_com_balance_error_m": fit_error,
        "terminal_com_correction_bound_per_axis_m": terminal_bound,
        "static_pose_candidate_sha256": poses["candidate_sha256"],
        "static_pose_com_m": static_com.tolist(),
        "static_minus_baseline_com_m": (static_com - base_com[1:]).tolist(),
        "static_minus_candidate_com_m": (static_com - fit_com[1:]).tolist(),
        "assumptions": [
            "same model, initial state, gravity and no external contact",
            "root translation efforts are world forces",
            "remaining efforts are couples or internal joint torques",
            "static poses are local diagnostic solutions, not required or globally optimal",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
