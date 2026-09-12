"""Audit one native force-to-marker variational direction against replay evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.forward_sensitivity import (
    integrate_sensitivities,
)
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
)
from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    n = len(names)
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    doc = candidate.document
    reference = json.loads(args.reference.read_text())
    if (
        reference["candidate_sha256"] != candidate.sha256
        or reference["input_sha256"]["model"] != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("Finite-difference reference identity mismatch")
    clock = np.asarray(reference["time_s"])
    direction = np.zeros((n, 7))
    direction[names.index(reference["coordinate"]), reference["bernstein_control"]] = (
        1.0
    )
    perturbed = increment_native_bernstein(
        candidate, direction, basis_duration_s=doc["duration_s"]
    )
    root = next(joint for joint in spec["joints"] if joint["parent"] == "world")
    rotation = np.asarray(root["parent_to_base"])[:3, :3].T
    profile = NativeEffortProfile(names, doc["coefficients"], rotation)
    direction_profile = NativeEffortProfile(
        names,
        np.asarray(perturbed.document["coefficients"])
        - np.asarray(doc["coefficients"]),
        rotation,
    )
    engine = NativePinocchioModel(spec)

    def mapping(values: np.ndarray) -> dict[str, float]:
        return dict(zip(names, map(float, values), strict=True))

    def linearize(t: float, state: np.ndarray) -> tuple:
        q, v = mapping(state[:n]), mapping(state[n:])
        effort = profile.evaluate(t)
        acceleration = engine.accelerations(q, v, effort)
        local = engine.acceleration_derivatives(q, v, effort)
        a = np.block([[np.zeros((n, n)), np.eye(n)], [local.dq, local.dv]])
        delta = direction_profile.evaluate(t)
        b = np.concatenate(
            (np.zeros(n), local.deffort @ np.array([delta[name] for name in names]))
        )[:, None]
        return np.concatenate((state[n:], [acceleration[name] for name in names])), a, b

    started = perf_counter()
    result = integrate_sensitivities(
        np.asarray(doc["q0"] + doc["qd0"]),
        clock,
        linearize,
        1,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.00025,
    )
    sensitivity_s = perf_counter() - started
    primal = replay_candidate(
        raw, candidate, clock, rtol=1e-11, atol=1e-13, max_step=0.00025
    )
    positions, derivatives, closures = [], [], []
    for t, state, s in zip(
        clock, result.integration.state, result.state_parameter_jacobian, strict=True
    ):
        q, v = mapping(state[:n]), mapping(state[n:])
        marker = engine.marker_derivatives(
            q, doc["marker_bodies"], doc["marker_offsets_m"]
        )
        positions.append(marker.positions_m)
        derivatives.append(np.einsum("mcn,n->mc", marker.dposition_dq, s[:n, 0]))
        engine.accelerations(q, v, profile.evaluate(float(t)))
        closures.append(
            max(float(np.max(np.abs(value))) for value in engine.closure_errors())
        )
    numeric = np.asarray(reference["samples"][-1]["derivative_m_per_N"])
    analytic = np.asarray(derivatives)
    relative = float(np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric))
    marker_difference = float(np.max(np.abs(np.asarray(positions) - primal.markers_m)))
    report = {
        "qualification": "one native force-direction sensitivity audit; not full optimizer Jacobian qualification",
        "candidate_sha256": candidate.sha256,
        "input_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ("model", "candidate", "reference")
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sensitivity_elapsed_s": sensitivity_s,
        "sensitivity_rhs_evaluations": result.integration.evaluations,
        "reference_replay_integration_s": primal.integration.elapsed_s,
        "primal_marker_max_abs_difference_m": marker_difference,
        "sampled_closure_max_abs": max(closures),
        "relative_marker_derivative_difference": relative,
        "max_abs_marker_derivative_difference": float(
            np.max(np.abs(analytic - numeric))
        ),
        "marker_derivative_m_per_N": analytic.tolist(),
        "accepted_audit": bool(
            relative < 1e-3 and marker_difference < 1e-7 and max(closures) < 1e-7
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted_audit"]:
        raise ValueError("Native trajectory sensitivity audit failed; preserve report")


if __name__ == "__main__":
    main()
