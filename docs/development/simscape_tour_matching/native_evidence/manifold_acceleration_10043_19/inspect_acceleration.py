"""Pointwise same-state native/manifold acceleration audit; no integration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_manifold_model import (
    NativeManifoldPinocchioModel,
)
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrices(engine, q) -> dict:
    jacobian = np.asarray(
        pin.getConstraintsJacobian(
            engine.model, engine.data, engine.constraints, engine.constraint_data
        )
    ).copy()
    upper = np.triu(pin.crba(engine.model, engine.data, q))
    mass = upper + np.triu(upper, 1).T
    eigen = np.linalg.eigvalsh(mass)
    schur = jacobian @ np.linalg.solve(mass, jacobian.T)
    kkt = np.block(
        [[mass, jacobian.T], [jacobian, np.zeros((len(jacobian), len(jacobian)))]]
    )
    return {
        "mass_eigenvalue_min": float(eigen[0]),
        "mass_eigenvalue_max": float(eigen[-1]),
        "mass_condition_unscaled": float(np.linalg.cond(mass)),
        "constraint_jacobian_shape": list(jacobian.shape),
        "constraint_jacobian_singular_values": np.linalg.svd(
            jacobian, compute_uv=False
        ).tolist(),
        "constraint_schur_condition_unscaled": float(np.linalg.cond(schur)),
        "kkt_condition_unscaled": float(np.linalg.cond(kkt)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "states", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.model.read_bytes())
    names = spec["coordinate_order"]
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_bytes()), names, digest(args.model)
    )
    data = candidate.document
    root = next(j for j in spec["joints"] if j["parent"] == "world")
    profile = NativeEffortProfile(
        names, data["coefficients"], np.asarray(root["parent_to_base"])[:3, :3].T
    )
    reference = dict(zip(names, data["q0"], strict=True))
    scalar = NativePinocchioModel(spec)
    alternate = NativeManifoldPinocchioModel(spec)
    with np.load(args.states, allow_pickle=False) as saved:
        clock, states = saved["time"], saved["state"]
    if (
        states.shape != (len(clock), 2 * len(names))
        or not np.isfinite(states).all()
        or not np.isfinite(clock).all()
    ):
        raise ValueError("Invalid saved scalar states")

    def state(index):
        return (
            dict(zip(names, states[index, : len(names)], strict=True)),
            dict(zip(names, states[index, len(names) :], strict=True)),
            profile.evaluate(float(clock[index])),
        )

    def ordered(values):
        return np.array([values[name] for name in names])

    def actual_route(q, v, tau):
        mq, mv, mapped = alternate.native_state(q, v, tau)
        aq, av = alternate.native_coordinates(mq, mv, reference)
        _, _, roundtrip = alternate.native_state(aq, av, tau)
        a = alternate.acceleration_from_native_efforts(mq, mv, tau, reference)
        # Reuse the production state adapter; do not invent a qdd formula.
        native_a = alternate.adapter.restore(
            alternate._native_state(mq, mv, a), q, preserve_middle_branch=True
        )[2]
        return (
            ordered(native_a),
            float(np.max(abs(roundtrip - mapped))),
            float(np.max(abs(ordered(aq) - ordered(q)))),
            float(np.max(abs(ordered(av) - ordered(v)))),
        )

    rows = []
    reference_accelerations = []
    direct_errors = []
    route_errors = []
    for i, t in enumerate(clock):
        q, v, tau = state(i)
        expected = ordered(scalar.accelerations(q, v, tau))
        direct = ordered(alternate.native_accelerations(q, v, tau))
        routed, effort_error, q_error, v_error = actual_route(q, v, tau)
        other = (i + len(clock) // 2) % len(clock)
        oq, ov, otau = state(other)
        scalar.accelerations(oq, ov, otau)
        repeated_scalar = ordered(scalar.accelerations(q, v, tau))
        alternate.native_accelerations(oq, ov, otau)
        repeated_route, *_ = actual_route(q, v, tau)
        direct_error = direct - expected
        route_error = routed - expected
        reference_accelerations.append(expected)
        direct_errors.append(direct_error)
        route_errors.append(route_error)
        peak = int(np.argmax(abs(route_error)))
        rows.append(
            {
                "time_s": float(t),
                "route_native_qdd_max_abs_error": float(np.max(abs(route_error))),
                "route_peak_coordinate": names[peak],
                "route_peak_reference_qdd": float(expected[peak]),
                "route_scaled_qdd_max_error": float(
                    np.max(abs(route_error) / (1 + abs(expected)))
                ),
                "direct_native_qdd_max_abs_error": float(np.max(abs(direct_error))),
                "direct_vs_route_max_abs_error": float(np.max(abs(routed - direct))),
                "scalar_repeat_after_other_state_max_abs_error": float(
                    np.max(abs(repeated_scalar - expected))
                ),
                "route_repeat_after_other_state_max_abs_error": float(
                    np.max(abs(repeated_route - routed))
                ),
                "roundtrip_tangent_effort_max_abs_error": effort_error,
                "roundtrip_native_q_max_abs_error": q_error,
                "roundtrip_native_v_max_abs_error": v_error,
                "other_state_time_s": float(clock[other]),
            }
        )
    peak = int(np.argmax([row["route_native_qdd_max_abs_error"] for row in rows]))
    problem = int(np.argmin(abs(clock - 0.7861111111111111)))
    probes = []
    for i in sorted({0, peak, problem, len(clock) - 1}):
        q, v, tau = state(i)
        scalar.accelerations(q, v, tau)
        scalar_matrices = matrices(scalar, scalar.configuration(q))
        mq, mv, mapped = alternate.native_state(q, v, tau)
        alternate.acceleration(mq, mv, mapped)
        manifold_matrices = matrices(alternate, mq)
        probes.append(
            {
                "time_s": float(clock[i]),
                "scalar": scalar_matrices,
                "manifold": manifold_matrices,
            }
        )
    route_errors = np.asarray(route_errors)
    direct_errors = np.asarray(direct_errors)
    reference_accelerations = np.asarray(reference_accelerations)
    coordinate_peaks = []
    for col, name in enumerate(names):
        row = int(np.argmax(abs(route_errors[:, col])))
        coordinate_peaks.append(
            {
                "coordinate": name,
                "time_s": float(clock[row]),
                "signed_route_error": float(route_errors[row, col]),
                "reference_qdd": float(reference_accelerations[row, col]),
            }
        )
    summary = {
        key: max(rows, key=lambda row: row[key])
        for key in (
            "route_native_qdd_max_abs_error",
            "direct_native_qdd_max_abs_error",
            "direct_vs_route_max_abs_error",
            "scalar_repeat_after_other_state_max_abs_error",
            "route_repeat_after_other_state_max_abs_error",
            "roundtrip_tangent_effort_max_abs_error",
        )
    }
    report = {
        "scope": "Same-state pointwise acceleration/history audit at every saved run59 scalar sample; no integration or acceptance claim.",
        "pinocchio_version": pin.__version__,
        "candidate_sha256": candidate.sha256,
        "input_sha256": {
            name: digest(getattr(args, name))
            for name in ("model", "candidate", "states")
        },
        "script_sha256": digest(Path(__file__)),
        "sample_count": len(clock),
        "summary": summary,
        "at_requested_problem_time": rows[problem],
        "coordinate_peaks": sorted(
            coordinate_peaks,
            key=lambda row: abs(row["signed_route_error"]),
            reverse=True,
        ),
        "conditioning_probes": probes,
        "conditioning_note": "Unscaled SI numerical conditioning; mixed translational/rotational units. These are sampled pointwise values, not global bounds.",
        "time_series": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
