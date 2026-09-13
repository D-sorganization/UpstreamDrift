"""Same-input scalar/manifold prefix replay and step-size comparison.

This is native representation qualification, not C3D matching or R2025b acceptance.
Both simulations start once from the candidate initial state; the native absolute-
time polynomial and physical specification remain identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_manifold_model import (
    NativeManifoldPinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_window
from src.shared.python.motion_matching.manifold_forward import (
    integrate_manifold_forward,
    integrate_manifold_adaptive,
    integrate_manifold_dop853,
)
from src.shared.python.motion_matching.marker_projection import project_markers
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=0.2)
    parser.add_argument(
        "--max-steps", type=float, nargs="+", default=[1 / 720, 1 / 1440, 1 / 2880]
    )
    parser.add_argument(
        "--method", choices=("fixed", "adaptive", "dop853"), default="fixed"
    )
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--max-evaluations", type=int, default=100000)
    parser.add_argument("--scalar-rtol", type=float, default=1e-10)
    parser.add_argument("--scalar-atol", type=float, default=1e-12)
    parser.add_argument("--scalar-max-step", type=float, default=0.00025)
    args = parser.parse_args()
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_bytes()), names, hashlib.sha256(raw).hexdigest()
    )
    data = candidate.document
    if (
        args.output.exists()
        or not np.isfinite(args.horizon)
        or not 0 < args.horizon <= data["duration_s"]
    ):
        raise ValueError(
            "Require a new output directory and horizon within candidate coverage"
        )
    if any(not np.isfinite(h) or h <= 0 for h in args.max_steps):
        raise ValueError("Step sizes must be finite and positive")
    roots = [j for j in spec["joints"] if j["parent"] == "world"]
    if (
        len(roots) != 1
        or [p["coordinate"] for p in roots[0]["primitives"][:3]] != names[:3]
    ):
        raise ValueError("Unexpected native force coordinate inventory")
    profile = NativeEffortProfile(
        names, data["coefficients"], np.asarray(roots[0]["parent_to_base"])[:3, :3].T
    )
    q0 = dict(zip(names, data["q0"], strict=True))
    v0 = dict(zip(names, data["qd0"], strict=True))
    clock = np.linspace(0, args.horizon, max(2, int(round(args.horizon * 360)) + 1))
    args.output.mkdir()
    scalar = replay_window(
        raw,
        candidate,
        clock,
        np.r_[data["q0"], data["qd0"]],
        rtol=args.scalar_rtol,
        atol=args.scalar_atol,
        max_step=args.scalar_max_step,
    )
    np.savez_compressed(
        args.output / "scalar.npz",
        time=clock,
        state=scalar.integration.state,
        markers=scalar.markers_m,
    )
    report = {
        "scope": "Same-input native representation prefix; not C3D or MATLAB acceptance",
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_sha256": candidate.sha256,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "horizon_s": args.horizon,
        "method": args.method,
        "rtol": args.rtol,
        "atol": args.atol,
        "adaptive_max_evaluations": args.max_evaluations,
        "scalar_rtol": args.scalar_rtol,
        "scalar_atol": args.scalar_atol,
        "scalar_max_step": args.scalar_max_step,
        "scalar_elapsed_s": scalar.integration.elapsed_s,
        "scalar_closure_pose_max_abs": scalar.closure_pose_max_abs,
        "scalar_closure_velocity_max_abs": scalar.closure_velocity_max_abs,
        "runs": [],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    previous = None
    for level, step in enumerate(args.max_steps):
        model = NativeManifoldPinocchioModel(spec)
        mq, mv, _ = model.native_state(q0, v0, profile.evaluate(0))

        def acceleration(
            t: float,
            q: np.ndarray,
            v: np.ndarray,
            engine: NativeManifoldPinocchioModel = model,
        ) -> np.ndarray:
            return engine.acceleration_from_native_efforts(
                q, v, profile.evaluate(t), q0
            )

        solver_options = {
            "integrate": model.integrate,
            "difference_rate": model.difference_rate,
            "max_step": step,
        }
        if args.method == "adaptive":
            solution = integrate_manifold_adaptive(
                mq,
                mv,
                clock,
                acceleration,
                **solver_options,
                difference=model.difference,
                rtol=args.rtol,
                atol=args.atol,
                max_evaluations=args.max_evaluations,
            )
        elif args.method == "dop853":
            solution = integrate_manifold_dop853(
                mq,
                mv,
                clock,
                acceleration,
                **solver_options,
                rtol=args.rtol,
                atol=args.atol,
                max_evaluations=args.max_evaluations,
            )
        else:
            solution = integrate_manifold_forward(
                mq, mv, clock, acceleration, **solver_options
            )
        states, markers, closures = [], [], []
        reference = q0
        for t, q, v in zip(
            clock, solution.configuration, solution.velocity, strict=True
        ):
            native_q, native_v = model.native_coordinates(q, v, reference)
            states.append([native_q[n] for n in names] + [native_v[n] for n in names])
            reference = native_q
            acceleration(float(t), q, v)
            pose_error, rate_error = model.closure_errors()
            closures.append(
                [float(np.max(abs(pose_error))), float(np.max(abs(rate_error)))]
            )
            markers.append(
                project_markers(
                    model.frame_poses(q),
                    data["marker_bodies"],
                    data["marker_offsets_m"],
                )
            )
        states, markers = np.asarray(states), np.asarray(markers)
        delta = states - scalar.integration.state
        marker_error = float(np.max(np.linalg.norm(markers - scalar.markers_m, axis=2)))
        pose, rate = map(float, np.max(closures, axis=0))
        item = {
            "max_step": step,
            "elapsed_s": solution.elapsed_s,
            "evaluations": solution.evaluations,
            "steps": solution.steps,
            "native_q_max_abs_difference": float(np.max(abs(delta[:, : len(names)]))),
            "native_v_max_abs_difference": float(np.max(abs(delta[:, len(names) :]))),
            "marker_max_distance_m": marker_error,
            "closure_pose_max_abs": pose,
            "closure_velocity_max_abs": rate,
        }
        if previous is not None:
            item["marker_max_distance_from_previous_step_m"] = float(
                np.max(np.linalg.norm(markers - previous, axis=2))
            )
        item["prefix_parity_gates_pass"] = bool(
            marker_error <= 1e-6
            and item["native_q_max_abs_difference"] <= 1e-6
            and item["native_v_max_abs_difference"] <= 1e-4
            and max(pose, rate) <= 1e-7
        )
        report["runs"].append(item)
        np.savez_compressed(
            args.output / f"manifold-{level}.npz",
            time=clock,
            configuration=solution.configuration,
            velocity=solution.velocity,
            native_state=states,
            markers=markers,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        previous = markers


if __name__ == "__main__":
    main()
