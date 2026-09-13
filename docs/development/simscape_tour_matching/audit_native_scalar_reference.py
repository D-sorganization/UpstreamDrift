"""Bounded scalar reference refinement against archived same-input rollouts.

Diagnostic experiment using existing qualified replay providers. No optimization,
state resets, feedback, altered effort mapping, or acceptance gate changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_window
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "baseline", "manifold", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new immutable output directory")
    raw = args.model.read_bytes()
    names = json.loads(raw)["coordinate_order"]
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_bytes()), names, hashlib.sha256(raw).hexdigest()
    )
    baseline = np.load(args.baseline, allow_pickle=False)
    manifold = np.load(args.manifold, allow_pickle=False)
    clock = baseline["time"]
    if not np.array_equal(clock, manifold["time"]):
        raise ValueError("Comparison clocks differ")
    n = len(names)
    expected = (len(clock), 2 * n)
    if (
        baseline["state"].shape != expected
        or manifold["native_state"].shape != expected
    ):
        raise ValueError("Comparison state inventories differ")
    initial = np.r_[candidate.document["q0"], candidate.document["qd0"]]
    if not np.array_equal(initial, baseline["state"][0]):
        raise ValueError("Baseline initial state differs from candidate")
    args.output.mkdir()
    report = {
        "scope": "Scalar reference refinement; not C3D or R2025b acceptance",
        "inputs_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ("model", "candidate", "baseline", "manifold")
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_acceleration_calls_per_level": 100000,
        "runs": [],
    }
    previous = baseline["state"]
    for level, (rtol, atol, step) in enumerate(((1e-12, 1e-14, 0.000125),)):
        calls = 0

        class BudgetedModel(NativePinocchioModel):
            def accelerations(
                self,
                coordinates: Mapping[str, float],
                rates: Mapping[str, float],
                primitive_efforts: Mapping[str, float],
            ) -> dict[str, float]:
                nonlocal calls
                calls += 1
                if calls > report["max_acceleration_calls_per_level"]:
                    raise RuntimeError("Scalar reference acceleration budget exhausted")
                return super().accelerations(coordinates, rates, primitive_efforts)

        item = {"rtol": rtol, "atol": atol, "max_step": step}
        try:
            result = replay_window(
                raw,
                candidate,
                clock,
                initial,
                model_factory=BudgetedModel,
                rtol=rtol,
                atol=atol,
                max_step=step,
            )
        except (RuntimeError, ValueError) as exc:
            item.update(status="failed", error=str(exc), acceleration_calls=calls)
            report["runs"].append(item)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            raise
        states = result.integration.state
        item.update(
            status="complete",
            elapsed_s=result.integration.elapsed_s,
            evaluations=result.integration.evaluations,
            acceleration_calls=calls,
            previous_q_max_abs=float(np.max(abs(states[:, :n] - previous[:, :n]))),
            previous_v_max_abs=float(np.max(abs(states[:, n:] - previous[:, n:]))),
            manifold_q_max_abs=float(
                np.max(abs(states[:, :n] - manifold["native_state"][:, :n]))
            ),
            manifold_v_max_abs=float(
                np.max(abs(states[:, n:] - manifold["native_state"][:, n:]))
            ),
            manifold_marker_max_distance_m=float(
                np.max(np.linalg.norm(result.markers_m - manifold["markers"], axis=2))
            ),
            closure_pose_max_abs=result.closure_pose_max_abs,
            closure_velocity_max_abs=result.closure_velocity_max_abs,
        )
        np.savez_compressed(
            args.output / f"scalar-{level}.npz",
            time=clock,
            state=states,
            markers=result.markers_m,
        )
        report["runs"].append(item)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        previous = states


if __name__ == "__main__":
    main()
