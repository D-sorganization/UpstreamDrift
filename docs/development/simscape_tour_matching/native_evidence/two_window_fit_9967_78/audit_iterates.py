"""Bounded post-run audit of run78's accepted SLSQP line-search iterates.

The bounded fit returned its starting point because no evaluated iterate met
the projected continuity equality within1e-7 before the evaluation budget was
exhausted. This audit replays each accepted iterate once (primal only): the
segmented once-only objective, the full scaled physical defect, the projected
violation, and an uninterrupted original-state replay of the same controls.
It measures the direction of travel; it accepts nothing.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_node_chart import (
    NativeNodeChart,
)
from src.engines.physics_engines.pinocchio.python.native_replay import (
    replay_candidate,
    replay_window,
)
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
    recover_native_bernstein,
)
from src.shared.python.motion_matching.native_effort_penalty import (
    native_effort_penalty,
)
from src.shared.python.motion_matching.native_effort_profile import (
    NativeEffortProfile,
)
from src.shared.python.motion_matching.shooting_schedule import (
    sampled_shooting_windows,
)

CANDIDATE73 = "786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a"
FIRST_CONTROL, DURATION = 4, 0.85
PRIMAL = {"rtol": 1e-11, "atol": 1e-13, "max_step": 0.0000625}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "candidate",
        "parent",
        "returned73",
        "target",
        "samples",
        "chart76",
        "evaluations",
        "output",
        "runtime",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--select", type=int, nargs="+", default=[1, 4, 6, 8, 10])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = perf_counter()
    receipt: dict = {"status": "running", "iterates": []}

    def save() -> None:
        receipt["elapsed_s"] = perf_counter() - started
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    try:
        raw = args.model.read_bytes()
        spec = json.loads(raw)
        names = spec["coordinate_order"]
        n = len(names)
        model_hash = hashlib.sha256(raw).hexdigest()

        def load(path: Path) -> NativeReplayCandidate:
            return NativeReplayCandidate.from_document(
                json.loads(path.read_text()), names, model_hash
            )

        base, parent = load(args.candidate), load(args.parent)
        assert base.sha256 == CANDIDATE73
        doc = base.document
        returned73 = json.loads(args.returned73.read_text())
        delta73 = recover_native_bernstein(parent, base, basis_duration_s=DURATION)[
            :, FIRST_CONTROL:
        ].ravel()
        payload = json.loads(args.target.read_text())
        assert payload["source_sha256"] == doc["capture_sha256"]
        idx = [payload["labels"].index(label) for label in doc["marker_labels"]]
        full_clock = np.asarray(payload["time_s"], dtype=float)
        mask = full_clock <= DURATION
        clock = full_clock[mask]
        points = np.asarray(payload["points_world_m"])[mask][:, idx].copy()
        valid = np.asarray(payload["valid"], dtype=bool)[mask][:, idx]
        points[~valid] = np.nan
        samples = np.load(args.samples)
        np.testing.assert_array_equal(samples["time_s"], clock)
        windows = sampled_shooting_windows(clock, [0.6, DURATION])
        reference = samples["native_state"][216]
        initial = np.asarray(doc["q0"] + doc["qd0"])
        chart76 = np.load(args.chart76)
        scales = chart76["state_scales"]
        chart = NativeNodeChart(
            names,
            NativePinocchioModel(spec),
            state_scales=scales,
            residual_scales=chart76["residual_scales"],
            radius=0.5,
            tolerance=1e-8,
            jacobian_step=1e-6,
        )
        basis = chart.basis(reference)
        assert np.max(abs(basis - chart76["basis"])) <= 1e-12
        root = next(j for j in spec["joints"] if j["parent"] == "world")
        rotation = np.asarray(root["parent_to_base"])[:3, :3].T
        penalty = native_effort_penalty(
            NativeEffortProfile(names, parent.document["coefficients"], rotation),
            duration_s=DURATION,
            first_control=FIRST_CONTROL,
            effort_scales=np.r_[np.full(3, 100.0), np.full(n - 3, 20.0)],
            weight=0.01,
        )
        club = np.array(
            [
                s.lower().startswith(("marker_2", "marker_3"))
                for s in doc["marker_labels"]
            ]
        )
        early = valid & (clock[:, None] <= 0.6)

        def metrics(pred: np.ndarray) -> dict:
            error = np.sum((pred - points) ** 2, axis=2)
            return {
                "whole_rms_m": float(np.sqrt(np.mean(error[valid]))),
                "early_rms_m": float(np.sqrt(np.mean(error[early]))),
                "terminal_rms_m": float(np.sqrt(np.mean(error[-1, valid[-1]]))),
                "club_cluster_rms_m": float(
                    np.sqrt(np.mean(error[-1, club & valid[-1]]))
                ),
                "score": float(
                    np.sum(error[valid]) + 100 * np.sum(error[-1, valid[-1]])
                ),
            }

        def candidate_for(theta: np.ndarray) -> NativeReplayCandidate:
            if not np.any(theta):
                return base
            controls = np.zeros((n, 7))
            controls[:, FIRST_CONTROL:] = theta.reshape(n, -1)
            return increment_native_bernstein(base, controls, basis_duration_s=DURATION)

        records = {
            json.loads(line)["evaluation"]: json.loads(line)
            for line in args.evaluations.read_text().splitlines()
            if line.strip()
        }
        receipt.update(
            candidate_sha256=base.sha256,
            evaluations_sha256=sha(args.evaluations),
            run73_metrics={
                k: returned73["metrics"][k]
                for k in ("whole_rms_m", "early_rms_m", "terminal_rms_m", "score")
            },
            primal=PRIMAL,
            input_hashes={
                k: sha(getattr(args, k))
                for k in (
                    "model",
                    "candidate",
                    "parent",
                    "returned73",
                    "target",
                    "samples",
                    "chart76",
                    "evaluations",
                )
            },
            driver_sha256=sha(Path(__file__)),
            source_hashes={
                str(p.relative_to(args.runtime)): sha(p)
                for p in args.runtime.rglob("*.py")
            },
        )
        save()
        for index in args.select:
            record = records[index]
            theta = np.asarray(record["theta"], dtype=float)
            z = np.asarray(record["node_chart"]["0.6"], dtype=float)
            node = chart.retract(reference, basis, z)
            recorded = np.asarray(record["physical_nodes"]["0.6"], dtype=float)
            cand = candidate_for(theta)
            first = replay_window(raw, cand, windows[0], initial, **PRIMAL)
            second = replay_window(raw, cand, windows[1], node.state, **PRIMAL)
            segmented = np.concatenate((first.markers_m, second.markers_m[1:]))
            defect = (first.integration.state[-1] - node.state) / scales
            effort = penalty.residual(delta73 + theta)
            full = replay_candidate(raw, cand, clock, **PRIMAL)
            seg = metrics(segmented)
            uninterrupted = metrics(full.markers_m)
            entry = {
                "evaluation": index,
                "candidate_sha256": cand.sha256,
                "assembled_cost_recorded": record["assembled_cost"],
                "node_reproduction_max_abs": float(np.max(abs(node.state - recorded))),
                "theta_max_abs": float(np.max(abs(theta))),
                "node_chart_norm": float(np.linalg.norm(z)),
                "node_physical_displacement_max_abs": float(
                    np.max(abs(node.state - reference))
                ),
                "effort_cost": float(effort @ effort),
                "segmented": seg,
                "segmented_objective_with_effort": seg["score"]
                + float(effort @ effort),
                "scaled_defect_norm": float(np.linalg.norm(defect)),
                "scaled_defect_max_abs": float(np.max(abs(defect))),
                "projected_violation_max_abs": float(np.max(abs(basis.T @ defect))),
                "uninterrupted": uninterrupted,
                "uninterrupted_objective_with_effort": uninterrupted["score"]
                + float(effort @ effort),
                "integration_evaluations": [
                    first.integration.evaluations,
                    second.integration.evaluations,
                    full.integration.evaluations,
                ],
            }
            receipt["iterates"].append(entry)
            np.savez_compressed(
                args.output / f"iterate-{index:02d}.npz",
                theta=theta,
                node_chart=z,
                node=node.state,
                state0=first.integration.state,
                state1=second.integration.state,
                markers0=first.markers_m,
                markers1=second.markers_m,
                uninterrupted_state=full.integration.state,
                uninterrupted_markers=full.markers_m,
            )
            save()
        receipt["status"] = "terminal"
        save()
    except (ValueError, RuntimeError, FloatingPointError, AssertionError) as error:
        receipt.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save()
        raise


if __name__ == "__main__":
    main()
