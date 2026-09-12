"""Bounded reproducible native sextic refinement; preserve every evaluation."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_candidate,
)
from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    PrefixFitOptions,
    fit_prefixes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "target", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    base = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()),
        spec["coordinate_order"],
        hashlib.sha256(raw).hexdigest(),
    )
    doc = base.document
    payload = json.loads(args.target.read_text())
    if payload["source_sha256"] != doc["capture_sha256"]:
        raise ValueError("Capture mismatch")
    indices = [payload["labels"].index(label) for label in doc["marker_labels"]]
    clock = np.asarray(payload["time_s"])
    mask = clock <= doc["duration_s"]
    clock = clock[mask]
    points = np.asarray(payload["points_world_m"])[mask][:, indices].copy()
    valid = np.asarray(payload["valid"], dtype=bool)[mask][:, indices]
    points[~valid] = np.nan
    target = MarkerTarget(clock, points, np.ones(len(indices)))
    n = len(doc["coordinate_names"])
    basis_duration = doc["duration_s"]
    config = {
        "parent_candidate_sha256": base.sha256,
        "basis_duration_s": basis_duration,
        "powers": [6],
        "amplitude_scale": 10.0,
        "dimensionless_center": 1.0,
        "bounds": [0.8, 1.2],
        "max_nfev": args.max_nfev,
        "finite_difference_step": 1e-3,
        "qualification": "bounded exploratory refinement, not native acceptance",
        "input_sha256": {
            k: hashlib.sha256(getattr(args, k).read_bytes()).hexdigest()
            for k in ("model", "candidate", "target")
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    observed = np.isfinite(points).all(axis=2)
    early = observed & (clock[:, None] <= 0.6)
    club = np.array(
        [s.lower().startswith(("marker_2", "marker_3")) for s in doc["marker_labels"]]
    )
    evaluations = 0
    best_score = float("inf")
    last = {}

    def candidate_for(x: np.ndarray) -> NativeReplayCandidate:
        increment = np.zeros((n, 7))
        increment[:, 6] = 10 * (x - 1)
        return increment_native_candidate(
            base, increment, basis_duration_s=basis_duration
        )

    def forward(x: np.ndarray, time: np.ndarray) -> np.ndarray:
        nonlocal evaluations, best_score, last
        candidate = candidate_for(x)
        result = replay_candidate(
            raw, candidate, time, rtol=1e-11, atol=1e-13, max_step=0.00025
        )
        prediction = result.markers_m
        error = np.sum((prediction - points) ** 2, axis=2)
        whole = float(np.sqrt(np.mean(error[observed])))
        early_rms = float(np.sqrt(np.mean(error[early])))
        terminal = float(np.sqrt(np.mean(error[-1, observed[-1]])))
        club_rms = float(np.sqrt(np.mean(error[-1, club & observed[-1]])))
        score = float(np.sum(error[observed]) + 100 * np.sum(error[-1, observed[-1]]))
        evaluations += 1
        last = {
            "evaluation": evaluations,
            "candidate_sha256": candidate.sha256,
            "whole_rms_m": whole,
            "early_rms_m": early_rms,
            "terminal_rms_m": terminal,
            "club_cluster_rms_m": club_rms,
            "score": score,
            "integration_s": result.integration.elapsed_s,
        }
        with (args.output_dir / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(last) + "\n")
        if early_rms <= 0.012 and score < best_score:
            best_score = score
            record = {
                "qualification": "best early-retaining exploratory evaluation; not accepted",
                "metrics": last,
                "candidate": candidate.document,
            }
            temporary = args.output_dir / "best.tmp"
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(args.output_dir / "best.json")
        return prediction

    opts = PrefixFitOptions(
        max_nfev=args.max_nfev,
        finite_difference_step=1e-3,
        terminal_weight=10.0,
        acceptance_terminal_rmse_m=0.035,
        pelvis_indices=(
            doc["marker_labels"].index("WaistLeft"),
            doc["marker_labels"].index("WaistRight"),
        ),
    )
    try:
        fit = fit_prefixes(
            target,
            forward,
            initial=np.ones(n),
            lower=np.full(n, 0.8),
            upper=np.full(n, 1.2),
            prefix_end_s=[doc["duration_s"]],
            acceptance_rmse_m=0.025,
            options=opts,
        )
        returned = candidate_for(fit.parameters)
        forward(fit.parameters, clock)
        report = {
            "qualification": "exploratory returned optimizer candidate; R2025b acceptance pending",
            "accepted_numerically": bool(
                fit.accepted
                and last["early_rms_m"] <= 0.012
                and last["club_cluster_rms_m"] <= 0.06
            ),
            "candidate": returned.document,
            "metrics": last,
            "optimizer_converged": fit.stages[-1].optimizer_converged,
            "optimizer_message": fit.stages[-1].message,
        }
        (args.output_dir / "returned.json").write_text(json.dumps(report, indent=2))
    except (ValueError, RuntimeError, FloatingPointError) as error:
        (args.output_dir / "failure.json").write_text(
            json.dumps({"error": str(error), "completed_evaluations": evaluations})
        )
        raise


if __name__ == "__main__":
    main()
