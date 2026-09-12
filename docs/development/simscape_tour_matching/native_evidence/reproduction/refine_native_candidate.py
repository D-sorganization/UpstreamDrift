"""Bounded reproducible native sextic refinement; preserve every evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.engines.physics_engines.pinocchio.python.native_sensitivity import (
    replay_marker_sensitivities,
)
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
    recover_native_bernstein,
)
from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    PrefixFitOptions,
    fit_prefixes,
)

_SHAPING_FIRST_CONTROL = {"sixth": 6, "bernstein456": 4, "bernstein23456": 2}


def control_matrix(
    parameters: np.ndarray,
    coordinate_count: int,
    first_control: int,
    root_forces_only: bool,
    amplitude_scale: float,
) -> np.ndarray:
    """Expand dimensionless search parameters without changing frozen efforts."""
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError("Amplitude scale must be finite and positive")
    if coordinate_count < 3 or first_control not in _SHAPING_FIRST_CONTROL.values():
        raise ValueError("Unsupported native control inventory")
    active = 3 if root_forces_only else coordinate_count
    values = np.asarray(parameters, dtype=float)
    if values.shape != (active * (7 - first_control),) or not np.isfinite(values).all():
        raise ValueError("Invalid refinement parameters")
    increment = np.zeros((coordinate_count, 7))
    increment[:active, first_control:] = amplitude_scale * (
        values.reshape(active, -1) - 1
    )
    return increment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "target", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=3)
    parser.add_argument("--restart-candidate", type=Path)
    parser.add_argument(
        "--shaping", choices=tuple(_SHAPING_FIRST_CONTROL), default="sixth"
    )
    parser.add_argument("--root-forces-only", action="store_true")
    parser.add_argument("--amplitude-scale", type=float, default=10.0)
    parser.add_argument("--finite-difference-step", type=float, default=1e-3)
    parser.add_argument("--analytic-jacobian", action="store_true")
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
    first_control = _SHAPING_FIRST_CONTROL[args.shaping]
    active = 3 if args.root_forces_only else n
    parameter_count = active * (7 - first_control)
    initial = np.ones(parameter_count)
    control_matrix(
        initial, n, first_control, args.root_forces_only, args.amplitude_scale
    )
    restart_hash = None
    if args.restart_candidate is not None:
        restart = NativeReplayCandidate.from_document(
            json.loads(args.restart_candidate.read_text()),
            spec["coordinate_order"],
            hashlib.sha256(raw).hexdigest(),
        )
        delta = recover_native_bernstein(base, restart, basis_duration_s=basis_duration)
        if np.any(delta[:, :first_control] != 0) or np.any(delta[active:] != 0):
            raise ValueError("Restart contains corrections outside the selected basis")
        initial += delta[:active, first_control:].ravel() / args.amplitude_scale
        if np.any(initial < 0.8) or np.any(initial > 1.2):
            raise ValueError("Restart exceeds original correction bounds")
        restart_hash = restart.sha256
    config = {
        "parent_candidate_sha256": base.sha256,
        "basis_duration_s": basis_duration,
        "powers": list(range(first_control, 7)),
        "representation": "degree-six Bernstein correction",
        "free_control_indices": list(range(first_control, 7)),
        "parameter_order": "coordinate-major, ascending control index",
        "amplitude_scale": args.amplitude_scale,
        "active_coordinates": doc["coordinate_names"][:active],
        "root_forces_only": args.root_forces_only,
        "correction_bound_per_control": 0.2 * args.amplitude_scale,
        "correction_units": "N"
        if args.root_forces_only
        else "N for first three, Nm for remaining",
        "dimensionless_center": 1.0,
        "bounds": [0.8, 1.2],
        "max_nfev": args.max_nfev,
        "restart_candidate_sha256": restart_hash,
        "initial_parameters": initial.tolist(),
        "finite_difference_step": args.finite_difference_step,
        "analytic_jacobian": args.analytic_jacobian,
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
        increment = control_matrix(
            x, n, first_control, args.root_forces_only, args.amplitude_scale
        )
        return increment_native_bernstein(
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
            "parameters": x.tolist(),
            "near_bound_count": int(np.sum((x < 0.8001) | (x > 1.1999))),
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

    jacobian_key = None
    cached_jacobian = None

    def marker_jacobian(x: np.ndarray, time: np.ndarray) -> np.ndarray:
        nonlocal jacobian_key, cached_jacobian
        candidate = candidate_for(x)
        clock_hash = hashlib.sha256(
            np.asarray(time, dtype=np.float64).tobytes()
        ).hexdigest()
        key = (candidate.sha256, clock_hash)
        if key != jacobian_key:
            started = perf_counter()
            result = replay_marker_sensitivities(
                raw, candidate, time, first_control=first_control
            )
            cached_jacobian = (
                result.marker_jacobian[:, :, :, :parameter_count] * args.amplitude_scale
            )
            cached_jacobian.setflags(write=False)
            record = {
                "candidate_sha256": candidate.sha256,
                "clock_sha256": clock_hash,
                "parameter_count": parameter_count,
                "total_elapsed_s": perf_counter() - started,
                "sensitivity_elapsed_s": result.sensitivity_elapsed_s,
                "sensitivity_evaluations": result.sensitivity_evaluations,
                "primal_marker_max_abs_difference_m": result.primal_marker_max_abs_difference_m,
                "additional_primal_replays": 1,
                "sensitivity_integrations": 1,
            }
            with (args.output_dir / "jacobians.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            jacobian_key = key
        if cached_jacobian is None:
            raise ValueError("Native Jacobian cache was not populated")
        return cached_jacobian

    opts = PrefixFitOptions(
        max_nfev=args.max_nfev,
        finite_difference_step=args.finite_difference_step,
        marker_jacobian=marker_jacobian if args.analytic_jacobian else None,
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
            initial=initial,
            lower=np.full(parameter_count, 0.8),
            upper=np.full(parameter_count, 1.2),
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
            "parameters": fit.parameters.tolist(),
        }
        (args.output_dir / "returned.json").write_text(json.dumps(report, indent=2))
        (args.output_dir / "returned-candidate.json").write_text(
            json.dumps(returned.document, indent=2) + "\n"
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        (args.output_dir / "failure.json").write_text(
            json.dumps({"error": str(error), "completed_evaluations": evaluations})
        )
        raise


if __name__ == "__main__":
    main()
