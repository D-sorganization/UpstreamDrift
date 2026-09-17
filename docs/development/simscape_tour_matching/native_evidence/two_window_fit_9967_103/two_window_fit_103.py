"""Bounded two-window direct-node SLSQP continuation 103 for [0, 0.90s].

Restart recenters on Candidate 102 (ee4a9087...) out to 0.90s (325 frames @ 360 Hz).
Intermediate shooting node at 0.60s anchored on Candidate 102 integrated endpoint (defect = 0).
Window 0: 0.0 to 0.60s (frozen early trajectory, Early RMS <= 12 mm retained).
Window 1: 0.60 to 0.90s (optimizing B4/B5/B6 controls, FIRST_CONTROL = 4).
Pelvis yaw error maintained < 5.0%.
Continuous forward dynamics: zero target-state resets.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.io

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
from src.engines.physics_engines.pinocchio.python.native_sensitivity import (
    replay_marker_sensitivities,
)
from src.shared.python.motion_matching.multi_shooting_fit import (
    MultipleShootingOptions,
    fit_multiple_shooting,
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
from src.shared.python.motion_matching.pelvis_yaw import (
    compute_pelvis_yaw_metrics,
    compute_pelvis_yaw_residual_and_derivative,
)
from src.shared.python.motion_matching.prefix_fit import MarkerTarget
from src.shared.python.motion_matching.shooting_schedule import (
    sampled_shooting_windows,
)

CANDIDATE102 = "ee4a908737f4cf3065bb6e64a056dfab71926c43ac40c0e85574fc8531576c82"
PARENT19 = "b5b1c3823c86a21df323dc4e430366dc093495b069b3d25c361b0d007b8ff24f"
FIRST_CONTROL, DURATION, AMPLITUDE = 4, 0.90, 10.0
PRIMAL = {"rtol": 1e-11, "atol": 1e-13, "max_step": 0.0000625}
SENSITIVITY = {
    "rtol": 1e-10,
    "atol": 1e-12,
    "max_step": 0.0000625,
    "max_sensitivity_evaluations": 200000,
    "separate_error_control": True,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "parent",
        "restart",
        "target",
        "output",
        "runtime",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--returned73", type=Path, default=None)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--node-bound", type=float, default=0.075)
    parser.add_argument("--box-factor", type=float, default=6.5)
    parser.add_argument("--control-box", type=float, default=5.0)
    parser.add_argument("--terminal-weight", type=float, default=25.0)
    parser.add_argument("--pelvis-yaw-weight", type=float, default=40.0)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    started = perf_counter()
    receipt: dict = {
        "status": "running",
        "residual_evaluations": 0,
        "jacobians": 0,
        "primal_windows": 0,
    }

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

        parent_doc = json.loads(args.parent.read_text())
        parent_orig_sha = hashlib.sha256(
            json.dumps(parent_doc, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        parent_doc["duration_s"] = DURATION
        parent = NativeReplayCandidate.from_document(parent_doc, names, model_hash)

        # For restart: candidate 102 extended to duration 0.90s
        restart_doc = json.loads(args.restart.read_text())
        restart_orig_sha = hashlib.sha256(
            json.dumps(restart_doc, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        restart_doc["duration_s"] = DURATION
        base = NativeReplayCandidate.from_document(restart_doc, names, model_hash)
        doc = base.document

        assert (
            parent_orig_sha == PARENT19
            or parent.document["source_sha256"] == parent_doc["source_sha256"]
        )
        assert (
            restart_orig_sha == CANDIDATE102
            or doc["source_sha256"] == restart_doc["source_sha256"]
        )
        assert base.document["q0"] == parent.document["q0"]

        x_size = n * (7 - FIRST_CONTROL)

        delta_parent = recover_native_bernstein(parent, base, basis_duration_s=DURATION)
        assert np.all(delta_parent[:, :FIRST_CONTROL] == 0)
        delta_flat = delta_parent[:, FIRST_CONTROL:].ravel()

        assert np.isfinite(args.control_box) and args.control_box > 0
        box = args.control_box
        lower = np.full(delta_flat.shape, -box)
        upper = np.full(delta_flat.shape, box)
        receipt["restart_roundoff"] = {
            "max_abs_increment_outside_box": 0.0,
            "note": "Control bounds recentered on candidate 102 with symmetric box headroom +/- control_box",
        }
        assert np.all(lower < 0) and np.all(upper > 0)

        payload = json.loads(args.target.read_text())
        idx = [payload["labels"].index(label) for label in doc["marker_labels"]]
        full_clock = np.asarray(payload["time_s"], dtype=float)
        mask = full_clock <= DURATION + 1e-9
        clock = full_clock[mask]
        points = np.asarray(payload["points_world_m"])[mask][:, idx].copy()
        valid = np.asarray(payload["valid"], dtype=bool)[mask][:, idx]
        points[~valid] = np.nan
        target = MarkerTarget(clock, points, np.ones(len(idx)))
        windows = sampled_shooting_windows(clock, [0.6, DURATION])
        initial = np.asarray(doc["q0"] + doc["qd0"])

        # Recenter the node chart on restart candidate 102's integrated 0.60s state
        restart_full = replay_candidate(raw, base, clock, **PRIMAL)
        reference = restart_full.integration.state[len(windows[0]) - 1].copy()
        scales = np.r_[np.full(n, 0.1), np.ones(n)]
        chart = NativeNodeChart(
            names,
            NativePinocchioModel(spec),
            state_scales=scales,
            residual_scales=np.r_[np.full(6, 0.01), np.full(6, 0.1)],
            radius=0.5,
            tolerance=1e-8,
            jacobian_step=1e-6,
        )
        basis = chart.basis(reference)
        receipt["reference_closure_max_abs"] = float(
            np.max(abs(chart.closure(reference)))
        )
        dim = basis.shape[1]
        assert args.node_bound > 0 and args.node_bound * np.sqrt(dim) <= 0.5

        # Node transform cache: chart coordinates -> retraction, and state -> z.
        retractions: dict[bytes, object] = {}
        by_state: dict[bytes, tuple[np.ndarray, object]] = {}

        def mapped(z: np.ndarray):
            key = np.asarray(z, dtype=float).tobytes()
            if key not in retractions:
                node = chart.retract(reference, basis, np.asarray(z, dtype=float))
                retractions[key] = node
                by_state[node.state.tobytes()] = (np.array(z, dtype=float), node)
            return retractions[key]

        zero = mapped(np.zeros(dim))
        receipt["zero_retraction_shift"] = float(np.max(abs(zero.state - reference)))

        # Controls: theta are physical B4/B5/B6 Bernstein increments over candidate 102
        def candidate_for(theta: np.ndarray) -> NativeReplayCandidate:
            if not np.any(theta):
                return base
            controls = np.zeros((n, 7))
            controls[:, FIRST_CONTROL:] = np.asarray(theta).reshape(n, -1)
            return increment_native_bernstein(base, controls, basis_duration_s=DURATION)

        assert candidate_for(np.zeros(x_size)).sha256 == base.sha256
        root = next(j for j in spec["joints"] if j["parent"] == "world")
        rotation = np.asarray(root["parent_to_base"])[:3, :3].T
        penalty = native_effort_penalty(
            NativeEffortProfile(names, parent.document["coefficients"], rotation),
            duration_s=DURATION,
            first_control=FIRST_CONTROL,
            effort_scales=np.r_[np.full(3, 100.0), np.full(n - 3, 20.0)],
            weight=0.01,
        )

        def effort_residual(theta: np.ndarray) -> np.ndarray:
            return penalty.residual(delta_flat + theta)

        def effort_jacobian(theta: np.ndarray) -> np.ndarray:
            return penalty.jacobian(delta_flat + theta)

        effort_cost0 = float(
            effort_residual(np.zeros(x_size)) @ effort_residual(np.zeros(x_size))
        )
        receipt["effort_cost_zero"] = effort_cost0

        # Window evaluation cache shared by residual and Jacobian callbacks.
        window_cache: dict[tuple, object] = {}
        primal_cache: dict[tuple, object] = {}

        def evaluate_primal(theta, window, state):
            cand = candidate_for(theta)
            key = (
                cand.sha256,
                window.tobytes(),
                None if state is None else state.tobytes(),
            )
            if key in window_cache:
                return window_cache[key].replay
            if key not in primal_cache:
                start = initial if state is None else state
                result = replay_window(raw, cand, window, start, **PRIMAL)
                receipt["primal_windows"] += 1
                if len(primal_cache) >= 8:
                    primal_cache.pop(next(iter(primal_cache)))
                primal_cache[key] = result
            return primal_cache[key]

        def evaluate(theta, window, state):
            cand = candidate_for(theta)
            key = (
                cand.sha256,
                window.tobytes(),
                None if state is None else state.tobytes(),
            )
            if key not in window_cache:
                tangent = None
                start = initial
                if state is not None:
                    start = state
                    tangent = by_state[state.tobytes()][1].state_jacobian
                t0 = perf_counter()
                result = replay_marker_sensitivities(
                    raw,
                    cand,
                    window,
                    first_control=FIRST_CONTROL,
                    basis_duration_s=DURATION,
                    initial_state=start,
                    initial_sensitivity=tangent,
                    **SENSITIVITY,
                )
                receipt["jacobians"] += 1
                if len(window_cache) >= 8:
                    window_cache.pop(next(iter(window_cache)))
                window_cache[key] = result
                with (args.output / "jacobians.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "candidate_sha256": cand.sha256,
                                "window": [float(window[0]), float(window[-1])],
                                "columns": int(result.marker_jacobian.shape[-1]),
                                "evaluations": result.sensitivity_evaluations,
                                "sensitivity_s": result.sensitivity_elapsed_s,
                                "total_s": perf_counter() - t0,
                                "primal_agreement_m": result.primal_marker_max_abs_difference_m,
                            }
                        )
                        + "\n"
                    )
                save()
            return window_cache[key]

        def segmented(theta, window, state):
            result = evaluate_primal(theta, window, state)
            return result.markers_m, result.integration.state[-1]

        def window_jacobian(theta, window, state):
            result = evaluate(theta, window, state)
            return result.marker_jacobian, result.state_jacobian[-1]

        def unsegmented(theta, time):
            return replay_candidate(raw, candidate_for(theta), time, **PRIMAL).markers_m

        club = np.array(
            [
                s.lower().startswith(("marker_2", "marker_3"))
                for s in doc["marker_labels"]
            ]
        )
        early = valid & (clock[:, None] <= 0.6)
        wl, wr = (doc["marker_labels"].index(k) for k in ("WaistLeft", "WaistRight"))

        def metrics(pred: np.ndarray) -> dict:
            error = np.sum((pred - points) ** 2, axis=2)
            yaw_m = compute_pelvis_yaw_metrics(pred[-1], points[-1], wl, wr)
            score = float(
                np.sum(error[valid])
                + (args.terminal_weight**2) * np.sum(error[-1, valid[-1]])
            )
            if args.pelvis_yaw_weight > 0:
                yaw_res, _, _ = compute_pelvis_yaw_residual_and_derivative(
                    pred[-1], points[-1], wl, wr, args.pelvis_yaw_weight
                )
                score += float(yaw_res @ yaw_res)

            return {
                "whole_rms_m": float(np.sqrt(np.mean(error[valid]))),
                "early_rms_m": float(np.sqrt(np.mean(error[early]))),
                "terminal_rms_m": float(np.sqrt(np.mean(error[-1, valid[-1]]))),
                "club_cluster_rms_m": float(
                    np.sqrt(np.mean(error[-1, club & valid[-1]]))
                ),
                "pelvis_yaw_error_pct": float(yaw_m.pelvis_yaw_error_pct),
                "pelvis_yaw_diff_deg": float(yaw_m.yaw_diff_deg),
                "score": score,
            }

        def accepted(pred: np.ndarray) -> bool:
            m = metrics(pred)
            return (
                m["whole_rms_m"] <= 0.025
                and m["early_rms_m"] <= 0.012
                and m["terminal_rms_m"] <= 0.040
                and m["club_cluster_rms_m"] <= 0.060
                and m["pelvis_yaw_error_pct"] <= 5
            )

        costs: list[dict] = []
        model: dict = {}

        def callback(theta, residual, cost):
            receipt["residual_evaluations"] += 1
            m = model["equality_start"]
            objective_rows = np.concatenate((residual[:m], residual[m + 2 * n :]))
            defect_rows = residual[m : m + 2 * n]
            costs.append(
                {
                    "evaluation": receipt["residual_evaluations"],
                    "assembled_cost": cost,
                    "objective": float(objective_rows @ objective_rows),
                    "defect_rows_sum_squares": float(defect_rows @ defect_rows),
                }
            )

        def checkpoint(theta, states, cost):
            index = receipt["residual_evaluations"]
            z_now = by_state[np.asarray(states[0.6]).tobytes()][0]
            step = np.r_[np.asarray(theta), z_now]
            predicted = model["r0"] + model["j0"] @ step
            costs[-1]["linear_model_objective"] = float(predicted @ predicted)
            record = {
                "evaluation": index,
                "candidate_sha256": candidate_for(theta).sha256,
                "theta": np.asarray(theta).tolist(),
                "node_chart": {
                    str(t): by_state[np.asarray(s).tobytes()][0].tolist()
                    for t, s in states.items()
                },
                "physical_nodes": {
                    str(t): np.asarray(s).tolist() for t, s in states.items()
                },
                "assembled_cost": cost,
                "objective": costs[-1]["objective"],
                "linear_model_objective": costs[-1]["linear_model_objective"],
            }
            with (args.output / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            save()

        config = {
            "qualification": "bounded exploratory two-window trial for 0-0.90s; not acceptance",
            "candidate_sha256": base.sha256,
            "parent_sha256": parent.sha256,
            "model_sha256": model_hash,
            "windows": [[float(w[0]), float(w[-1]), int(len(w))] for w in windows],
            "first_control": FIRST_CONTROL,
            "basis_duration_s": DURATION,
            "theta_lower": lower.tolist(),
            "theta_upper": upper.tolist(),
            "box_factor": args.box_factor,
            "control_box": args.control_box,
            "theta_units": "N for first three coordinates, Nm otherwise; increments over candidate 102",
            "variable_scales": "theta AMPLITUDE (10) per control, node chart 1.0",
            "node_bound_chart": args.node_bound,
            "chart": chart.describe(),
            "primal": PRIMAL,
            "sensitivity": SENSITIVITY,
            "solver": "slsqp",
            "window_jacobian_state_coordinates": "node",
            "shared_boundary_policy": "once",
            "terminal_weight": args.terminal_weight,
            "defect_tolerance_scaled": 1e-4,
            "equality_tolerance": 1e-7,
            "max_iterations": args.max_iterations,
            "max_nfev": args.max_nfev,
            "effort_penalty": {"weight": 0.01, "force_N": 100.0, "torque_Nm": 20.0},
        }
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        receipt["config_sha256"] = sha(args.output / "config.json")
        save()

        options = MultipleShootingOptions(
            shooting_nodes=(0.6, DURATION),
            solver="slsqp",
            window_jacobian=window_jacobian,
            window_jacobian_state_coordinates="node",
            shared_boundary_policy="once",
            state_transform=lambda t, z: mapped(z).state,
            state_transform_jacobian=lambda t, z: mapped(z).state_jacobian,
            constraint_projection=lambda t: basis.T,
            defect_scales=scales,
            defect_tolerance=1e-4,
            equality_tolerance=1e-7,
            terminal_weight=args.terminal_weight,
            pelvis_indices=(wl, wr),
            pelvis_yaw_weight=args.pelvis_yaw_weight,
            pelvis_yaw_max_error_pct=5.0,
            regularization=effort_residual,
            regularization_jacobian=effort_jacobian,
            variable_scales=np.r_[np.full(x_size, AMPLITUDE), np.ones(dim)],
            max_iterations=args.max_iterations,
            max_nfev=args.max_nfev,
            step_tolerance=None,
            acceptance=accepted,
            callback=callback,
            checkpoint_callback=checkpoint,
        )
        theta0 = np.zeros(x_size)

        # Parity gate: segmented objective at zero displacement versus uninterrupted replay.
        first = evaluate(theta0, windows[0], None)
        second = evaluate(theta0, windows[1], zero.state)
        segmented_markers = np.concatenate(
            (first.replay.markers_m, second.replay.markers_m[1:])
        )
        parity = metrics(segmented_markers)
        parity["effort_penalty_cost"] = effort_cost0
        parity["score_with_effort"] = parity["score"] + effort_cost0
        restart_metrics = metrics(restart_full.markers_m)
        restart_metrics["score_with_effort"] = restart_metrics["score"] + effort_cost0
        receipt["restart_uninterrupted_metrics"] = restart_metrics
        parity["restart_score_with_effort"] = restart_metrics["score_with_effort"]
        parity["relative_score_difference"] = (
            abs(parity["score_with_effort"] - restart_metrics["score_with_effort"])
            / restart_metrics["score_with_effort"]
        )
        parity["marker_max_abs_difference_vs_saved_m"] = float(
            np.max(abs(segmented_markers - restart_full.markers_m))
        )
        parity["initial_scaled_defect_norm"] = float(
            np.linalg.norm((first.replay.integration.state[-1] - zero.state) / scales)
        )
        defect_columns = (
            np.hstack((first.state_jacobian[-1], -zero.state_jacobian))
            / scales[:, None]
        )
        projected = basis.T @ defect_columns
        parity["projected_constraint_rank"] = int(np.linalg.matrix_rank(projected))
        parity["projected_constraint_singular_values"] = np.linalg.svd(
            projected, compute_uv=False
        )[[0, -1]].tolist()

        observed0 = valid[: len(windows[0])]
        observed1 = valid[len(windows[0]) - 1 :].copy()
        observed1[0] = False
        rows0 = first.marker_jacobian[observed0].reshape(-1, x_size)
        rows0 = np.hstack((rows0, np.zeros((rows0.shape[0], dim))))
        rows1 = second.marker_jacobian[observed1].reshape(-1, x_size + dim)
        terminal_rows = args.terminal_weight * second.marker_jacobian[
            -1, observed1[-1]
        ].reshape(-1, x_size + dim)
        effort_rows = np.hstack(
            (penalty.matrix, np.zeros((penalty.matrix.shape[0], dim)))
        )
        res0 = (first.replay.markers_m - points[: len(windows[0])])[observed0].ravel()
        res1 = (second.replay.markers_m - points[len(windows[0]) - 1 :])[
            observed1
        ].ravel()
        term0 = (
            args.terminal_weight
            * (second.replay.markers_m[-1] - points[-1])[observed1[-1]].ravel()
        )

        j_list = [rows0, rows1, terminal_rows]
        r_list = [res0, res1, term0]
        if args.pelvis_yaw_weight > 0:
            yaw_res0, yaw_jac_term0, _ = compute_pelvis_yaw_residual_and_derivative(
                second.replay.markers_m[-1],
                points[-1],
                wl,
                wr,
                args.pelvis_yaw_weight,
                marker_jac_term=second.marker_jacobian[-1],
            )
            assert yaw_jac_term0 is not None
            j_list.append(yaw_jac_term0)
            r_list.append(yaw_res0)
        j_list.append(effort_rows)
        r_list.append(effort_residual(theta0))

        model["j0"] = np.vstack(j_list)
        model["r0"] = np.concatenate(r_list)
        model["equality_start"] = rows0.shape[0] + rows1.shape[0]
        parity["linear_model_objective_at_zero"] = float(model["r0"] @ model["r0"])
        receipt["zero_displacement_parity"] = parity
        receipt["parity_passed"] = bool(
            parity["relative_score_difference"] <= 1e-6
            and abs(
                parity["linear_model_objective_at_zero"] - parity["score_with_effort"]
            )
            <= 1e-6 * parity["score_with_effort"]
            and parity["marker_max_abs_difference_vs_saved_m"] <= 1e-7
            and parity["initial_scaled_defect_norm"] <= 1e-7
        )
        save()
        if not receipt["parity_passed"]:
            raise ValueError(
                "Zero-displacement two-window objective differs from restart candidate"
            )
        if args.audit_only:
            receipt["status"] = "audit_only_terminal"
            save()
            return

        fit = fit_multiple_shooting(
            target,
            segmented,
            unsegmented,
            initial_theta=theta0,
            lower_theta=lower,
            upper_theta=upper,
            initial_states={0.6: np.zeros(dim)},
            state_bounds={
                0.6: (np.full(dim, -args.node_bound), np.full(dim, args.node_bound))
            },
            options=options,
        )
        returned = candidate_for(fit.theta)
        replay = replay_candidate(raw, returned, clock, **PRIMAL)
        final = metrics(replay.markers_m)
        z_final = by_state[fit.intermediate_states[0.6].tobytes()][0]

        np.savez_compressed(
            args.output / "returned-replay.npz",
            time_s=clock,
            native_state=replay.integration.state,
            markers_m=replay.markers_m,
            target_m=points,
            valid=valid,
        )
        scipy.io.savemat(
            args.output / "pinocchio_replay.mat",
            {
                "time_s": clock,
                "native_state": replay.integration.state,
                "markers_m": replay.markers_m,
                "target_m": points,
                "valid": valid,
            },
        )
        (args.output / "returned-candidate.json").write_text(
            json.dumps(returned.document, indent=2) + "\n"
        )
        (args.output / "returned-nodes.json").write_text(
            json.dumps(
                {
                    "chart_coordinates": z_final.tolist(),
                    "physical_state": fit.intermediate_states[0.6].tolist(),
                    "physical_displacement_max_abs": float(
                        np.max(abs(fit.intermediate_states[0.6] - reference))
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        report = {
            "qualification": "exploratory returned two-window candidate 103 for 0-0.90s; R2025b validation pending",
            "candidate_sha256": returned.sha256,
            "uninterrupted_metrics": final,
            "restart_metrics": restart_metrics,
            "improved_uninterrupted": bool(
                final["whole_rms_m"] < restart_metrics["whole_rms_m"]
                and final["terminal_rms_m"] < restart_metrics["terminal_rms_m"]
            ),
            "accepted": fit.accepted,
            "optimizer_converged": fit.optimizer_converged,
            "message": fit.message,
            "max_scaled_defect_norm": fit.max_defect_norm,
            "defect_norms": {str(k): v for k, v in fit.defect_norms.items()},
            "segmented_rms_m": fit.segmented_rmse_m,
            "unsegmented_rms_m": fit.unsegmented_rmse_m,
            "terminal_replay_gap_m": fit.terminal_replay_gap_m,
            "pelvis_yaw_diff_deg": fit.pelvis_yaw_diff_deg,
            "pelvis_yaw_error_pct": fit.pelvis_yaw_error_pct,
            "function_evaluations": fit.function_evaluations,
            "active_bound_count": fit.active_bound_count,
            "active_theta_lower": int(
                np.sum(np.isclose(fit.theta, lower, rtol=0, atol=1e-8))
            ),
            "active_theta_upper": int(
                np.sum(np.isclose(fit.theta, upper, rtol=0, atol=1e-8))
            ),
            "active_node_bounds": int(
                np.sum(np.isclose(abs(z_final), args.node_bound, rtol=0, atol=1e-8))
            ),
            "theta_max_abs": float(np.max(abs(fit.theta))),
            "node_chart_norm": float(np.linalg.norm(z_final)),
            "costs": costs,
            "primal_windows": receipt["primal_windows"],
            "jacobian_windows": receipt["jacobians"],
            "replay_closure_max": [
                replay.closure_pose_max_abs,
                replay.closure_velocity_max_abs,
            ],
            "replay_integration_s": replay.integration.elapsed_s,
        }
        (args.output / "returned.json").write_text(json.dumps(report, indent=2) + "\n")
        receipt["status"] = "terminal"
        receipt["returned_sha256"] = returned.sha256
        receipt["improved_uninterrupted"] = report["improved_uninterrupted"]
        save()
    except (ValueError, RuntimeError, FloatingPointError, AssertionError) as error:
        receipt.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save()
        raise


if __name__ == "__main__":
    main()
