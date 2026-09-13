"""Bounded two-window direct-node SLSQP trial 79 from the exact run73 candidate.

Same formulation as run78 with two changes: residual evaluations use primal
window replays only (sensitivities are integrated only when the optimizer asks
for a Jacobian), and larger iteration/evaluation budgets so SLSQP can reach a
continuity-feasible iterate. The callback additionally records the objective
without defect rows and the first-Jacobian linear-model prediction so predicted
versus actual reduction is explicit. Stage3 of NEXT_AGENT_CONVERGENCE_EXECUTION.md: exact run73 coefficients with
zero Bernstein increments, the run73 B4/B5/B6 subset and effort penalty,
bounds derived from run73's saved optimizer parameters, the integrated0.6 s
node with zero chart coordinates, direct node-chart Jacobians, once-only
shared-boundary observations and small iteration/evaluation budgets. The
segmented objective at zero displacement must reproduce run73's final score
before any optimization. Final metrics use an uninterrupted original-state
replay. Nothing here is acceptance of a full swing.
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
from src.shared.python.motion_matching.prefix_fit import MarkerTarget
from src.shared.python.motion_matching.shooting_schedule import (
    sampled_shooting_windows,
)

CANDIDATE73 = "786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a"
PARENT19 = "b5b1c3823c86a21df323dc4e430366dc093495b069b3d25c361b0d007b8ff24f"
FIRST_CONTROL, DURATION, AMPLITUDE = 4, 0.85, 10.0
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
        "candidate",
        "parent",
        "returned73",
        "target",
        "samples",
        "chart76",
        "output",
        "runtime",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--max-nfev", type=int, default=45)
    parser.add_argument("--node-bound", type=float, default=0.05)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
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

        base = load(args.candidate)
        parent = load(args.parent)
        assert base.sha256 == CANDIDATE73 and parent.sha256 == PARENT19
        doc = base.document
        returned73 = json.loads(args.returned73.read_text())
        x73 = np.asarray(returned73["parameters"], dtype=float)
        assert x73.shape == (n * (7 - FIRST_CONTROL),)
        assert np.all(x73 >= 0.8) and np.all(x73 <= 1.2)
        # Equivalent physical bounds: run73 used increments AMPLITUDE*(x-1) with
        # x in [0.8, 1.2] relative to the parent; theta is relative to run73.
        lower = AMPLITUDE * (0.8 - x73)
        upper = AMPLITUDE * (1.2 - x73)
        assert np.all(lower <= 0) and np.all(upper >= 0) and np.all(lower < upper)
        delta73 = recover_native_bernstein(parent, base, basis_duration_s=DURATION)
        assert np.all(delta73[:, :FIRST_CONTROL] == 0)
        delta73_flat = delta73[:, FIRST_CONTROL:].ravel()
        receipt["restart_roundoff"] = {
            "max_abs_recovered_minus_amplitude_x_minus_1": float(
                np.max(abs(delta73_flat - AMPLITUDE * (x73 - 1)))
            ),
            "note": "Recovered Bernstein increment versus run73 parameter map; theta is applied to the exact run73 coefficients, never reconstructed from the parent",
        }
        payload = json.loads(args.target.read_text())
        assert payload["source_sha256"] == doc["capture_sha256"]
        idx = [payload["labels"].index(label) for label in doc["marker_labels"]]
        full_clock = np.asarray(payload["time_s"], dtype=float)
        mask = full_clock <= DURATION
        clock = full_clock[mask]
        points = np.asarray(payload["points_world_m"])[mask][:, idx].copy()
        valid = np.asarray(payload["valid"], dtype=bool)[mask][:, idx]
        points[~valid] = np.nan
        target = MarkerTarget(clock, points, np.ones(len(idx)))
        samples = np.load(args.samples)
        np.testing.assert_array_equal(samples["time_s"], clock)
        windows = sampled_shooting_windows(clock, [0.6, DURATION])
        reference = samples["native_state"][216]
        initial = np.asarray(doc["q0"] + doc["qd0"])
        np.testing.assert_array_equal(samples["native_state"][0], initial)
        chart76 = np.load(args.chart76)
        np.testing.assert_array_equal(chart76["reference"], reference)
        chart = NativeNodeChart(
            names,
            NativePinocchioModel(spec),
            state_scales=chart76["state_scales"],
            residual_scales=chart76["residual_scales"],
            radius=0.5,
            tolerance=1e-8,
            jacobian_step=1e-6,
        )
        scales = chart76["state_scales"]
        basis = chart.basis(reference)
        assert np.max(abs(basis - chart76["basis"])) <= 1e-12
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

        # Controls: theta are physical B4/B5/B6 Bernstein increments over run73.
        def candidate_for(theta: np.ndarray) -> NativeReplayCandidate:
            if not np.any(theta):
                return base
            controls = np.zeros((n, 7))
            controls[:, FIRST_CONTROL:] = np.asarray(theta).reshape(n, -1)
            return increment_native_bernstein(base, controls, basis_duration_s=DURATION)

        assert candidate_for(np.zeros(x73.size)).sha256 == CANDIDATE73
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
            return penalty.residual(delta73_flat + theta)

        def effort_jacobian(theta: np.ndarray) -> np.ndarray:
            return penalty.jacobian(delta73_flat + theta)

        effort_cost0 = float(
            effort_residual(np.zeros(x73.size)) @ effort_residual(np.zeros(x73.size))
        )
        receipt["effort_cost_zero"] = effort_cost0
        receipt["effort_cost_run73"] = returned73["metrics"]["effort_penalty_cost"]

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
            vp = pred[-1, wr, :2] - pred[-1, wl, :2]
            vo = points[-1, wr, :2] - points[-1, wl, :2]
            yt = np.degrees(np.arctan2(vo[1], vo[0]))
            yp = np.degrees(np.arctan2(vp[1], vp[0]))
            return {
                "whole_rms_m": float(np.sqrt(np.mean(error[valid]))),
                "early_rms_m": float(np.sqrt(np.mean(error[early]))),
                "terminal_rms_m": float(np.sqrt(np.mean(error[-1, valid[-1]]))),
                "club_cluster_rms_m": float(
                    np.sqrt(np.mean(error[-1, club & valid[-1]]))
                ),
                "pelvis_yaw_error_pct": float(
                    abs((yp - yt + 180) % 360 - 180) / max(abs(yt), 1.0) * 100
                ),
                "score": float(
                    np.sum(error[valid]) + 100 * np.sum(error[-1, valid[-1]])
                ),
            }

        def accepted(pred: np.ndarray) -> bool:
            m = metrics(pred)
            return (
                m["whole_rms_m"] <= 0.025
                and m["early_rms_m"] <= 0.012
                and m["terminal_rms_m"] <= 0.035
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
            "qualification": "bounded exploratory two-window trial; not acceptance",
            "candidate_sha256": base.sha256,
            "parent_sha256": parent.sha256,
            "model_sha256": model_hash,
            "returned73_sha256": sha(args.returned73),
            "windows": [[float(w[0]), float(w[-1]), int(len(w))] for w in windows],
            "first_control": FIRST_CONTROL,
            "basis_duration_s": DURATION,
            "theta_lower": lower.tolist(),
            "theta_upper": upper.tolist(),
            "theta_units": "N for first three coordinates, Nm otherwise; increments over exact run73 coefficients",
            "variable_scales": "theta AMPLITUDE (10) per control, node chart 1.0",
            "node_bound_chart": args.node_bound,
            "chart": chart.describe(),
            "primal": PRIMAL,
            "sensitivity": SENSITIVITY,
            "solver": "slsqp",
            "window_jacobian_state_coordinates": "node",
            "shared_boundary_policy": "once",
            "terminal_weight": 10.0,
            "defect_tolerance_scaled": 1e-4,
            "equality_tolerance": 1e-7,
            "max_iterations": args.max_iterations,
            "max_nfev": args.max_nfev,
            "effort_penalty": {"weight": 0.01, "force_N": 100.0, "torque_Nm": 20.0},
            "input_hashes": {
                k: sha(getattr(args, k))
                for k in (
                    "model",
                    "candidate",
                    "parent",
                    "returned73",
                    "target",
                    "samples",
                    "chart76",
                )
            },
            "driver_sha256": sha(Path(__file__)),
            "source_hashes": {
                str(p.relative_to(args.runtime)): sha(p)
                for p in args.runtime.rglob("*.py")
            },
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
            terminal_weight=10.0,
            regularization=effort_residual,
            regularization_jacobian=effort_jacobian,
            variable_scales=np.r_[np.full(x73.size, AMPLITUDE), np.ones(dim)],
            max_iterations=args.max_iterations,
            max_nfev=args.max_nfev,
            step_tolerance=None,
            acceptance=accepted,
            callback=callback,
            checkpoint_callback=checkpoint,
        )
        theta0 = np.zeros(x73.size)

        # Parity gate: segmented objective at zero displacement versus run73.
        first = evaluate(theta0, windows[0], None)
        second = evaluate(theta0, windows[1], zero.state)
        segmented_markers = np.concatenate(
            (first.replay.markers_m, second.replay.markers_m[1:])
        )
        parity = metrics(segmented_markers)
        parity["effort_penalty_cost"] = effort_cost0
        parity["score_with_effort"] = parity["score"] + effort_cost0
        parity["run73_score"] = returned73["metrics"]["score"]
        parity["relative_score_difference"] = (
            abs(parity["score_with_effort"] - returned73["metrics"]["score"])
            / returned73["metrics"]["score"]
        )
        parity["marker_max_abs_difference_vs_saved_m"] = float(
            np.max(abs(segmented_markers - samples["returned_m"]))
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
        # Linear model of the objective rows at zero displacement, in the exact
        # row order used by fit_multiple_shooting with once-only boundaries.
        observed0 = valid[: len(windows[0])]
        observed1 = valid[len(windows[0]) - 1 :].copy()
        observed1[0] = False
        rows0 = first.marker_jacobian[observed0].reshape(-1, x73.size)
        rows0 = np.hstack((rows0, np.zeros((rows0.shape[0], dim))))
        rows1 = second.marker_jacobian[observed1].reshape(-1, x73.size + dim)
        terminal_rows = 10.0 * second.marker_jacobian[-1, observed1[-1]].reshape(
            -1, x73.size + dim
        )
        effort_rows = np.hstack(
            (penalty.matrix, np.zeros((penalty.matrix.shape[0], dim)))
        )
        model["j0"] = np.vstack((rows0, rows1, terminal_rows, effort_rows))
        res0 = (first.replay.markers_m - points[: len(windows[0])])[observed0].ravel()
        res1 = (second.replay.markers_m - points[len(windows[0]) - 1 :])[
            observed1
        ].ravel()
        term0 = 10.0 * (second.replay.markers_m[-1] - points[-1])[observed1[-1]].ravel()
        model["r0"] = np.concatenate((res0, res1, term0, effort_residual(theta0)))
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
                "Zero-displacement two-window objective differs from run73"
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
            "qualification": "exploratory returned two-window candidate; R2025b acceptance pending",
            "candidate_sha256": returned.sha256,
            "uninterrupted_metrics": final,
            "run73_metrics": {
                k: returned73["metrics"][k]
                for k in (
                    "whole_rms_m",
                    "early_rms_m",
                    "terminal_rms_m",
                    "club_cluster_rms_m",
                    "score",
                )
            },
            "improved_uninterrupted": bool(
                final["whole_rms_m"] < returned73["metrics"]["whole_rms_m"]
                and final["terminal_rms_m"] < returned73["metrics"]["terminal_rms_m"]
            ),
            "accepted": fit.accepted,
            "optimizer_converged": fit.optimizer_converged,
            "message": fit.message,
            "max_scaled_defect_norm": fit.max_defect_norm,
            "defect_norms": {str(k): v for k, v in fit.defect_norms.items()},
            "segmented_rms_m": fit.segmented_rmse_m,
            "unsegmented_rms_m": fit.unsegmented_rmse_m,
            "terminal_replay_gap_m": fit.terminal_replay_gap_m,
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
