"""Bounded two-window control/node derivative audit with unchanged providers."""

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
from src.engines.physics_engines.pinocchio.python.native_replay import replay_window
from src.engines.physics_engines.pinocchio.python.native_sensitivity import (
    replay_marker_sensitivities,
)
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
)
from src.shared.python.motion_matching.node_retraction import retract_node
from src.shared.python.motion_matching.shooting_schedule import sampled_shooting_windows


def comparison(analytic: np.ndarray, central: np.ndarray) -> dict:
    difference = central - analytic
    norm = float(np.linalg.norm(analytic))
    central_norm = float(np.linalg.norm(central))
    error = float(np.linalg.norm(difference))
    weak = norm < 1e-12
    exact_zero = norm == 0 and central_norm == 0
    return {
        "analytic_l2": norm,
        "central_l2": central_norm,
        "absolute_l2_error": error,
        "maximum_absolute_error": float(np.max(abs(difference))),
        "relative_l2_error": error / max(norm, 1e-300),
        "weak_block": weak,
        "exact_structural_zero": exact_zero,
        "passed": exact_zero or (not weak and error / norm <= 1e-3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "candidate",
        "samples",
        "chart",
        "preflight_summary",
        "output",
        "runtime",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = perf_counter()
    receipt = {
        "status": "running",
        "directional_trials": 0,
        "audits": [],
        "window_sensitivities": [],
    }

    def save() -> None:
        receipt["elapsed_s"] = perf_counter() - started
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    try:
        raw = args.model.read_bytes()
        spec = json.loads(raw)
        names = spec["coordinate_order"]
        candidate = NativeReplayCandidate.from_document(
            json.loads(args.candidate.read_text()),
            names,
            hashlib.sha256(raw).hexdigest(),
        )
        assert (
            candidate.sha256
            == "786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a"
        )
        summary = json.loads(args.preflight_summary.read_text())
        assert (
            hashlib.sha256(args.chart.read_bytes()).hexdigest()
            == summary["output_hashes"]["node-chart.npz"]
        )
        samples = np.load(args.samples)
        clock = samples["time_s"]
        windows = sampled_shooting_windows(clock, [0.6, 0.85])
        chart = np.load(args.chart)
        reference, basis = chart["reference"], chart["basis"]
        scales, residual_scales = chart["state_scales"], chart["residual_scales"]
        np.testing.assert_array_equal(reference, samples["native_state"][216])
        engine = NativePinocchioModel(spec)

        def mapping(x: np.ndarray) -> dict[str, float]:
            return dict(zip(names, x.tolist(), strict=True))

        def closure(x: np.ndarray) -> np.ndarray:
            return np.concatenate(
                engine.closure_residuals(mapping(x[:27]), mapping(x[27:]))
            )

        def jacobian(x: np.ndarray) -> np.ndarray:
            value = engine.closure_trajectory_linearization(
                mapping(x[:27]),
                mapping(x[27:]),
                dict.fromkeys(names, 0.0),
                finite_difference_step=1e-6,
            )
            return np.hstack((value.dq[:12], value.dv[:12]))

        def retract(z: np.ndarray):
            return retract_node(
                reference,
                basis,
                z,
                closure,
                jacobian,
                state_scales=scales,
                residual_scales=residual_scales,
                radius=0.5,
                tolerance=1e-8,
            )

        zero = retract(np.zeros(42))
        initial = np.asarray(candidate.document["q0"] + candidate.document["qd0"])
        settings = {
            "first_control": 4,
            "basis_duration_s": 0.85,
            "rtol": 1e-10,
            "atol": 1e-12,
            "max_step": 0.0000625,
            "max_sensitivity_evaluations": 200000,
            "separate_error_control": True,
        }
        receipt.update(
            candidate_sha256=candidate.sha256,
            settings=settings,
            gate_relative_l2=1e-3,
            weak_norm_reporting_threshold=1e-12,
            node_centering="Existing zero-retraction state and its implicit Jacobian; original saved node remains chart reference",
            zero_retraction_shift=float(np.max(abs(zero.state - reference))),
            input_hashes={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    args.model,
                    args.candidate,
                    args.samples,
                    args.chart,
                    args.preflight_summary,
                    Path(__file__),
                )
            },
            source_hashes={
                str(p.relative_to(args.runtime)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in args.runtime.rglob("*.py")
            },
        )
        save()
        sensitivities = []
        for i, (time, state) in enumerate(
            zip(windows, (initial, zero.state), strict=True)
        ):
            extra = {} if i == 0 else {"initial_sensitivity": zero.state_jacobian}
            result = replay_marker_sensitivities(
                raw, candidate, time, initial_state=state, **extra, **settings
            )
            sensitivities.append(result)
            np.savez_compressed(
                args.output / f"sensitivity-{i}.npz",
                time_s=time,
                marker_jacobian=result.marker_jacobian,
                state_jacobian=result.state_jacobian,
                primal_state=result.replay.integration.state,
                primal_markers_m=result.replay.markers_m,
            )
            receipt["window_sensitivities"].append(
                {
                    "index": i,
                    "evaluations": result.sensitivity_evaluations,
                    "elapsed_s": result.sensitivity_elapsed_s,
                    "primal_marker_max_abs_difference_m": result.primal_marker_max_abs_difference_m,
                }
            )
            save()
        _, singular, right = np.linalg.svd(zero.state_jacobian[:27], full_matrices=True)
        node_position, node_velocity = right[0], right[-1]
        velocity_position_norm = float(
            np.linalg.norm(zero.state_jacobian[:27] @ node_velocity)
        )
        assert velocity_position_norm <= 1e-10
        control = np.zeros(81)
        control[names.index("LSInputX") * 3 + 2] = 1
        rng = np.random.default_rng(996776)
        mixed = np.r_[rng.normal(size=81), node_position + node_velocity]
        mixed /= np.linalg.norm(mixed)
        directions = {
            "LSInputX_B6": (np.r_[control, np.zeros(42)], [1e-5, 1e-4]),
            "node_position": (np.r_[np.zeros(81), node_position], [1e-5, 1e-4]),
            "node_velocity": (np.r_[np.zeros(81), node_velocity], [1e-5, 1e-4]),
            "mixed": (mixed, [1e-7, 1e-6]),
        }
        receipt["directions"] = {
            name: {"vector": d.tolist(), "steps": steps}
            for name, (d, steps) in directions.items()
        }
        receipt["node_direction_diagnostics"] = {
            "Dq_singular_values": singular.tolist(),
            "node_velocity_position_norm": velocity_position_norm,
            "node_velocity_rate_norm": float(
                np.linalg.norm(zero.state_jacobian[27:] @ node_velocity)
            ),
        }
        np.savez_compressed(
            args.output / "node-chart.npz",
            reference=reference,
            zero_state=zero.state,
            state_jacobian=zero.state_jacobian,
            basis=basis,
            state_scales=scales,
            residual_scales=residual_scales,
        )
        save()
        marker_valid = samples["valid"] & np.isfinite(samples["target_m"]).all(axis=2)
        first_base = sensitivities[0].replay

        def outputs(first, second, node):
            markers = np.concatenate((first.markers_m, second.markers_m[1:]))
            endpoints = np.stack(
                (first.integration.state[-1], second.integration.state[-1])
            )
            continuity = first.integration.state[-1] - node
            return {
                "markers": markers[marker_valid],
                "endpoint_q": endpoints[:, :27],
                "endpoint_qd": endpoints[:, 27:],
                "endpoint_scaled": endpoints / scales,
                "continuity_q": continuity[:27],
                "continuity_qd": continuity[27:],
                "continuity_scaled": continuity / scales,
            }

        base_outputs = outputs(first_base, sensitivities[1].replay, zero.state)
        for name, (direction, steps) in directions.items():
            dc, dz = direction[:81], direction[81:]
            first_marker = sensitivities[0].marker_jacobian @ dc
            second_marker = sensitivities[1].marker_jacobian @ direction
            end_first = sensitivities[0].state_jacobian[-1] @ dc
            end_second = sensitivities[1].state_jacobian[-1] @ direction
            endpoints = np.stack((end_first, end_second))
            continuity = end_first - zero.state_jacobian @ dz
            analytic = {
                "markers": np.concatenate((first_marker, second_marker[1:]))[
                    marker_valid
                ],
                "endpoint_q": endpoints[:, :27],
                "endpoint_qd": endpoints[:, 27:],
                "endpoint_scaled": endpoints / scales,
                "continuity_q": continuity[:27],
                "continuity_qd": continuity[27:],
                "continuity_scaled": continuity / scales,
            }
            for h in steps:
                trial_outputs = []
                for sign in (1, -1):
                    if receipt["directional_trials"] >= 16:
                        raise RuntimeError(
                            "Explicit16 directional trial budget exhausted"
                        )
                    receipt["directional_trials"] += 1
                    z = sign * h * dz
                    node = retract(z).state
                    perturbed = candidate
                    if np.any(dc):
                        controls = np.zeros((27, 7))
                        controls[:, 4:] = sign * h * dc.reshape(27, 3)
                        perturbed = increment_native_bernstein(
                            candidate, controls, basis_duration_s=0.85
                        )
                    first = (
                        first_base
                        if not np.any(dc)
                        else replay_window(
                            raw,
                            perturbed,
                            windows[0],
                            initial,
                            rtol=1e-11,
                            atol=1e-13,
                            max_step=0.0000625,
                        )
                    )
                    second = replay_window(
                        raw,
                        perturbed,
                        windows[1],
                        node,
                        rtol=1e-11,
                        atol=1e-13,
                        max_step=0.0000625,
                    )
                    trial_outputs.append(outputs(first, second, node))
                    label = f"{name}-{h:g}-{'plus' if sign > 0 else 'minus'}"
                    np.savez_compressed(
                        args.output / f"{label}.npz",
                        time0=windows[0],
                        time1=windows[1],
                        state0=first.integration.state,
                        state1=second.integration.state,
                        markers0=first.markers_m,
                        markers1=second.markers_m,
                        node=node,
                        coefficients=np.asarray(perturbed.document["coefficients"]),
                    )
                    receipt.setdefault("trials", []).append(
                        {
                            "label": label,
                            "candidate_sha256": perturbed.sha256,
                            "first_window_reused": not bool(np.any(dc)),
                            "second_integration_evaluations": second.integration.evaluations,
                            "first_integration_evaluations": 0
                            if not np.any(dc)
                            else first.integration.evaluations,
                        }
                    )
                    save()
                central = {
                    key: (trial_outputs[0][key] - trial_outputs[1][key]) / (2 * h)
                    for key in analytic
                }
                blocks = {
                    key: comparison(analytic[key], central[key]) for key in analytic
                }
                even = {
                    key: float(
                        np.linalg.norm(
                            (
                                trial_outputs[0][key]
                                + trial_outputs[1][key]
                                - 2 * base_outputs[key]
                            )
                            / (2 * h)
                        )
                    )
                    for key in analytic
                }
                receipt["audits"].append(
                    {
                        "direction": name,
                        "step": h,
                        "blocks": blocks,
                        "even_response_l2_per_step": even,
                        "note": "Even response combines curvature and numerical noise; not a proven integration-error floor",
                        "passed": all(block["passed"] for block in blocks.values()),
                    }
                )
                np.savez_compressed(
                    args.output / f"comparison-{name}-{h:g}.npz",
                    **{f"analytic_{key}": value for key, value in analytic.items()},
                    **{f"central_{key}": value for key, value in central.items()},
                )
                save()
        receipt["status"] = (
            "passed"
            if all(row["passed"] for row in receipt["audits"])
            else "failed_derivative_gates"
        )
        save()
    except (ValueError, RuntimeError, FloatingPointError, AssertionError) as error:
        receipt.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save()
        raise


if __name__ == "__main__":
    main()
