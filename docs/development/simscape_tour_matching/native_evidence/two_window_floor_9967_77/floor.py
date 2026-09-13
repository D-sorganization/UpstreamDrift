"""Bounded derivative-resolution audit at the exact run76 fixture.

Measures per-block replay error by tolerance variation, reruns only the four
failed small-step trials at tighter/looser tolerance plus one better-resolved
mixed step, and classifies every run76 block with the shared provider. No
gate is widened; no optimizer runs. Archived run76 Jacobians are reused.
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
from src.engines.physics_engines.pinocchio.python.native_replay import replay_window
from src.shared.python.motion_matching.derivative_resolution import (
    DerivativeBlockVerdict,
    classify_derivative_block,
    cross_block_norm_bound,
    qualify_direction,
)
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
)
from src.shared.python.motion_matching.shooting_schedule import sampled_shooting_windows

CANDIDATE = "786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a"
TOLERANCES = {"work": (1e-11, 1e-13), "tight": (1e-12, 1e-14), "loose": (1e-10, 1e-12)}
MAX_STEP = 0.0000625
BLOCKS = (
    "markers",
    "endpoint_q",
    "endpoint_qd",
    "endpoint_scaled",
    "continuity_q",
    "continuity_qd",
    "continuity_scaled",
)
EXPECTED_ZERO = {"node_position": "continuity_qd", "node_velocity": "continuity_q"}
REPLAY_BUDGET = 32


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "candidate",
        "samples",
        "audit76",
        "audit76_summary",
        "output",
        "runtime",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = perf_counter()
    receipt: dict = {"status": "running", "window_replays": 0, "trials": []}

    def save() -> None:
        receipt["elapsed_s"] = perf_counter() - started
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    try:
        raw = args.model.read_bytes()
        spec = json.loads(raw)
        names = spec["coordinate_order"]
        n = len(names)
        candidate = NativeReplayCandidate.from_document(
            json.loads(args.candidate.read_text()),
            names,
            hashlib.sha256(raw).hexdigest(),
        )
        assert candidate.sha256 == CANDIDATE
        summary76 = json.loads(args.audit76_summary.read_text())
        for name in (
            "receipt.json",
            "sensitivity-0.npz",
            "sensitivity-1.npz",
            "node-chart.npz",
        ):
            assert sha(args.audit76 / name) == summary76["output_hashes"][name], name
        receipt76 = json.loads((args.audit76 / "receipt.json").read_text())
        assert receipt76["input_hashes"][str(args.samples)] == sha(args.samples)
        assert receipt76["input_hashes"][str(args.model)] == sha(args.model)
        samples = np.load(args.samples)
        clock = samples["time_s"]
        windows = sampled_shooting_windows(clock, [0.6, 0.85])
        chart76 = np.load(args.audit76 / "node-chart.npz")
        reference, zero_state = chart76["reference"], chart76["zero_state"]
        jac_zero, basis = chart76["state_jacobian"], chart76["basis"]
        scales = chart76["state_scales"]
        np.testing.assert_array_equal(reference, samples["native_state"][216])
        chart = NativeNodeChart(
            names,
            NativePinocchioModel(spec),
            state_scales=scales,
            residual_scales=chart76["residual_scales"],
            radius=0.5,
            tolerance=1e-8,
            jacobian_step=1e-6,
        )
        zero = chart.retract(reference, basis, np.zeros(basis.shape[1]))
        assert np.max(abs(zero.state - zero_state)) <= 1e-12
        assert np.max(abs(zero.state_jacobian - jac_zero)) <= 1e-9
        initial = np.asarray(candidate.document["q0"] + candidate.document["qd0"])
        marker_valid = samples["valid"] & np.isfinite(samples["target_m"]).all(axis=2)
        sensitivities = [np.load(args.audit76 / f"sensitivity-{i}.npz") for i in (0, 1)]
        receipt.update(
            candidate_sha256=candidate.sha256,
            tolerances=TOLERANCES,
            max_step=MAX_STEP,
            gate_relative_l2=1e-3,
            chart=chart.describe(),
            replay_budget=REPLAY_BUDGET,
            input_hashes={
                str(p): sha(p)
                for p in (
                    args.model,
                    args.candidate,
                    args.samples,
                    args.audit76_summary,
                    Path(__file__),
                )
            },
            audit76_hashes={
                k: sha(args.audit76 / k) for k in summary76["output_hashes"]
            },
            source_hashes={
                str(p.relative_to(args.runtime)): sha(p)
                for p in args.runtime.rglob("*.py")
            },
        )
        save()

        def replay(cand, window, state, tol):
            if receipt["window_replays"] >= REPLAY_BUDGET:
                raise RuntimeError("Explicit window replay budget exhausted")
            receipt["window_replays"] += 1
            rtol, atol = TOLERANCES[tol]
            return replay_window(
                raw, cand, window, state, rtol=rtol, atol=atol, max_step=MAX_STEP
            )

        def outputs(first, second, node):
            markers = np.concatenate((first.markers_m, second.markers_m[1:]))
            endpoints = np.stack(
                (first.integration.state[-1], second.integration.state[-1])
            )
            continuity = first.integration.state[-1] - node
            return {
                "markers": markers[marker_valid],
                "endpoint_q": endpoints[:, :n],
                "endpoint_qd": endpoints[:, n:],
                "endpoint_scaled": endpoints / scales,
                "continuity_q": continuity[:n],
                "continuity_qd": continuity[n:],
                "continuity_scaled": continuity / scales,
            }

        # 1. Base replays at three tolerances; measure per-block replay error.
        base = {}
        for tol in TOLERANCES:
            first = replay(candidate, windows[0], initial, tol)
            second = replay(candidate, windows[1], zero.state, tol)
            base[tol] = (first, second)
            np.savez_compressed(
                args.output / f"base-{tol}.npz",
                state0=first.integration.state,
                state1=second.integration.state,
                markers0=first.markers_m,
                markers1=second.markers_m,
            )
            receipt.setdefault("base", {})[tol] = {
                "evaluations": [
                    first.integration.evaluations,
                    second.integration.evaluations,
                ],
                "integration_s": [
                    first.integration.elapsed_s,
                    second.integration.elapsed_s,
                ],
            }
            save()
        base_out = {
            tol: outputs(base[tol][0], base[tol][1], zero.state) for tol in TOLERANCES
        }
        receipt["reproduces_audit76_primal"] = {
            "state0_max_abs": float(
                np.max(
                    abs(
                        base["work"][0].integration.state
                        - sensitivities[0]["primal_state"]
                    )
                )
            ),
            "state1_max_abs": float(
                np.max(
                    abs(
                        base["work"][1].integration.state
                        - sensitivities[1]["primal_state"]
                    )
                )
            ),
        }

        def block_error(a, b):
            return {k: float(np.linalg.norm(a[k] - b[k])) for k in BLOCKS}

        replay_error = {
            "work": block_error(base_out["work"], base_out["tight"]),
            "loose": block_error(base_out["loose"], base_out["tight"]),
        }
        replay_error["tight"] = dict(replay_error["work"])
        # Window-only pieces for directions that reuse the first window.
        w1 = {
            tol: {
                "markers": float(
                    np.linalg.norm(
                        (base[tol][1].markers_m[1:] - base["tight"][1].markers_m[1:])[
                            marker_valid[len(windows[0]) :]
                        ]
                    )
                ),
                "endpoint": float(
                    np.linalg.norm(
                        base[tol][1].integration.state[-1]
                        - base["tight"][1].integration.state[-1]
                    )
                ),
            }
            for tol in ("work", "loose")
        }
        receipt["replay_error_l2"] = replay_error
        receipt["window1_only_error_l2"] = w1
        receipt["replay_error_note"] = (
            "Per-block L2 difference between the stated tolerance and the tight replay; "
            "tight replays are assumed at least as accurate as work, so the tight floor "
            "reuses the work-versus-tight measurement conservatively."
        )
        save()

        # 2. Retraction resolution for node-only directions from archived nodes.
        directions = {
            k: np.array(v["vector"]) for k, v in receipt76["directions"].items()
        }
        eps = np.finfo(float).eps
        retraction_error: dict = {}
        for name in ("node_position", "node_velocity"):
            dz = directions[name][81:]
            per_step = {}
            for h in receipt76["directions"][name]["steps"]:
                deviation = []
                for sign, label in ((1, "plus"), (-1, "minus")):
                    node = np.load(args.audit76 / f"{name}-{h:g}-{label}.npz")["node"]
                    deviation.append(node - (zero.state + sign * h * (jac_zero @ dz)))
                dev = np.max(np.abs(deviation), axis=0) + eps * np.abs(zero.state)
                per_step[str(h)] = {
                    "continuity_q": float(np.linalg.norm(dev[:n])),
                    "continuity_qd": float(np.linalg.norm(dev[n:])),
                    "continuity_scaled": float(np.linalg.norm(dev / scales)),
                }
            retraction_error[name] = per_step
        receipt["retraction_error_l2"] = retraction_error
        receipt["structural_zero_bounds"] = {
            "node_position_velocity_block": cross_block_norm_bound(
                jac_zero,
                scales,
                directions["node_position"][81:],
                block=slice(0, n),
                other=slice(n, 2 * n),
            ),
            "node_velocity_position_block": cross_block_norm_bound(
                jac_zero,
                scales,
                directions["node_velocity"][81:],
                block=slice(n, 2 * n),
                other=slice(0, n),
            ),
            "node_position_velocity_analytic": float(
                np.linalg.norm(jac_zero[n:] @ directions["node_position"][81:])
            ),
            "node_velocity_position_analytic": float(
                np.linalg.norm(jac_zero[:n] @ directions["node_velocity"][81:])
            ),
        }
        save()

        def floors(name, tol, h):
            dc, dz = directions[name][:81], directions[name][81:]
            node_only = not np.any(dc)
            err = replay_error[tol]
            out = {}
            for key in BLOCKS:
                if node_only:
                    if key.startswith("continuity"):
                        out[key] = retraction_error[name][str(h)][key]
                    elif key == "markers":
                        out[key] = w1[tol if tol != "tight" else "work"]["markers"]
                    else:
                        out[key] = w1[tol if tol != "tight" else "work"]["endpoint"]
                else:
                    out[key] = err[key]
                    if key.startswith("continuity") and np.any(dz):
                        out[key] = float(
                            np.hypot(out[key], eps * np.linalg.norm(zero.state))
                        )
            return out

        def analytic_for(direction):
            dc = direction[:81]
            first_marker = sensitivities[0]["marker_jacobian"] @ dc
            second_marker = sensitivities[1]["marker_jacobian"] @ direction
            end_first = sensitivities[0]["state_jacobian"][-1] @ dc
            end_second = sensitivities[1]["state_jacobian"][-1] @ direction
            endpoints = np.stack((end_first, end_second))
            continuity = end_first - jac_zero @ direction[81:]
            return {
                "markers": np.concatenate((first_marker, second_marker[1:]))[
                    marker_valid
                ],
                "endpoint_q": endpoints[:, :n],
                "endpoint_qd": endpoints[:, n:],
                "endpoint_scaled": endpoints / scales,
                "continuity_q": continuity[:n],
                "continuity_qd": continuity[n:],
                "continuity_scaled": continuity / scales,
            }

        def classify(name, h, tol, analytic, central):
            floor = floors(name, tol, h)
            result = {}
            for key in BLOCKS:
                verdict = classify_derivative_block(
                    analytic[key],
                    central[key],
                    step=h,
                    replay_error=floor[key],
                    expected_zero=EXPECTED_ZERO.get(name) == key,
                )
                result[key] = verdict._asdict()
            return result

        # 3. Classify every archived run76 block with measured floors.
        verdicts: dict = {}
        for name, spec76 in receipt76["directions"].items():
            analytic = analytic_for(directions[name])
            for h in spec76["steps"]:
                comp = np.load(args.audit76 / f"comparison-{name}-{h:g}.npz")
                for key in BLOCKS:
                    np.testing.assert_allclose(
                        comp[f"analytic_{key}"], analytic[key], rtol=0, atol=1e-12
                    )
                central = {key: comp[f"central_{key}"] for key in BLOCKS}
                verdicts.setdefault(name, []).append(
                    {
                        "step": h,
                        "tolerance": "work",
                        "source": "audit76",
                        "blocks": classify(name, h, "work", analytic, central),
                    }
                )
        save()

        # 4. Bounded new trials: failed small steps at tight/loose, mixed at 1e-5 work.
        plan = [
            ("LSInputX_B6", 1e-5, "tight"),
            ("LSInputX_B6", 1e-5, "loose"),
            ("mixed", 1e-7, "tight"),
            ("mixed", 1e-7, "loose"),
            ("mixed", 1e-5, "work"),
        ]
        for name, h, tol in plan:
            direction = directions[name]
            dc, dz = direction[:81], direction[81:]
            analytic = analytic_for(direction)
            trial_outputs = []
            for sign, label in ((1, "plus"), (-1, "minus")):
                node = (
                    chart.retract(reference, basis, sign * h * dz).state
                    if np.any(dz)
                    else zero.state
                )
                perturbed = candidate
                if np.any(dc):
                    controls = np.zeros((n, 7))
                    controls[:, 4:] = sign * h * dc.reshape(n, 3)
                    perturbed = increment_native_bernstein(
                        candidate, controls, basis_duration_s=0.85
                    )
                first = replay(perturbed, windows[0], initial, tol)
                second = replay(perturbed, windows[1], node, tol)
                trial_outputs.append(outputs(first, second, node))
                tag = f"{name}-{h:g}-{tol}-{label}"
                np.savez_compressed(
                    args.output / f"{tag}.npz",
                    state0=first.integration.state,
                    state1=second.integration.state,
                    markers0=first.markers_m,
                    markers1=second.markers_m,
                    node=node,
                    coefficients=np.asarray(perturbed.document["coefficients"]),
                )
                receipt["trials"].append(
                    {
                        "label": tag,
                        "candidate_sha256": perturbed.sha256,
                        "evaluations": [
                            first.integration.evaluations,
                            second.integration.evaluations,
                        ],
                        "integration_s": [
                            first.integration.elapsed_s,
                            second.integration.elapsed_s,
                        ],
                    }
                )
                save()
            central = {
                k: (trial_outputs[0][k] - trial_outputs[1][k]) / (2 * h) for k in BLOCKS
            }
            np.savez_compressed(
                args.output / f"comparison-{name}-{h:g}-{tol}.npz",
                **{f"analytic_{k}": v for k, v in analytic.items()},
                **{f"central_{k}": v for k, v in central.items()},
            )
            verdicts[name].append(
                {
                    "step": h,
                    "tolerance": tol,
                    "source": "audit77",
                    "blocks": classify(name, h, tol, analytic, central),
                }
            )
            receipt["verdicts"] = verdicts
            save()

        # 5. Qualification: every block of every direction needs a resolved pass
        # or verified structural zero at some step, and no unexplained failure.
        qualification = {}
        for name, rows in verdicts.items():
            qualification[name] = {}
            for key in BLOCKS:
                series = [DerivativeBlockVerdict(**row["blocks"][key]) for row in rows]
                qualification[name][key] = qualify_direction(series)
        receipt["verdicts"] = verdicts
        receipt["qualification"] = qualification
        receipt["status"] = (
            "qualified"
            if all(all(v.values()) for v in qualification.values())
            else "unqualified_derivative_blocks"
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
