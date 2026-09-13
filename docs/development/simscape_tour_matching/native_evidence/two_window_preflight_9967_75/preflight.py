"""Two-window zero-displacement preflight; no optimizer or target-state resets."""

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
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.shared.python.motion_matching.node_retraction import (
    retract_node,
    scaled_tangent_basis,
)
from src.shared.python.motion_matching.shooting_schedule import sampled_shooting_windows
from src.shared.python.motion_matching.shooting_state_seed import select_shooting_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "candidate",
        "samples",
        "source_summary",
        "target",
        "output",
        "runtime",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = perf_counter()
    receipt = {"status": "running"}
    report_path = args.output / "receipt.json"

    def save() -> None:
        receipt["elapsed_s"] = perf_counter() - start
        report_path.write_text(json.dumps(receipt, indent=2) + "\n")

    try:
        raw = args.model.read_bytes()
        model_hash = hashlib.sha256(raw).hexdigest()
        assert (
            model_hash
            == "b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248"
        )
        spec = json.loads(raw)
        names = spec["coordinate_order"]
        candidate = NativeReplayCandidate.from_document(
            json.loads(args.candidate.read_text()), names, model_hash
        )
        assert (
            candidate.sha256
            == "786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a"
        )
        source = json.loads(args.source_summary.read_text())
        expected_samples_hash = source["output_hashes"][
            "evidence-replay/sampled-markers-state.npz"
        ]
        assert (
            hashlib.sha256(args.samples.read_bytes()).hexdigest()
            == expected_samples_hash
        )
        samples = np.load(args.samples)
        states = samples["native_state"]
        clock = samples["time_s"]
        assert states.shape == (307, 54) and np.isfinite(states).all()
        np.testing.assert_array_equal(samples["coordinate_names"], names)
        payload = json.loads(args.target.read_text())
        assert payload["source_sha256"] == candidate.document["capture_sha256"]
        target_clock = np.asarray(payload["time_s"])
        np.testing.assert_array_equal(clock, target_clock[target_clock <= 0.85])
        initial = np.asarray(candidate.document["q0"] + candidate.document["qd0"])
        np.testing.assert_array_equal(states[0], initial)
        windows = sampled_shooting_windows(clock, [0.6, 0.85])
        boundary = int(np.flatnonzero(clock == 0.6)[0])
        stored_node = select_shooting_states(
            {
                "model_sha256": model_hash,
                "times_s": clock,
                "coordinates": states[:, :27],
                "rates": states[:, 27:],
            },
            [0.6],
            model_sha256=model_hash,
            dimension=27,
        )[0.6]
        receipt.update(
            candidate_sha256=candidate.sha256,
            model_sha256=model_hash,
            input_hashes={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    args.model,
                    args.candidate,
                    args.samples,
                    args.source_summary,
                    args.target,
                    Path(__file__),
                )
            },
            source_hashes={
                str(p.relative_to(args.runtime)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in args.runtime.rglob("*.py")
            },
            boundary_index=boundary,
            window_sizes=[len(w) for w in windows],
            node_protocol="First window original q0/qd0; second window starts saved uninterrupted73 exact0.6 node. First endpoint minus saved node is measured independently. This is preflight, not final replay acceptance.",
            settings={
                "rtol": 1e-11,
                "atol": 1e-13,
                "max_step": 0.0000625,
                "closure_tolerance": 1e-7,
            },
            gates={
                "q_max_abs": 1e-6,
                "qd_max_abs": 1e-4,
                "marker_max_distance_m": 1e-7,
                "closure_max_abs": 1e-7,
            },
        )
        save()
        results = []
        next_state = initial
        for index, window in enumerate(windows):
            result = replay_window(
                raw, candidate, window, next_state, **receipt["settings"]
            )
            results.append(result)
            next_state = stored_node.copy()
            np.savez_compressed(
                args.output / f"window-{index}.npz",
                time_s=window,
                state=result.integration.state,
                markers_m=result.markers_m,
            )
            receipt.setdefault("windows", []).append(
                {
                    "index": index,
                    "time_start": float(window[0]),
                    "time_end": float(window[-1]),
                    "integration_s": result.integration.elapsed_s,
                    "evaluations": result.integration.evaluations,
                }
            )
            save()
        segmented = np.concatenate(
            (results[0].integration.state, results[1].integration.state[1:])
        )
        markers = np.concatenate((results[0].markers_m, results[1].markers_m[1:]))
        delta = segmented - states
        marker_difference = np.linalg.norm(markers - samples["returned_m"], axis=2)
        metrics = {
            "q_max_abs": float(np.max(abs(delta[:, :27]))),
            "qd_max_abs": float(np.max(abs(delta[:, 27:]))),
            "marker_max_distance_m": float(np.max(marker_difference)),
        }
        metrics["boundary_state_max_abs"] = float(
            np.max(abs(results[0].integration.state[-1] - stored_node))
        )
        metrics["terminal_q_max_abs"] = float(np.max(abs(delta[-1, :27])))
        metrics["terminal_qd_max_abs"] = float(np.max(abs(delta[-1, 27:])))
        metrics["terminal_marker_max_distance_m"] = float(np.max(marker_difference[-1]))
        receipt["segmented_comparison"] = metrics
        np.savez_compressed(
            args.output / "comparison.npz",
            time_s=clock,
            state=segmented,
            markers_m=markers,
            state_difference=delta,
            marker_distance_m=marker_difference,
        )
        engine = NativePinocchioModel(spec)

        def mapping(x: np.ndarray) -> dict[str, float]:
            return dict(zip(names, x.tolist(), strict=True))

        def closure(x: np.ndarray) -> np.ndarray:
            return np.concatenate(
                engine.closure_residuals(mapping(x[:27]), mapping(x[27:]))
            )

        def jacobian(x: np.ndarray) -> np.ndarray:
            linear = engine.closure_trajectory_linearization(
                mapping(x[:27]),
                mapping(x[27:]),
                dict.fromkeys(names, 0.0),
                finite_difference_step=1e-6,
            )
            return np.hstack((linear.dq[:12], linear.dv[:12]))

        reference = stored_node.copy()
        scales = np.r_[np.full(27, 0.1), np.ones(27)]
        residual_scales = np.r_[np.full(6, 0.01), np.full(6, 0.1)]
        matrix = jacobian(reference)
        basis = scaled_tangent_basis(matrix / residual_scales[:, None], scales)
        retracted = retract_node(
            reference,
            basis,
            np.zeros(basis.shape[1]),
            closure,
            jacobian,
            state_scales=scales,
            residual_scales=residual_scales,
            radius=0.5,
            tolerance=1e-8,
        )
        closure_rows = np.asarray([closure(x) for x in segmented])
        stored_closure = closure(stored_node)
        receipt["node_chart"] = {
            "dimension": basis.shape[1],
            "state_scales": scales.tolist(),
            "residual_scales": residual_scales.tolist(),
            "radius": 0.5,
            "tolerance": 1e-8,
            "jacobian_difference_step": 1e-6,
            "zero_retraction_max_abs_displacement": float(
                np.max(abs(retracted.state - reference))
            ),
            "zero_retraction_scaled_displacement": retracted.scaled_displacement,
            "retracted_scaled_closure_max_abs": retracted.closure_max_abs,
            "stored_node_closure_max_abs": float(np.max(abs(stored_closure))),
            "segmented_closure_max_abs": float(np.max(abs(closure_rows))),
            "scaled_singular_values": np.linalg.svd(
                matrix * scales[None, :] / residual_scales[:, None], compute_uv=False
            ).tolist(),
        }
        np.savez_compressed(
            args.output / "node-chart.npz",
            reference=reference,
            stored_node=stored_node,
            basis=basis,
            closure_jacobian=matrix,
            retracted_state=retracted.state,
            retraction_jacobian=retracted.state_jacobian,
            state_scales=scales,
            residual_scales=residual_scales,
            closure=closure_rows,
        )
        passed = (
            all(
                metrics[key] <= receipt["gates"][key]
                for key in ("q_max_abs", "qd_max_abs", "marker_max_distance_m")
            )
            and np.max(abs(closure_rows)) <= 1e-7
            and np.max(abs(stored_closure)) <= 1e-7
        )
        receipt["status"] = "passed" if passed else "failed_gates"
        save()
        if not passed:
            raise ValueError(
                "Two-window baseline failed unchanged trajectory/closure gates"
            )
    except (ValueError, RuntimeError, FloatingPointError, AssertionError) as error:
        receipt.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save()
        raise


if __name__ == "__main__":
    main()
