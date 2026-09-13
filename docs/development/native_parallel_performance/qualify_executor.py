"""Qualify the new executor on the previous immutable native study inputs.

Requires the archived window_benchmark.py beside this script under its original
ControlTower name window_benchmark_10021.py. All pickle payloads are created here
from trusted local inputs; this is not a public deserialization service.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np

from window_benchmark_10021 import compare_arrays, evaluate


def evaluate_bytes(payload: bytes) -> dict:
    """Adapt trusted in-process task serialization to the unchanged evaluator."""
    return evaluate(pickle.loads(payload))  # noqa: S301 - locally created trusted tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executor", "study", "run", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    module_spec = importlib.util.spec_from_file_location(
        "executor_under_test", args.executor
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError("Cannot load executor under test")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    previous = json.loads((args.study / "report.json").read_text())
    array_path = args.study / "arrays.npz"
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != previous["arrays_sha256"]:
        raise ValueError("Prior arrays identity mismatch")
    raw = args.model.read_bytes()
    document = json.loads((args.run / "initial-candidate.json").read_text())
    candidate_hash = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # Candidate validation is performed by the original evaluator itself.
    arrays = np.load(array_path, allow_pickle=False)
    tasks = [
        pickle.dumps(
            {
                "model": raw,
                "candidate": document,
                "clock": arrays[f"input_{i}_clock"],
                "state": arrays[f"input_{i}_state"],
                "tangent": None if i == 0 else arrays[f"input_{i}_tangent"],
            },
            protocol=5,
        )
        for i in range(6)
    ]
    started = perf_counter()
    with module.NativeWindowExecutor(evaluate_bytes, workers=0) as executor:
        sequential = executor.evaluate(tasks)
    sequential_seconds = perf_counter() - started
    started = perf_counter()
    with module.NativeWindowExecutor(evaluate_bytes, workers=2) as executor:
        parallel = executor.evaluate(tasks)
    parallel_seconds = perf_counter() - started
    comparisons = [
        compare_arrays(a["arrays"], b["arrays"])
        for a, b in zip(sequential, parallel, strict=True)
    ]
    historical = [
        compare_arrays(
            {key: arrays[f"sequential_{i}_{key}"] for key in result["arrays"]},
            result["arrays"],
        )
        for i, result in enumerate(sequential)
    ]
    assembled = {}
    for key in ("markers", "states"):
        assembled[key] = compare_arrays(
            {
                key: np.concatenate(
                    [
                        result["arrays"][key][int(i > 0) :]
                        for i, result in enumerate(sequential)
                    ]
                )
            },
            {
                key: np.concatenate(
                    [
                        result["arrays"][key][int(i > 0) :]
                        for i, result in enumerate(parallel)
                    ]
                )
            },
        )
    report = {
        "qualification": "Native executor fixed-input equivalence only; no solver integration or fit acceptance",
        "candidate_document_sorted_sha256": candidate_hash,
        "source_candidate_sha256": previous["candidate_sha256"],
        "sequential_seconds": sequential_seconds,
        "two_worker_seconds_including_startup_ipc_shutdown": parallel_seconds,
        "comparisons": comparisons,
        "historical_reference_comparisons": historical,
        "concatenated_boundary_deduplicated_comparisons": assembled,
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.executor,
                Path(__file__),
                args.model,
                args.run / "initial-candidate.json",
                args.study / "report.json",
            )
        },
        "input_arrays_sha256": previous["arrays_sha256"],
        "sequential_rhs": [r["rhs_evaluations"] for r in sequential],
        "parallel_rhs": [r["rhs_evaluations"] for r in parallel],
        "notes": [
            "Every full marker/state Jacobian equals sequential and historical native results exactly.",
            "Concatenation checks ordering only; solver residual/defect assembly and caching remain unqualified.",
            "Repeated batches and cleanup are tested with deterministic unit functions; this native qualification runs one batch per mode.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
