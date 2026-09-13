"""Compare trusted spawned native worker arrays against archived sequential inputs.

This is a fixed-input qualification only.  It neither executes the optimizer nor
changes a saved run.  All payloads are created locally from known archived
inputs; never use this script as a public pickle-deserialization service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_sensitivity_batch import (
    NativeSensitivityWindowRequest,
    evaluate_trusted_native_sensitivity_request,
    serialize_trusted_native_sensitivity_request,
)
from src.shared.python.motion_matching.native_window_executor import (
    NativeWindowExecutor,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(expected: np.ndarray, observed: np.ndarray, *, name: str) -> float:
    if expected.shape != observed.shape or not (
        np.isfinite(expected).all() and np.isfinite(observed).all()
    ):
        raise ValueError(f"{name} shape or finiteness differs")
    maximum = float(np.max(np.abs(expected - observed)))
    if not np.array_equal(expected, observed):
        raise ValueError(f"{name} differs: maximum {maximum}")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "run", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if any(
        __import__("os").environ.get(key) != "1"
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    ):
        raise ValueError("Single-thread BLAS/OMP is required")
    args.output.mkdir(exist_ok=False)
    report = json.loads((args.study / "report.json").read_text())
    arrays_path = args.study / "arrays.npz"
    if _sha(arrays_path) != report["arrays_sha256"]:
        raise ValueError("Archived sensitivity arrays identity differs")
    model = args.model.read_bytes()
    candidate = json.loads((args.run / "initial-candidate.json").read_text())
    arrays = np.load(arrays_path, allow_pickle=False)
    requests = []
    for index in range(6):
        requests.append(
            NativeSensitivityWindowRequest(
                model,
                candidate,
                arrays[f"input_{index}_clock"],
                arrays[f"input_{index}_state"],
                None if index == 0 else arrays[f"input_{index}_tangent"],
                0,
                0.8,
            )
        )
    payloads = tuple(map(serialize_trusted_native_sensitivity_request, requests))
    started = perf_counter()
    with NativeWindowExecutor(evaluate_trusted_native_sensitivity_request) as executor:
        sequential = executor.evaluate(payloads)
    sequential_s = perf_counter() - started
    started = perf_counter()
    with NativeWindowExecutor(
        evaluate_trusted_native_sensitivity_request, workers=2
    ) as executor:
        parallel = executor.evaluate(payloads)
    parallel_s = perf_counter() - started
    comparisons = []
    for index, (reference, worker) in enumerate(zip(sequential, parallel, strict=True)):
        comparisons.append(
            {
                key: _compare(getattr(reference, key), getattr(worker, key), name=key)
                for key in ("markers_m", "states", "marker_jacobian", "state_jacobian")
            }
        )
        for key, archived in {
            "markers_m": arrays[f"sequential_{index}_markers"],
            "states": arrays[f"sequential_{index}_states"],
            "marker_jacobian": arrays[f"sequential_{index}_marker_jacobian"],
            "state_jacobian": arrays[f"sequential_{index}_state_jacobian"],
        }.items():
            _compare(archived, getattr(worker, key), name=f"archived {index} {key}")
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "qualification": "Trusted native worker fixed-input equality only; no optimizer or solver assembly",
                "arrays_sha256": report["arrays_sha256"],
                "sequential_seconds": sequential_s,
                "two_worker_seconds_including_startup_ipc_shutdown": parallel_s,
                "comparisons": comparisons,
                "candidate_document_sha256": hashlib.sha256(
                    json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "source_hashes": {str(path): _sha(path) for path in (args.model, arrays_path, Path(__file__))},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
