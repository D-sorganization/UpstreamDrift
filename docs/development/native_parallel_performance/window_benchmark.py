"""One fixed six-window sequential/spawn benchmark; never an optimizer.

Run in the qualified runtime19 environment. Existing driver chart setup is
extracted verbatim, without running its CLI, replay, or optimization body.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform

from time import perf_counter
from typing import Any

import numpy as np


def compare_arrays(reference: dict, actual: dict) -> dict[str, float]:
    """Require bitwise-identical finite numerical outputs, not just close RMS."""
    if reference.keys() != actual.keys():
        raise ValueError("Returned arrays differ in keys")
    differences = {}
    for key, expected in reference.items():
        observed = actual[key]
        if expected.shape != observed.shape or not (
            np.isfinite(expected).all() and np.isfinite(observed).all()
        ):
            raise ValueError(f"Returned arrays differ or are nonfinite: {key}")
        differences[key] = float(np.max(np.abs(expected - observed)))
        if not np.array_equal(expected, observed):
            raise ValueError(f"Returned arrays differ: {key}")
    return differences


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(task: dict[str, Any]) -> dict[str, Any]:
    """Use the unchanged public sensitivity API in the calling process."""
    import resource

    started = perf_counter()
    from src.engines.physics_engines.pinocchio.python.native_sensitivity import (
        replay_marker_sensitivities,
    )
    from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate

    raw = task["model"]
    candidate = NativeReplayCandidate.from_document(
        task["candidate"],
        json.loads(raw)["coordinate_order"],
        hashlib.sha256(raw).hexdigest(),
    )
    result = replay_marker_sensitivities(
        raw,
        candidate,
        task["clock"],
        first_control=0,
        basis_duration_s=0.8,
        initial_state=task["state"],
        initial_sensitivity=task["tangent"],
    )
    arrays = {
        "markers": result.replay.markers_m,
        "states": result.replay.integration.state,
        "marker_jacobian": result.marker_jacobian,
        "state_jacobian": result.state_jacobian,
    }
    return {
        "arrays": arrays,
        "seconds": perf_counter() - started,
        "sensitivity_seconds": result.sensitivity_elapsed_s,
        "rhs_evaluations": result.sensitivity_evaluations,
        "pid": os.getpid(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "model", "target", "driver", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if any(
        os.environ.get(key) != "1"
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    ):
        raise ValueError("Single-thread BLAS/OMP is required")
    args.output.mkdir(exist_ok=False)
    setup_start = perf_counter()
    from src.engines.physics_engines.pinocchio.python.native_model import (
        NativePinocchioModel,
    )
    from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
    from src.shared.python.motion_matching.node_retraction import retract_node
    from src.shared.python.motion_matching.shooting_schedule import (
        sampled_shooting_windows,
    )
    from src.engines.physics_engines.pinocchio.python import native_sensitivity

    config_path = args.run / "config.json"
    snapshot_path = args.run / "evaluation-00001.json"
    candidate_path = args.run / "initial-candidate.json"
    config, snapshot, document = (
        json.loads(p.read_text()) for p in (config_path, snapshot_path, candidate_path)
    )
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    candidate = NativeReplayCandidate.from_document(document, names, sha(args.model))
    if (
        candidate.sha256 != snapshot["candidate_sha256"]
        or candidate.sha256 != config["seed"]
    ):
        raise ValueError("Initial candidate/snapshot identity mismatch")
    if config["model_sha256"] != sha(args.model) or snapshot["config_sha256"] != sha(
        config_path
    ):
        raise ValueError("Model/config identity mismatch")
    if sha(args.driver) != config["runner_sha256"]:
        raise ValueError("Original driver changed")
    payload = json.loads(args.target.read_text())
    if payload["source_sha256"] != document["capture_sha256"]:
        raise ValueError("Capture identity mismatch")
    if (
        config["nodes"] != [0.2, 0.4, 0.6, 0.7, 0.8, 0.85]
        or config["basis_duration_s"] != 0.8
    ):
        raise ValueError("Unexpected benchmark clock/basis")
    windows = sampled_shooting_windows(np.array(payload["time_s"]), config["nodes"])
    nodes = {
        float(key): np.array(value) for key, value in snapshot["physical_nodes"].items()
    }
    # Reuse precisely the original driver's chart construction statements only.
    source = args.driver.read_text()
    chart_source = source[
        source.index("engine=NativePinocchioModel(spec);") : source.index(
            "node_cache={}"
        )
    ]
    environment = {
        "np": np,
        "NativePinocchioModel": NativePinocchioModel,
        "spec": spec,
        "names": names,
        "n": len(names),
        "nodes": nodes,
    }
    exec(compile(ast.parse(chart_source), str(args.driver), "exec"), environment)
    tasks = []
    for clock in windows:
        start = float(clock[0])
        if start == 0:
            state, tangent = np.array(document["q0"] + document["qd0"]), None
        else:
            retracted = retract_node(
                nodes[start],
                environment["bases"][start],
                np.zeros(42),
                environment["closure"],
                environment["cj"],
                state_scales=environment["d"],
                residual_scales=environment["cs"],
                radius=0.5,
                tolerance=1e-8,
            )
            state, tangent = retracted.state, retracted.state_jacobian
            if np.max(np.abs(state - nodes[start])) > 1e-10:
                raise ValueError("Zero chart retraction changes saved node")
        tasks.append(
            {
                "model": raw,
                "candidate": document,
                "clock": clock,
                "state": state,
                "tangent": tangent,
            }
        )
    setup_s = perf_counter() - setup_start
    started = perf_counter()
    sequential = [evaluate(task) for task in tasks]
    sequential_s = perf_counter() - started
    started = perf_counter()
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        parallel = list(pool.map(evaluate, tasks))
    parallel_s = perf_counter() - started
    comparisons = [
        compare_arrays(a["arrays"], b["arrays"])
        for a, b in zip(sequential, parallel, strict=True)
    ]
    saved = {}
    for mode, results in (("sequential", sequential), ("parallel", parallel)):
        for index, result in enumerate(results):
            for key, array in result.pop("arrays").items():
                saved[f"{mode}_{index}_{key}"] = array
    for index, task in enumerate(tasks):
        for key in ("clock", "state", "tangent"):
            if task[key] is not None:
                saved[f"input_{index}_{key}"] = task[key]
    np.savez_compressed(args.output / "arrays.npz", **saved)
    report = {
        "qualification": "Fixed-input performance diagnostic only; no optimizer or production integration",
        "sequential_seconds": sequential_s,
        "two_worker_seconds_including_startup_ipc_shutdown": parallel_s,
        "observed_speedup": sequential_s / parallel_s,
        "shared_input_setup_seconds": setup_s,
        "sequential": sequential,
        "parallel": parallel,
        "comparisons": comparisons,
        "exact_array_equality": True,
        "candidate_sha256": candidate.sha256,
        "source_hashes": {
            str(p): sha(p)
            for p in (
                config_path,
                snapshot_path,
                candidate_path,
                args.model,
                args.target,
                args.driver,
                Path(__file__),
                Path(native_sensitivity.__file__),
            )
        },
        "arrays_sha256": sha(args.output / "arrays.npz"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "notes": [
            "One sequential pass then one two-worker spawn pass; no timing distribution or cache-order control.",
            "Parallel wall includes input/output serialization, worker imports, startup and shutdown.",
            "RSS is per-process high-water mark in KiB on Linux; not simultaneous aggregate memory.",
            "All marker samples, all primal states, and full marker/state Jacobians compared exactly.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
