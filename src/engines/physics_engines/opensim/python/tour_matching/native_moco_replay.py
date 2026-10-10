"""Exact-knot Moco export, observation-free T01 replay, then physical scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.engines.native_replay_contracts import native_replay_contract_types

from .native_marker_geometry import NativeMarkerGeometry
from .native_moco_runner import (
    NativeMocoPreparation,
    NativeMocoRequest,
    NativeMocoSolve,
    _sha,
    _require_current_inputs,
    _solution_names,
)
from .native_muscle_bundle import (
    build_native_muscle_replay_bundle,
    replay_native_muscle_bundle,
)
from .native_prepared_state import DeclaredColdStart
from .registration import register_points
from .trc import read_trc

if TYPE_CHECKING:
    from .muscle_replay import NativeMuscleReplayResult
    from .native_constrained_muscle import ConstrainedMuscleReplay


@dataclass(frozen=True)
class NativeMocoExport:
    """The original control knots remain a strict subset of output times."""

    original_knot_count: int
    output_count: int
    original_knots_sha256: str
    output_clock_sha256: str
    bundle_sha256: str
    output_times: NDArray[np.float64]
    wall_seconds: float
    cpu_seconds: float


@dataclass(frozen=True)
class NativeMocoScore:
    """Executed marker/state error; optimization success is not qualification."""

    marker_rmse_m: float
    observed_samples: int
    full_state_max_abs: float
    optimized_objective: float
    source_sha256: str
    capture_sha256: str
    replay_bundle_sha256: str
    observation_clock_sha256: str
    scoring_provider_sha256: str
    wall_seconds: float
    cpu_seconds: float
    qualification: str = "synthetic-or-exploratory-executed-match-only"


def _solution(
    request: NativeMocoRequest, solved: NativeMocoSolve, directory: Path
) -> Any:
    import opensim as osim

    if not solved.success or solved.solution_sha256 is None:
        raise ValueError("Successful, saved native solution is required")
    path = directory / "solution.sto"
    if _sha(path) != solved.solution_sha256:
        raise ValueError("Saved native solution changed after solve")
    if _sha(request.model_path) != request.model_sha256:
        raise ValueError("Native source changed after solve")
    solution = osim.MocoTrajectory(str(path))
    if set(_solution_names(solution.getStateNames())) != set(
        request.bindings.state_bounds
    ):
        raise ValueError("Saved solution state names differ from prepared model")
    if set(_solution_names(solution.getControlNames())) != set(
        request.bindings.control_bounds
    ):
        raise ValueError("Saved solution controls differ from prepared model")
    return solution


def _exact_output_grid(
    solution: Any, prepared: NativeMocoPreparation, directory: Path
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    original = np.asarray(solution.getTimeMat(), dtype=float).reshape(-1)
    clock = np.asarray(
        json.loads((directory / "observation_clock.json").read_text(encoding="utf-8")),
        dtype=float,
    )
    if (
        original.size < 2
        or not np.isfinite(original).all()
        or not np.all(np.diff(original) > 0)
        or clock.size < 2
        or not np.isfinite(clock).all()
        or not np.all(np.diff(clock) > 0)
        or hashlib.sha256(clock.tobytes()).hexdigest()
        != prepared.observation_clock_sha256
        or clock[0] != original[0]
        or clock[-1] != original[-1]
    ):
        raise ValueError(
            "Frozen observation and exact solution clocks are incompatible"
        )
    output = np.union1d(original, clock)
    if not np.array_equal(output[np.searchsorted(output, original)], original):
        raise ValueError("An original native control knot was lost")
    if not np.array_equal(output[np.searchsorted(output, clock)], clock):
        raise ValueError("An original observation time was lost")
    output.setflags(write=False)
    original.setflags(write=False)
    return original, output


def _native_control_history(
    request: NativeMocoRequest,
    solution: Any,
    original: NDArray[np.float64],
    output: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    import opensim as osim

    model = osim.Model(str(request.model_path))
    muscles = model.getMuscles()
    channels = {
        muscles.get(i).getAbsolutePathString(): muscles.get(i).getName()
        for i in range(muscles.getSize())
    }
    if set(channels) != set(request.bindings.control_bounds):
        raise ValueError("Saved controls are not exactly all native muscles")
    history: dict[str, NDArray[np.float64]] = {}
    for path, name in channels.items():
        knots = np.asarray(solution.getControlMat(path), dtype=float).reshape(-1)
        if knots.shape != original.shape or not np.isfinite(knots).all():
            raise ValueError("Native control samples do not cover every knot")
        lower, upper = request.bindings.control_bounds[path]
        if np.any(knots < lower) or np.any(knots > upper):
            raise ValueError("Native control samples exceed declared bounds")
        values = np.interp(output, original, knots)
        if not np.array_equal(values[np.searchsorted(output, original)], knots):
            raise ValueError("Dense input schedule changed an exact native knot")
        history[name] = values
    return history


def export_native_moco_bundle(
    request: NativeMocoRequest,
    prepared: NativeMocoPreparation,
    solved: NativeMocoSolve,
    output_dir: Path,
) -> NativeMocoExport:
    """Save an immutable T01 bundle with all control and observation clocks."""
    directory = Path(output_dir)
    wall = time.perf_counter()
    cpu = time.process_time()
    if not prepared.ready_for_software_solve:
        raise ValueError("Preparation blockers prohibit native export")
    _require_current_inputs(request, prepared, directory)
    solution = _solution(request, solved, directory)
    original, output = _exact_output_grid(solution, prepared, directory)
    controls = _native_control_history(request, solution, original, output)
    if request.constrained_cold_start is None:
        bundle = build_native_muscle_replay_bundle(
            request.model_path,
            request.bindings.initial_state,
            output,
            controls,
            experiment_id="offline-native-moco-independent-replay",
        )
    else:
        from .native_constrained_muscle import build_constrained_muscle_bundle

        bundle = build_constrained_muscle_bundle(
            request.constrained_cold_start,
            output,
            controls,
            experiment_id="offline-native-moco-constrained-independent-replay",
        )
    contracts = native_replay_contract_types()
    payload = contracts.dumps_experiment_replay_bundle(bundle)
    reloaded = contracts.load_experiment_replay_bundle(payload)
    if reloaded != bundle:
        raise ValueError("Serialized native replay input differs after reload")
    bundle_path = directory / "native_replay_bundle.json"
    bundle_path.write_text(payload, encoding="utf-8")
    receipt = NativeMocoExport(
        int(original.size),
        int(output.size),
        hashlib.sha256(original.tobytes()).hexdigest(),
        hashlib.sha256(output.tobytes()).hexdigest(),
        _sha(bundle_path),
        output,
        time.perf_counter() - wall,
        time.process_time() - cpu,
    )
    (directory / "export.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in asdict(receipt).items()
                if key != "output_times"
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def replay_native_moco_bundle(
    exported: NativeMocoExport,
    model_path: Path,
    output_dir: Path,
    *,
    constrained_cold_start: DeclaredColdStart | None = None,
) -> Any:
    """Execute the saved T01 inputs on a fresh model, without a reference API."""
    directory = Path(output_dir)
    bundle_path = directory / "native_replay_bundle.json"
    if _sha(bundle_path) != exported.bundle_sha256:
        raise ValueError("Saved T01 input changed before independent replay")
    contracts = native_replay_contract_types()
    bundle = contracts.load_experiment_replay_bundle(
        bundle_path.read_text(encoding="utf-8")
    )
    if _sha(Path(model_path)) != bundle.model.source_model_sha256:
        raise ValueError("Native model changed before independent replay")
    wall = time.perf_counter()
    cpu = time.process_time()
    result: NativeMuscleReplayResult | ConstrainedMuscleReplay
    if constrained_cold_start is None:
        if bundle.model.variant_id != "reviewed-cold-start-muscles":
            raise ValueError("constrained replay requires its declared cold start")
        result = replay_native_muscle_bundle(bundle, model_path)
    else:
        from .native_constrained_muscle import replay_constrained_muscle_bundle

        if constrained_cold_start.model_path != Path(model_path).resolve():
            raise ValueError("constrained replay model path differs")
        result = replay_constrained_muscle_bundle(bundle, constrained_cold_start)
    if not np.array_equal(result.times, exported.output_times):
        raise ValueError("Native execution did not retain the frozen output clock")
    evidence = directory / "native_replay.npz"
    np.savez(
        evidence,
        time_seconds=result.times,
        state_names=np.asarray(result.state_names),
        states=result.states,
        muscle_forces_n=result.muscle_forces_n,
        applied_excitations=result.applied_excitations,
    )
    (directory / "replay.json").write_text(
        json.dumps(
            {
                "bundle_sha256": exported.bundle_sha256,
                "native_output_sha256": _sha(evidence),
                "state_count": len(result.state_names),
                "sample_count": len(result.times),
                "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu,
                "qualification": "independent-native-input-reproduction-only",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _native_marker_points(
    request: NativeMocoRequest,
    native: Any,
    indices: NDArray[np.intp],
    labels: tuple[str, ...],
) -> tuple[NDArray[np.float64], str]:
    if request.constrained_cold_start is not None:
        from .native_constrained_muscle import observe_constrained_markers

        return observe_constrained_markers(
            request.constrained_cold_start,
            native,
            {label: request.marker_bindings[label] for label in labels},
            indices,
        )
    coordinate_names = tuple(
        name.removesuffix("/value")
        for name in native.state_names
        if name.endswith("/value")
    )
    state_columns = [
        native.state_names.index(name + "/value") for name in coordinate_names
    ]
    geometry = NativeMarkerGeometry(request.model_path, coordinate_names)
    bindings = {label: request.marker_bindings[label] for label in labels}
    points = np.asarray(
        [
            geometry.marker_positions(native.states[index, state_columns], bindings)
            for index in indices
        ],
        dtype=float,
    )
    return points, geometry.identity_sha256


def score_native_moco_replay(
    request: NativeMocoRequest,
    prepared: NativeMocoPreparation,
    solved: NativeMocoSolve,
    exported: NativeMocoExport,
    native: Any,
    output_dir: Path,
) -> NativeMocoScore:
    """Score executed positions on original masked times, separately from Moco cost."""
    directory = Path(output_dir)
    wall = time.perf_counter()
    cpu = time.process_time()
    if _sha(request.trc_path) != prepared.capture_sha256:
        raise ValueError("Original observation changed before physical scoring")
    if request.registration is None:
        raise ValueError("Explicit capture registration is required for scoring")
    capture = read_trc(request.trc_path)
    if hashlib.sha256(capture.time_s.tobytes()).hexdigest() != (
        prepared.observation_clock_sha256
    ):
        raise ValueError("Original observation clock differs from preparation")
    indices = np.searchsorted(native.times, capture.time_s)
    if np.any(indices >= len(native.times)) or not np.array_equal(
        native.times[indices], capture.time_s
    ):
        raise ValueError("Independent replay omitted an exact observation time")
    points, geometry_sha = _native_marker_points(
        request, native, indices, capture.labels
    )
    observed = register_points(capture.points_m, request.registration)
    weights = np.asarray([request.marker_weights[label] for label in capture.labels])
    active = weights[None, :] * capture.valid
    if not np.isfinite(points).all() or not np.any(active):
        raise ValueError("Native marker support is absent or nonfinite")
    error = points - observed
    squared = np.sum(error[capture.valid] ** 2, axis=1)
    rmse = float(np.sqrt(np.sum(active[capture.valid] * squared) / np.sum(active)))
    solution = _solution(request, solved, directory)
    knot_times = np.asarray(solution.getTimeMat(), dtype=float).reshape(-1)
    knot_indices = np.searchsorted(native.times, knot_times)
    if not np.array_equal(native.times[knot_indices], knot_times):
        raise ValueError("Independent replay omitted an optimized state knot")
    state_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(solution.getStateMat(name)).reshape(-1)
                    - native.states[knot_indices, native.state_names.index(name)]
                )
            )
        )
        for name in native.state_names
    )
    objective = solved.objective
    if objective is None or not np.isfinite(objective):
        raise ValueError("Saved native solution has no finite objective")
    result = NativeMocoScore(
        rmse,
        int(capture.valid.sum()),
        state_error,
        float(objective),
        request.model_sha256,
        request.trc_sha256,
        exported.bundle_sha256,
        prepared.observation_clock_sha256,
        geometry_sha,
        time.perf_counter() - wall,
        time.process_time() - cpu,
    )
    (directory / "score.json").write_text(
        json.dumps(asdict(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "NativeMocoExport",
    "NativeMocoScore",
    "export_native_moco_bundle",
    "replay_native_moco_bundle",
    "score_native_moco_replay",
]
