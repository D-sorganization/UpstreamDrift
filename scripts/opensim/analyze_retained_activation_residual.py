"""Offline activation-law comparison for a retained OpenSim Moco experiment.

This diagnostic reads saved text artifacts only. It never imports OpenSim or
changes the retained native experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.integrate import solve_ivp


OPEN_SIM_ACTIVATION_SOURCE = (
    "https://raw.githubusercontent.com/opensim-org/opensim-core/4.6/"
    "OpenSim/Actuators/MuscleFirstOrderActivationDynamicModel.cpp"
)
OPEN_SIM_MOCO_SOLVER_SOURCE = (
    "https://raw.githubusercontent.com/opensim-org/opensim-core/4.6/"
    "OpenSim/Moco/MocoDirectCollocationSolver.cpp"
)
OPEN_SIM_MOCO_HERMITE_SIMPSON_DOCS = (
    "https://opensim-org.github.io/opensim-moco-site/docs/1.1.0/html_user/"
    "classOpenSim_1_1MocoDirectCollocationSolver.html"
)


def sha256_file(path: Path) -> str:
    """Return the content digest for an evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sto(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a simple whitespace-delimited OpenSim storage table."""
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("endheader")
    names = lines[end + 1].split()
    rows = [
        [float(value) for value in line.split()]
        for line in lines[end + 2 :]
        if line.strip()
    ]
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(f"invalid storage table shape in {path}")
    return names, matrix


def activation_derivative(
    activation: float,
    excitation: float,
    *,
    tau_activation: float,
    tau_deactivation: float,
    minimum_activation: float,
) -> float:
    """Evaluate the OpenSim 4.6 first-order activation law."""
    if not all(math.isfinite(value) for value in (activation, excitation)):
        raise ValueError("activation and excitation must be finite")
    if not all(
        math.isfinite(value)
        for value in (tau_activation, tau_deactivation, minimum_activation)
    ):
        raise ValueError("activation-law parameters must be finite")
    if tau_activation <= 0 or tau_deactivation <= 0 or not 0 < minimum_activation <= 1:
        raise ValueError("activation-law parameters are outside their domain")
    bounded = min(1.0, max(minimum_activation, activation))
    scale = 0.5 + 1.5 * bounded
    tau = tau_activation * scale if excitation > bounded else tau_deactivation / scale
    return (excitation - bounded) / tau


def _linear_excitation(t: float, t0: float, t1: float, e0: float, e1: float) -> float:
    fraction = (t - t0) / (t1 - t0)
    return e0 + fraction * (e1 - e0)


def _integrate_channel(
    times: np.ndarray,
    excitations: np.ndarray,
    initial: float,
    parameters: dict[str, float],
    *,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Integrate independently on each exact piecewise-linear input interval."""
    result = np.empty_like(times)
    result[0] = initial
    current = initial
    for index, (t0, t1) in enumerate(zip(times[:-1], times[1:], strict=True)):
        e0, e1 = excitations[index : index + 2]

        def rhs(
            t: float,
            state: np.ndarray,
            *,
            left_time: float = float(t0),
            right_time: float = float(t1),
            left_input: float = float(e0),
            right_input: float = float(e1),
        ) -> tuple[float]:
            excitation = _linear_excitation(
                t, left_time, right_time, left_input, right_input
            )
            return (activation_derivative(float(state[0]), excitation, **parameters),)

        solved = solve_ivp(
            rhs,
            (float(t0), float(t1)),
            (current,),
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=float(t1 - t0) / 16,
        )
        if not solved.success or not np.all(np.isfinite(solved.y[:, -1])):
            raise RuntimeError(
                f"segmented scalar integration failed at interval {index}"
            )
        current = float(solved.y[0, -1])
        result[index + 1] = current
    return result


def _hermite_simpson_defects(
    times: np.ndarray,
    activations: np.ndarray,
    excitations: np.ndarray,
    parameters: dict[str, float],
) -> dict[str, np.ndarray]:
    """Evaluate Hermite-Simpson defects on endpoint/midpoint triples."""
    if (
        len(times) % 2 != 1
        or len(activations) != len(times)
        or len(excitations) != len(times)
    ):
        raise ValueError("Hermite-Simpson output requires endpoint/midpoint triples")
    cell_count = (len(times) - 1) // 2
    midpoint_defects = np.empty(cell_count, dtype=np.float64)
    simpson_defects = np.empty(cell_count, dtype=np.float64)
    for cell in range(cell_count):
        left, midpoint, right = 2 * cell, 2 * cell + 1, 2 * cell + 2
        h = float(times[right] - times[left])
        a_left, a_mid, a_right = activations[[left, midpoint, right]]
        e_left, e_mid, e_right = excitations[[left, midpoint, right]]
        f_left = activation_derivative(float(a_left), float(e_left), **parameters)
        f_mid = activation_derivative(float(a_mid), float(e_mid), **parameters)
        f_right = activation_derivative(float(a_right), float(e_right), **parameters)
        hermite_mid = 0.5 * (a_left + a_right) + h * (f_left - f_right) / 8.0
        midpoint_defects[cell] = a_mid - hermite_mid
        simpson_defects[cell] = (
            a_right - a_left - h * (f_left + 4.0 * f_mid + f_right) / 6.0
        )
    return {"midpoint_state": midpoint_defects, "simpson_dynamics": simpson_defects}


def _muscle_defaults(source_path: Path) -> dict[str, dict[str, Any]]:
    """Read muscle XML overrides and identify properties inherited from 4.6."""
    root = ET.parse(source_path).getroot()
    results: dict[str, dict[str, Any]] = {}
    for element in root.iter("Millard2012EquilibriumMuscle"):
        name = element.attrib["name"]
        properties = {child.tag: child.text for child in element}
        defaults = {
            "activation_time_constant": 0.01,
            "deactivation_time_constant": 0.04,
            "minimum_activation": 0.01,
            "ignore_activation_dynamics": "false",
        }
        results[name] = {
            "source_overrides": {
                key: properties[key] for key in defaults if key in properties
            },
            "effective_from_xml_or_4_6_defaults": {
                key: properties.get(key, value) for key, value in defaults.items()
            },
            "omitted_properties_use_tag_defaults": sorted(
                key for key in defaults if key not in properties
            ),
        }
    return results


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "max_abs": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "signed_sum": float(np.sum(values)),
    }


@dataclass(frozen=True)
class RetainedData:
    """Validated immutable inputs for the offline activation comparison."""

    files: dict[str, Path]
    request: dict[str, Any]
    audit: dict[str, Any]
    times: np.ndarray
    state_names: list[str]
    state_matrix: np.ndarray
    control_names: list[str]
    control_matrix: np.ndarray
    replay_state_names: list[str]
    replay_states: np.ndarray
    muscle_defaults: dict[str, dict[str, Any]]


def _load_retained_data(experiment: Path, replay_audit: Path) -> RetainedData:
    """Load files and validate their source, request, state and control bindings."""
    source, request = experiment / "source.osim", experiment / "request.json"
    solution, controls = (
        experiment / "solution.sto",
        experiment / "time_only_controls.sto",
    )
    request_data = json.loads(request.read_text(encoding="utf-8"))
    audit_data = json.loads(replay_audit.read_text(encoding="utf-8"))
    state_names, state_matrix = read_sto(solution)
    control_names, control_matrix = read_sto(controls)
    times = state_matrix[:, 0]
    control_times = control_matrix[:, 0]
    if not np.array_equal(times, control_times):
        raise ValueError("saved solution and controls must share exact knot times")
    replay_times = np.asarray(audit_data["times_seconds"], dtype=np.float64)
    replay_states = np.asarray(audit_data["replayed_state_matrix"], dtype=np.float64)
    replay_state_names = audit_data["state_names"]
    solution_replay_order = [state_names.index(name) for name in replay_state_names]
    solution_replay_states = state_matrix[:, solution_replay_order]
    if (
        not np.array_equal(times, replay_times)
        or replay_states.shape != solution_replay_states.shape
    ):
        raise ValueError("retained replay and solution state grids/shapes differ")
    if request_data["source_sha256"] != sha256_file(source):
        raise ValueError("request source digest does not match retained model")
    if audit_data["source_sha256"] != sha256_file(source):
        raise ValueError("replay audit source digest does not match retained model")
    expected_audit_bindings = {
        "request_sha256": sha256_file(request),
        "solution_sha256": sha256_file(solution),
        "saved_controls_sha256": sha256_file(controls),
    }
    for key, digest in expected_audit_bindings.items():
        if audit_data[key] != digest:
            raise ValueError(f"replay audit {key} does not match retained input")
    if not audit_data["applied_controls_equal_saved_samples_exactly"]:
        raise ValueError("replay audit did not confirm exact saved-control samples")
    if any(name not in state_names for name in replay_state_names):
        raise ValueError("replay state column is absent from saved solution")
    return RetainedData(
        files={
            "source_osim": source,
            "request": request,
            "solution_sto": solution,
            "time_only_controls_sto": controls,
            "replay_audit": replay_audit,
        },
        request=request_data,
        audit=audit_data,
        times=times,
        state_names=state_names,
        state_matrix=state_matrix,
        control_names=control_names,
        control_matrix=control_matrix,
        replay_state_names=replay_state_names,
        replay_states=replay_states,
        muscle_defaults=_muscle_defaults(source),
    )


def _analyze_muscle(
    data: RetainedData, muscle: str, defaults: dict[str, Any]
) -> dict[str, Any]:
    """Compare one activation state with its collocation, replay and scalar traces."""
    activation_name, control_name = (
        f"/forceset/{muscle}/activation",
        f"/forceset/{muscle}",
    )
    if (
        activation_name not in data.state_names
        or activation_name not in data.replay_state_names
        or control_name not in data.control_names
        or control_name not in data.state_names
    ):
        raise ValueError(f"required activation/control channel is missing for {muscle}")
    effective = defaults["effective_from_xml_or_4_6_defaults"]
    parameters = {
        "tau_activation": float(effective["activation_time_constant"]),
        "tau_deactivation": float(effective["deactivation_time_constant"]),
        "minimum_activation": float(effective["minimum_activation"]),
    }
    solution = data.state_matrix[:, data.state_names.index(activation_name)]
    replay = data.replay_states[:, data.replay_state_names.index(activation_name)]
    excitation = data.control_matrix[:, data.control_names.index(control_name)]
    optimized_control = data.state_matrix[:, data.state_names.index(control_name)]
    if not np.array_equal(optimized_control, excitation):
        raise ValueError(f"saved PWL excitation changed optimized control {muscle}")
    return _activation_channel_report(
        data.times, solution, replay, excitation, parameters, defaults
    )


def _activation_channel_report(
    times: np.ndarray,
    solution: np.ndarray,
    replay: np.ndarray,
    excitation: np.ndarray,
    parameters: dict[str, float],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Build the numerical comparisons and diagnostics for one muscle channel."""
    segmented = _integrate_channel(
        times, excitation, float(solution[0]), parameters, rtol=1e-12, atol=1e-14
    )
    tighter = _integrate_channel(
        times, excitation, float(solution[0]), parameters, rtol=1e-13, atol=1e-15
    )
    defects = _hermite_simpson_defects(times, solution, excitation, parameters)
    interval_index = int(np.argmax(np.abs(defects["simpson_dynamics"])))
    solution_scalar_delta = solution - segmented
    maximum_index = int(np.argmax(np.abs(solution_scalar_delta)))
    midpoint_delta = excitation[1:-1:2] - 0.5 * (excitation[:-2:2] + excitation[2::2])
    return {
        "activation_parameters": parameters,
        "xml_default_provenance": defaults,
        "solution_activation_vs_replay": _summary(solution - replay),
        "solution_activation_vs_segmented_scalar": _summary(solution_scalar_delta),
        "replay_activation_vs_segmented_scalar": _summary(replay - segmented),
        "max_solution_vs_scalar_error_time_seconds": float(times[maximum_index]),
        "solution_vs_scalar_initial_and_final": [
            float(solution_scalar_delta[0]),
            float(solution_scalar_delta[-1]),
        ],
        "scalar_integration_sensitivity_rtol_1e-12_vs_1e-13": _summary(
            segmented - tighter
        ),
        "hermite_simpson_midpoint_state_defect": _summary(defects["midpoint_state"]),
        "hermite_simpson_simpson_dynamics_defect": _summary(
            defects["simpson_dynamics"]
        ),
        "max_abs_simpson_defect_cell_index": interval_index,
        "max_abs_simpson_defect_cell_seconds": [
            float(times[2 * interval_index]),
            float(times[2 * interval_index + 2]),
        ],
        "activation_min_max": [float(np.min(solution)), float(np.max(solution))],
        "excitation_min_max": [float(np.min(excitation)), float(np.max(excitation))],
        "max_abs_control_midpoint_minus_endpoint_linear_interpolation": float(
            np.max(np.abs(midpoint_delta))
        ),
        "segmented_scalar_activation_at_knots": segmented.tolist(),
    }


def _report_document(data: RetainedData, channels: dict[str, Any]) -> dict[str, Any]:
    """Assemble the machine-readable, source-bound offline analysis result."""
    times = data.times
    request, audit = data.request, data.audit
    return {
        "schema_version": "retained-activation-residual-v1",
        "status": "offline_diagnostic_only",
        "source_law_reference": OPEN_SIM_ACTIVATION_SOURCE,
        "source_law_tag": "OpenSim 4.6",
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "analysis_runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
        "moco_transcription_references": [
            OPEN_SIM_MOCO_SOLVER_SOURCE,
            OPEN_SIM_MOCO_HERMITE_SIMPSON_DOCS,
        ],
        "method": _analysis_method(request, times),
        "input_sha256": {key: sha256_file(path) for key, path in data.files.items()},
        "replay_audit_declared_input_sha256": audit["input_sha256"],
        "source_properties": data.muscle_defaults,
        "sample_count": int(len(times)),
        "interval_count": int(len(times) - 1),
        "horizon_seconds": [float(times[0]), float(times[-1])],
        "replay_status": audit["status"],
        "replay_applied_controls_equal_saved_samples_exactly": audit[
            "applied_controls_equal_saved_samples_exactly"
        ],
        "channels": channels,
    }


def _analysis_method(request: dict[str, Any], times: np.ndarray) -> dict[str, Any]:
    """Describe the exact discretization, integration method and limitations."""
    return {
        "moco_transcription": request["transcription_scheme"],
        "control_interpolation": request["replay_control_interpolation"],
        "segmented_integrator": "SciPy solve_ivp DOP853, each saved PWL interval separately",
        "rtol_atol": [[1e-12, 1e-14], [1e-13, 1e-15]],
        "mesh_interval_seconds": float(
            (times[-1] - times[0]) / request["mesh_intervals"]
        ),
        "mesh_interval_row_pattern": "41 trajectory rows for 20 Hermite-Simpson cells; each cell uses endpoint, midpoint, endpoint rows",
        "hermite_simpson_equations": {
            "midpoint": "x_mid = (x_left + x_right)/2 + h*(f_left - f_right)/8",
            "dynamics": "x_right - x_left = h*(f_left + 4*f_mid + f_right)/6",
        },
        "control_midpoint_policy": "request sets interpolate_control_mesh_interior_points=true; retained controls include the midpoint samples",
        "numerical_sensitivity_limitation": "difference between two tolerances is a sensitivity estimate, not a rigorous error bound",
        "runtime_limitation": "activation defaults come from OpenSim 4.6 source because omitted from XML; installed binary was not queried",
    }


def analyze(experiment: Path, replay_audit: Path, output: Path) -> dict[str, Any]:
    """Compute source-bound activation comparisons using retained files."""
    data = _load_retained_data(experiment, replay_audit)
    channels = {
        muscle: _analyze_muscle(data, muscle, defaults)
        for muscle, defaults in data.muscle_defaults.items()
    }
    result = _report_document(data, channels)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.experiment, args.replay_audit, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "channels": report["channels"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
