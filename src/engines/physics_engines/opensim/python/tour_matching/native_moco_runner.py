"""File-driven native Moco preparation and independent muscle replay.

This runner joins existing providers without granting scientific qualification.
Synthetic fixtures can prove the software path; real-source blockers remain
visible before an optimizer is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from defusedxml import ElementTree as SafeET

from .moco_initial_bindings import MocoInitialBindings
from .moco_tracking import MocoTrackingConfig, build_moco_study
from .native_moco_request import NativeMocoRequest
from .native_capability_probe import (
    NativeProbeFailure,
    _native_runtime_identity,
    _native_short_capability_probe,
)
from .native_passive_readiness import (
    PassiveReadinessPolicy,
    audit_native_passive_readiness,
)
from .native_passive_receipt import validate_native_passive_observation_receipt
from .native_reference_conventions import (
    ReferenceStateDeclaration,
    audit_source_reference,
)
from .registration import CaptureRegistration, register_points
from .native_prepared_state import DeclaredColdStart, reconstruct_declared_cold_start
from .native_mixed_actuation import MixedActuationProfile
from .native_body_inertia import NativeBodyInertiaAudit, audit_native_body_inertia
from .native_diagnostic_contract import (
    DiagnosticCode,
    NativeDiagnosticAssessment,
    NativeDiagnosticFinding,
    NativeDiagnosticPolicy,
    NativeDiagnosticRuntimeEvidence,
    NativeQualificationContext,
    RuntimePhase,
    RuntimeViability,
    diagnostic_effects,
)
from .native_wrapped_path import (
    WrappedPathDeclaration,
    audit_native_wrapped_path_source,
)
from .trc import read_trc, write_trc
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.estimation import dime_global_calibration
from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_bytes


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


from . import native_moco_preparation as _native_moco_preparation

NativeMocoPreparation = _native_moco_preparation.NativeMocoPreparation
_physical_audit_bytes = _native_moco_preparation._physical_audit_bytes


def prepare_native_moco(
    request: NativeMocoRequest,
    output_dir: Path,
    *,
    run_capability_probe: bool = True,
) -> NativeMocoPreparation:
    """Preserve the public runner boundary and injectable test seams."""
    return _native_moco_preparation.prepare_native_moco(
        request,
        output_dir,
        wrapped_path_auditor=audit_native_wrapped_path_source,
        probe_callable=_native_short_capability_probe,
        run_capability_probe=run_capability_probe,
    )


@dataclass(frozen=True)
class NativeMocoSolve:
    """One native optimizer attempt; success alone does not qualify a fit."""

    success: bool
    status: str
    objective: float | None
    control_knot_count: int
    control_order_sha256: str | None
    state_order_sha256: str | None
    marker_weight_sha256: str | None
    solution_sha256: str | None
    wall_seconds: float
    cpu_seconds: float
    qualification: str = "not-qualified-for-muscle-matching"
    diagnostic_assessment: NativeDiagnosticAssessment | None = None


def _native_goal_weights(study: Any) -> dict[str, float]:
    """Read the native goal's serialized weights, not the Python request object."""
    import opensim as osim

    goal = osim.MocoMarkerTrackingGoal.safeDownCast(
        study.updProblem().updGoal("marker_tracking")
    )
    if goal is None:
        raise ValueError("Native marker tracking goal is unavailable")
    xml = SafeET.fromstring(goal.dump(), forbid_dtd=True, forbid_entities=True)
    items = xml.findall(".//MarkerWeight")
    weights = {item.attrib["name"]: float(item.findtext("weight")) for item in items}
    if len(weights) != len(items):
        raise ValueError("Native marker weights contain duplicate names")
    return weights


def _solution_names(vector: Any) -> tuple[str, ...]:
    if isinstance(vector, (tuple, list)):
        return tuple(str(name) for name in vector)
    return tuple(str(vector.get(i)) for i in range(vector.getSize()))


def _require_physical_body_audit(
    request: NativeMocoRequest, prepared: NativeMocoPreparation, directory: Path
) -> None:
    """Recompute native body physics, not just a claimed sidecar checksum."""
    expected = prepared.physical_body_inertia_sha256
    if not prepared.physical_body_inertia_qualified or expected is None:
        raise ValueError("Physical body inertia admission is missing or rejected")
    try:
        frozen = (directory / "physical_body_inertia.json").read_bytes()
    except OSError as exc:
        raise ValueError("Physical body inertia audit sidecar is missing") from exc
    if hashlib.sha256(frozen).hexdigest() != expected:
        raise ValueError("Physical body inertia audit sidecar changed")
    try:
        current = audit_native_body_inertia(request.model_path, request.model_sha256)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError("Physical body inertia native re-audit failed") from exc
    if not current.admitted or _physical_audit_bytes(current) != frozen:
        raise ValueError("Physical body inertia audit identity changed")


def _require_current_inputs(
    request: NativeMocoRequest, prepared: NativeMocoPreparation, directory: Path
) -> Path:
    """Reject any source, observation or preparation drift before native work."""
    if request.identity_sha256 != prepared.request_sha256:
        raise ValueError("Native Moco request changed after preparation")
    if (
        request.allow_unqualified_passive_diagnostic
        != prepared.allow_unqualified_passive_diagnostic
    ):
        raise ValueError("Native Moco diagnostic disposition changed after preparation")
    if (
        not prepared.physical_body_inertia_qualified
        or prepared.physical_body_inertia_sha256 is None
    ):
        raise ValueError("Physical body inertia admission is missing or rejected")
    if not prepared.ready_for_software_solve:
        raise ValueError("Preparation blockers prohibit native Moco solve")
    for path, expected in (
        (request.model_path, request.model_sha256),
        (request.trc_path, request.trc_sha256),
        (request.states_guess_path, request.states_guess_sha256),
    ):
        if _sha(path) != expected:
            raise ValueError("Source, observation or guess changed after preparation")
    _require_physical_body_audit(request, prepared, directory)
    if (
        prepared.passive_observation_sha256 is not None
        or request.allow_unqualified_passive_diagnostic
    ):
        reference = prepared.passive_observation_receipt
        if reference is None:
            raise ValueError("Passive observation receipt is missing")
        if reference.audit_sha256 != prepared.passive_observation_sha256:
            raise ValueError("Passive observation digest changed after preparation")
        if (
            prepared.native_runtime_sha256 is None
            or prepared.loaded_model_sha256 is None
        ):
            raise ValueError("Passive observation source identities are incomplete")
        observer_sha256 = _sha(
            Path(audit_native_passive_readiness.__code__.co_filename)
        )
        validate_native_passive_observation_receipt(
            directory,
            reference,
            request_sha256=request.identity_sha256,
            source_sha256=prepared.source_sha256 or "",
            loaded_model_sha256=prepared.loaded_model_sha256,
            runtime_sha256=prepared.native_runtime_sha256,
            policy_sha256=(
                request.passive_policy.identity_sha256
                if request.passive_policy is not None
                else None
            ),
            observer_sha256=observer_sha256,
        )
    registered = directory / "registered.trc"
    if _sha(registered) != prepared.registered_trc_sha256:
        raise ValueError("Registered marker reference changed after preparation")
    if hashlib.sha256(read_trc(request.trc_path).time_s.tobytes()).hexdigest() != (
        prepared.observation_clock_sha256
    ):
        raise ValueError("Original observation clock changed after preparation")
    return registered


_SOLVER_PROGRESS_GETTERS = {
    "optim_solver": "get_optim_solver",
    "optim_hessian_approximation": "get_optim_hessian_approximation",
    "optim_sparsity_detection": "get_optim_sparsity_detection",
    "optim_finite_difference_scheme": "get_optim_finite_difference_scheme",
    "transcription_scheme": "get_transcription_scheme",
    "multibody_dynamics_mode": "get_multibody_dynamics_mode",
    "verbosity": "get_verbosity",
    "parallel": "get_parallel",
    "num_mesh_intervals": "get_num_mesh_intervals",
    "optim_max_iterations": "get_optim_max_iterations",
    "optim_convergence_tolerance": "get_optim_convergence_tolerance",
    "optim_constraint_tolerance": "get_optim_constraint_tolerance",
    "optim_ipopt_print_level": "get_optim_ipopt_print_level",
}


def _solver_progress_error(getter_name: str, error: Exception) -> dict[str, Any]:
    """Format an observational getter error without changing its failure detail."""
    return {
        "getter": getter_name,
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error)[:500],
    }


def _solver_progress_properties(
    solver: Any, getter_names: Mapping[str, str] = _SOLVER_PROGRESS_GETTERS
) -> dict[str, dict[str, Any]]:
    """Capture actual solver getter outcomes without inventing missing values."""
    result: dict[str, dict[str, Any]] = {}
    for name, getter_name in getter_names.items():
        try:
            getter = getattr(solver, getter_name, None)
        except (RuntimeError, ValueError) as error:
            result[name] = _solver_progress_error(getter_name, error)
            continue
        if not callable(getter):
            result[name] = {"getter": getter_name, "status": "missing"}
            continue
        try:
            value = getter()
        except (RuntimeError, ValueError) as error:
            result[name] = _solver_progress_error(getter_name, error)
            continue
        if type(value) is float and not math.isfinite(value):
            result[name] = {
                "getter": getter_name,
                "status": "nonfinite",
                "value_type": "float",
                "value_repr": repr(value),
            }
        elif value is None or type(value) in (str, bool, int, float):
            result[name] = {
                "getter": getter_name,
                "status": "returned",
                "value": value,
                "value_type": type(value).__name__,
            }
        else:
            result[name] = {
                "getter": getter_name,
                "status": "non_scalar",
                "value_type": type(value).__name__,
            }
    return result


def _atomic_write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialize finite progress JSON, then use shared atomic byte publication."""
    encoded = (
        json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, encoded)


class _ProgressRecorder(Protocol):
    def __call__(
        self, stage: str, details: Mapping[str, Any] | None = None
    ) -> None: ...


def _make_progress_recorder(
    request: NativeMocoRequest,
    directory: Path,
    wall: float,
    cpu: float,
    current_input_audit_elapsed_wall_seconds: float,
) -> _ProgressRecorder:
    """Bind one attempt's identity and clocks to its atomic event recorder."""
    progress_path = directory / "native_moco_progress.json"
    progress_events: list[dict[str, Any]] = []

    def record_progress(stage: str, details: Mapping[str, Any] | None = None) -> None:
        elapsed_wall = time.perf_counter() - wall
        elapsed_cpu = time.process_time() - cpu
        event: dict[str, Any] = {
            "request_sha256": request.identity_sha256,
            "source_sha256": request.model_sha256,
            "sequence": len(progress_events),
            "stage": stage,
            "monotonic_seconds": time.perf_counter(),
            "elapsed_wall_seconds": elapsed_wall,
            "elapsed_cpu_seconds": elapsed_cpu,
            "current_input_audit_elapsed_wall_seconds": (
                current_input_audit_elapsed_wall_seconds
            ),
        }
        if details is not None:
            event.update(details)
        progress_events.append(event)
        payload: dict[str, Any] = {
            "request_sha256": request.identity_sha256,
            "source_sha256": request.model_sha256,
            "stage": stage,
            "monotonic_seconds": event["monotonic_seconds"],
            "elapsed_wall_seconds": elapsed_wall,
            "elapsed_cpu_seconds": elapsed_cpu,
            "current_input_audit_elapsed_wall_seconds": (
                current_input_audit_elapsed_wall_seconds
            ),
            "timer_origin": "solve_native_moco_after_current_input_audit",
            "events": progress_events,
        }
        if details is not None:
            payload.update(details)
        _atomic_write_progress(progress_path, payload)

    return record_progress


def _native_solve_once(
    request: NativeMocoRequest,
    registered: Path,
    directory: Path,
    wall: float,
    cpu: float,
    *,
    current_input_audit_elapsed_wall_seconds: float = 0.0,
) -> NativeMocoSolve:
    """Run and read back one exact native solver attempt."""
    import opensim as osim

    record_progress = _make_progress_recorder(
        request, directory, wall, cpu, current_input_audit_elapsed_wall_seconds
    )

    record_progress("building-study")

    study = build_moco_study(
        str(request.model_path),
        str(registered),
        str(request.states_guess_path),
        request.config,
        initial_bindings=request.bindings,
        marker_weights=request.marker_weights,
        progress_callback=record_progress,
    )
    record_progress("study-built")
    weights = _native_goal_weights(study)
    if weights != dict(request.marker_weights):
        raise ValueError("Native marker goal did not preserve selected weights")
    weight_sha = hashlib.sha256(
        json.dumps(weights, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_parallel(0)
    record_progress(
        "optimizing",
        {"solver_properties_after_set_parallel": _solver_progress_properties(solver)},
    )
    solution = study.solve()
    record_progress("optimization-returned")
    success = bool(solution.success())
    status = str(solution.getStatus())
    objective = float(solution.getObjective()) if success else None
    count = 0
    controls_sha = states_sha = solution_sha = None
    if success:
        grid = np.asarray(solution.getTimeMat(), dtype=float).reshape(-1)
        controls = _solution_names(solution.getControlNames())
        states = _solution_names(solution.getStateNames())
        if (
            grid.size < 2
            or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)
            or not np.isclose(grid[0], request.config.t_start_s, atol=1e-12, rtol=0)
            or not np.isclose(grid[-1], request.config.horizon_s, atol=1e-12, rtol=0)
            or set(controls) != set(request.bindings.control_bounds)
            or set(states) != set(request.bindings.state_bounds)
        ):
            raise ValueError("Native solution names or full control horizon differ")
        count = int(grid.size)
        controls_sha = hashlib.sha256(json.dumps(controls).encode()).hexdigest()
        states_sha = hashlib.sha256(json.dumps(states).encode()).hexdigest()
        path = directory / "solution.sto"
        solution.write(str(path))
        solution_sha = _sha(path)
    result = NativeMocoSolve(
        success,
        status,
        objective,
        count,
        controls_sha,
        states_sha,
        weight_sha,
        solution_sha,
        time.perf_counter() - wall,
        time.process_time() - cpu,
    )
    return result


def solve_native_moco(
    request: NativeMocoRequest,
    prepared: NativeMocoPreparation,
    output_dir: Path,
) -> NativeMocoSolve:
    """Solve prepared inputs, retaining failed native attempts as receipts."""
    directory = Path(output_dir)
    input_audit_start = time.perf_counter()
    registered = _require_current_inputs(request, prepared, directory)
    input_audit_elapsed = time.perf_counter() - input_audit_start
    wall = time.perf_counter()
    cpu = time.process_time()
    try:
        result = _native_solve_once(
            request,
            registered,
            directory,
            wall,
            cpu,
            current_input_audit_elapsed_wall_seconds=input_audit_elapsed,
        )
    except (RuntimeError, ValueError) as error:
        result = NativeMocoSolve(
            False,
            f"native-error:{type(error).__name__}",
            None,
            0,
            None,
            None,
            None,
            None,
            time.perf_counter() - wall,
            time.process_time() - cpu,
        )
    preparation_assessment = prepared.diagnostic_assessment
    if preparation_assessment is None:
        raise ValueError("Native preparation has no typed diagnostic assessment")
    solve_evidence_sha = hashlib.sha256(
        json.dumps(
            {
                "preparation_assessment": preparation_assessment.to_json_dict(),
                "solve_attempt": asdict(result),
                "time_window_seconds": (
                    request.config.t_start_s,
                    request.config.horizon_s,
                ),
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    solve_assessment = replace(
        preparation_assessment,
        runtime_viability=(
            RuntimeViability.VIABLE if result.success else RuntimeViability.FAILED
        ),
        runtime_phase=RuntimePhase.OPTIMIZATION,
        runtime_window_seconds=(
            request.config.t_start_s,
            request.config.horizon_s,
        ),
        runtime_evidence_sha256=solve_evidence_sha,
    )
    result = replace(result, diagnostic_assessment=solve_assessment)
    payload = json.dumps(asdict(result), indent=2, allow_nan=False) + "\n"
    attempts = directory / "solve_attempts"
    attempts.mkdir(exist_ok=True)
    for number in range(1, 1_000_001):
        try:
            with (attempts / f"{number:06d}.json").open(
                "x", encoding="utf-8"
            ) as stream:
                stream.write(payload)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Native solve attempt receipt space exhausted")
    (directory / "solve.json").write_text(payload, encoding="utf-8")
    return result


__all__ = [
    "NativeMocoPreparation",
    "NativeMocoRequest",
    "NativeMocoSolve",
    "prepare_native_moco",
    "solve_native_moco",
]
