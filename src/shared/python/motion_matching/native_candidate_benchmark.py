"""F03 candidate checks through F06's independent native torque replay.

This bounded one-hinge fixture has an analytic held-input solution. It tests
transcription and replay boundaries; it is not a full-body/contact benchmark.
"""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from src.shared.python.motion_matching.sparse_collocation_spike import (
    BenchmarkBackend,
    BenchmarkHardware,
    BenchmarkStart,
    CollocationResult,
    SecondOrderFixture,
    SparseCollocationProblem,
    capture_benchmark_hardware,
)

Array: TypeAlias = NDArray[np.float64]
_MAX_NATIVE_STEPS = 100_000
_MAX_ATTEMPTS = 100


@dataclass(frozen=True)
class NativeCandidateGate:
    """Predeclared synthetic tolerances, separated by physical meaning."""

    max_native_node_gap: float
    max_native_exact_gap: float
    max_observation_rmse_rad: float
    max_torque_violation_nm: float
    max_rate_violation_nm_s: float
    max_constraint_defect_by_kind: Mapping[str, float]

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.max_native_node_gap,
                self.max_native_exact_gap,
                self.max_observation_rmse_rad,
                self.max_torque_violation_nm,
                self.max_rate_violation_nm_s,
                *self.max_constraint_defect_by_kind.values(),
            ]
        )
        if (
            not self.max_constraint_defect_by_kind
            or not np.isfinite(values).all()
            or min(values[:3]) <= 0
            or np.any(values[3:] < 0)
        ):
            raise ValueError("native candidate thresholds must be finite and valid")


@dataclass(frozen=True)
class NativeCandidateReceipt:
    """Four distinct numerical and observation checks plus native provenance."""

    accepted: bool
    failure_reason: str | None
    defect_kind: str
    transcription_defect: float
    max_torque_violation_nm: float
    max_rate_violation_nm_s: float
    max_continuous_exact_gap: float
    max_native_exact_gap: float
    max_native_node_gap: float
    observation_rmse_rad: float
    input_sha256: str
    policy_sha256: str
    model_sha256: str
    time_grid_sha256: str
    native_step_seconds: float
    applied_torque_rows: int
    native_replay_mode: str
    native_contact_present: bool


def exact_held_solution(
    fixture: SecondOrderFixture,
    times_s: Array,
    initial_state: Array,
    torque_nm: Array,
) -> Array:
    """Compute ZOH truth by matrix exponential, independent of RK4/RK45."""
    times = np.asarray(times_s, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    torque = np.asarray(torque_nm, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
        or initial.shape != (2,)
        or not np.isfinite(initial).all()
        or torque.shape != (len(times) - 1,)
        or not np.isfinite(torque).all()
    ):
        raise ValueError("exact held solve needs finite state, clock and input")
    inertia = fixture.inertia_kg_m2
    augmented = np.array(
        [
            [0.0, 1.0, 0.0],
            [
                -fixture.stiffness_nm_rad / inertia,
                -fixture.damping_nm_s_rad / inertia,
                1.0 / inertia,
            ],
            [0.0, 0.0, 0.0],
        ]
    )
    states: Array = np.empty((len(times), 2), dtype=float)
    states[0] = initial
    for i, (dt, applied) in enumerate(zip(np.diff(times), torque, strict=True)):
        transition = expm(augmented * dt)
        states[i + 1] = transition[:2, :2] @ states[i] + transition[:2, 2] * applied
    if not np.isfinite(states).all():
        raise ValueError("exact held solution is nonfinite")
    return states


def _refined_history(
    problem: SparseCollocationProblem, torque_nm: Array, substeps: int
) -> tuple[Array, Array, float]:
    times = np.asarray(problem.times_s, dtype=float)
    torque = np.asarray(torque_nm, dtype=float)
    if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps <= 0:
        raise ValueError("native refinement must be a positive integer")
    steps = np.diff(times)
    if times[0] != 0 or not np.allclose(steps, steps[0], atol=1e-12, rtol=0):
        raise ValueError("native candidate clock must be uniform and start at zero")
    if torque.shape != (problem.intervals,) or not np.isfinite(torque).all():
        raise ValueError("candidate torque must cover finite original intervals")
    count = problem.intervals * substeps
    if count > _MAX_NATIVE_STEPS:
        raise ValueError("native candidate exceeds bounded step budget")
    native_dt = float(steps[0] / substeps)
    native_times = np.arange(count + 1, dtype=float) * native_dt
    held: Array = np.repeat(torque, substeps)
    return native_times, held, native_dt


def _rotary_xml(
    fixture: SecondOrderFixture,
    *,
    native_dt: float,
    torque_lower_nm: float,
    torque_upper_nm: float,
) -> str:
    """One collocated hinge with exactly specified inertia and passive forces."""
    inertia = fixture.inertia_kg_m2
    damping = fixture.damping_nm_s_rad
    stiffness = fixture.stiffness_nm_rad
    return (
        '<mujoco model="f03b-rotary">\n'
        f'  <option timestep="{native_dt:.17g}" integrator="RK4" gravity="0 0 0"/>\n'
        '  <worldbody><body name="rotor">\n'
        f'    <joint name="q" type="hinge" axis="0 1 0" damping="{damping:.17g}" stiffness="{stiffness:.17g}"/>\n'
        f'    <inertial pos="0 0 0" mass="1" diaginertia="{inertia:.17g} {inertia:.17g} {inertia:.17g}"/>\n'
        "  </body></worldbody>\n"
        f'  <actuator><motor name="tau" joint="q" gear="1" ctrllimited="true" ctrlrange="{torque_lower_nm:.17g} {torque_upper_nm:.17g}"/></actuator>\n'
        "</mujoco>\n"
    )


def _ensure_native_model(path: Path, source: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != source:
            raise ValueError("existing native model differs from candidate fixture")
        return
    if not path.parent.is_dir():
        raise ValueError("native fixture output directory must exist")
    path.write_text(source, encoding="utf-8", newline="\n")


def _native_initial_state(path: Path, initial_state: Array) -> Array:
    from src.engines.physics_engines.mujoco.python.native_torque_replay import (
        native_initial_state_from_joint_state,
    )

    return native_initial_state_from_joint_state(path, initial_state)


def _failure_reason(
    result: CollocationResult,
    gate: NativeCandidateGate,
    *,
    torque_violation_nm: float,
    rate_violation_nm_s: float,
    native_exact_gap: float,
    native_node_gap: float,
    observation_rmse: float,
) -> str | None:
    if not result.optimizer_converged:
        return "optimizer"
    defect_limit = gate.max_constraint_defect_by_kind.get(result.defect_kind)
    if defect_limit is None or result.max_dynamics_defect > defect_limit:
        return "transcription_defect"
    if torque_violation_nm > gate.max_torque_violation_nm:
        return "actuator_bound"
    if rate_violation_nm_s > gate.max_rate_violation_nm_s:
        return "rate_bound"
    if native_exact_gap > gate.max_native_exact_gap:
        return "native_integrator_gap"
    if native_node_gap > gate.max_native_node_gap:
        return "native_node_gap"
    if observation_rmse > gate.max_observation_rmse_rad:
        return "observation_rmse"
    return None


def _measured_input_violations(
    problem: SparseCollocationProblem, torque_nm: Array
) -> tuple[float, float]:
    """Check executed torque and slew independently of solver metadata."""
    torque_violation = max(
        0.0,
        float(np.max(problem.torque_lower_nm - torque_nm)),
        float(np.max(torque_nm - problem.torque_upper_nm)),
    )
    rate = np.diff(torque_nm) / np.diff(problem.times_s[:-1])
    rate_violation = max(
        0.0,
        float(np.max(np.abs(rate), initial=0.0) - problem.rate_limit_nm_s),
    )
    return torque_violation, rate_violation


def validate_candidate_native(
    problem: SparseCollocationProblem,
    result: CollocationResult,
    *,
    model_path: Path,
    substeps: int,
    gate: NativeCandidateGate,
) -> NativeCandidateReceipt:
    """Run candidate's actual ZOH torque through a fresh F06 native replay."""
    from src.engines.physics_engines.mujoco.python.native_torque_replay import (
        build_native_torque_bundle,
        replay_native_torque_bundle,
    )

    if result.states.shape != (len(problem.times_s), 2):
        raise ValueError("candidate transcription state shape is incompatible")
    if not np.isfinite(result.states).all():
        raise ValueError("candidate transcription states are nonfinite")
    native_times, held, native_dt = _refined_history(
        problem, result.torque_nm, substeps
    )
    source = _rotary_xml(
        problem.fixture,
        native_dt=native_dt,
        torque_lower_nm=problem.torque_lower_nm,
        torque_upper_nm=problem.torque_upper_nm,
    )
    _ensure_native_model(model_path, source)
    initial = _native_initial_state(model_path, problem.initial_state)
    # The terminal row is a sentinel; every preceding row is the exact torque
    # held for one native step. No authored polynomial is called "same input".
    values = np.r_[held, held[-1]].reshape(-1, 1)
    bundle = build_native_torque_bundle(
        model_path, initial, native_times, values, experiment_id="f03b-native"
    )
    replay = replay_native_torque_bundle(bundle, model_path)
    replay_policy = bundle.policy
    native_states = np.column_stack((replay.qpos[:, 0], replay.qvel[:, 0]))
    exact = exact_held_solution(
        problem.fixture, native_times, problem.initial_state, held
    )
    coarse_native = native_states[::substeps]
    continuous = problem.fixture.forward_held(
        problem.times_s, problem.initial_state, result.torque_nm
    )
    native_exact_gap = float(np.max(np.abs(native_states - exact)))
    native_node_gap = float(np.max(np.abs(coarse_native - result.states)))
    observation_rmse = float(
        np.sqrt(np.mean((coarse_native[:, 0] - problem.target_q_rad) ** 2))
    )
    measured_torque, measured_rate = _measured_input_violations(
        problem, result.torque_nm
    )
    torque_violation = max(result.max_torque_violation, measured_torque)
    rate_violation = max(result.max_rate_violation, measured_rate)
    reason = _failure_reason(
        result,
        gate,
        torque_violation_nm=torque_violation,
        rate_violation_nm_s=rate_violation,
        native_exact_gap=native_exact_gap,
        native_node_gap=native_node_gap,
        observation_rmse=observation_rmse,
    )
    return NativeCandidateReceipt(
        accepted=reason is None,
        failure_reason=reason,
        defect_kind=result.defect_kind,
        transcription_defect=result.max_dynamics_defect,
        max_torque_violation_nm=torque_violation,
        max_rate_violation_nm_s=rate_violation,
        max_continuous_exact_gap=float(np.max(np.abs(continuous - exact[::substeps]))),
        max_native_exact_gap=native_exact_gap,
        max_native_node_gap=native_node_gap,
        observation_rmse_rad=observation_rmse,
        input_sha256=bundle.applied_input_sha256,
        policy_sha256=bundle.policy_sha256,
        model_sha256=bundle.model.source_model_sha256,
        time_grid_sha256=bundle.time_grid_sha256,
        native_step_seconds=native_dt,
        applied_torque_rows=len(replay.applied_actuator_torques),
        native_replay_mode=replay_policy.replay_mode.value,
        native_contact_present=False,
    )


@dataclass(frozen=True)
class NativeBenchmarkAttempt:
    """One predeclared start, including failures and receipt export cost."""

    backend: str
    start: str
    warm: bool
    accepted: bool
    failure_reason: str | None
    preparation_seconds: float
    solve_seconds: float
    native_seconds: float | None
    export_seconds: float
    total_seconds: float
    peak_python_bytes: int
    receipt_sha256: str
    receipt_path: Path
    candidate_receipt: NativeCandidateReceipt | None


@dataclass(frozen=True)
class NativeBenchmarkSummary:
    """Attempt-distribution summary; failed attempts remain in total costs."""

    backend: str
    warm: bool
    attempts: int
    accepted: int
    p50_total_seconds: float
    p95_total_seconds: float
    p50_peak_python_bytes: float
    p95_peak_python_bytes: float


@dataclass(frozen=True)
class NativeBenchmarkReport:
    """Named-hardware receipts across all declared native candidate starts."""

    hardware: BenchmarkHardware
    attempts: tuple[NativeBenchmarkAttempt, ...]

    def summary(self, backend: str, *, warm: bool) -> NativeBenchmarkSummary:
        rows = tuple(
            attempt
            for attempt in self.attempts
            if attempt.backend == backend and attempt.warm == warm
        )
        if not rows:
            raise ValueError("native benchmark group has no declared attempts")
        seconds = np.array([row.total_seconds for row in rows])
        peak = np.array([row.peak_python_bytes for row in rows])
        return NativeBenchmarkSummary(
            backend=backend,
            warm=warm,
            attempts=len(rows),
            accepted=sum(row.accepted for row in rows),
            p50_total_seconds=float(np.percentile(seconds, 50)),
            p95_total_seconds=float(np.percentile(seconds, 95)),
            p50_peak_python_bytes=float(np.percentile(peak, 50)),
            p95_peak_python_bytes=float(np.percentile(peak, 95)),
        )

    def time_to_first_accepted_s(self, backend: str) -> float | None:
        elapsed = 0.0
        seen = False
        for attempt in self.attempts:
            if attempt.backend != backend:
                continue
            seen = True
            elapsed += attempt.total_seconds
            if attempt.accepted:
                return elapsed
        if not seen:
            raise ValueError("unknown native benchmark backend")
        return None


@dataclass(frozen=True)
class _AttemptTiming:
    preparation_seconds: float
    solve_seconds: float
    native_seconds: float | None


def _export_attempt(
    path: Path,
    *,
    backend: str,
    hardware: BenchmarkHardware,
    start: BenchmarkStart,
    reason: str | None,
    receipt: NativeCandidateReceipt | None,
    timing: _AttemptTiming,
) -> tuple[str, float]:
    """Write one deterministic synthetic receipt, including rejected starts."""
    began = time.perf_counter()
    payload = {
        "schema_version": "f03b-native-candidate/1.0.0",
        "backend": backend,
        "hardware": asdict(hardware),
        "start": start.name,
        "warm": start.warm,
        "failure_reason": reason,
        "candidate": None if receipt is None else asdict(receipt),
        "preparation_seconds": timing.preparation_seconds,
        "solve_seconds": timing.solve_seconds,
        "native_seconds": timing.native_seconds,
    }
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return digest, time.perf_counter() - began


def _run_native_attempt(
    problem: SparseCollocationProblem,
    backend: BenchmarkBackend,
    hardware: BenchmarkHardware,
    start: BenchmarkStart,
    gate: NativeCandidateGate,
    substeps: int,
    model_path: Path,
    receipt_path: Path,
) -> NativeBenchmarkAttempt:
    """Measure preparation, derivatives/solve, native validation and export."""
    tracemalloc.start()
    began = time.perf_counter()
    preparation_started = time.perf_counter()
    initial = start.initial_torque_nm.copy()
    preparation_seconds = time.perf_counter() - preparation_started
    solve_seconds = 0.0
    native_seconds: float | None = None
    receipt: NativeCandidateReceipt | None = None
    stage = "solve"
    stage_started = time.perf_counter()
    try:
        result = backend.solve(problem, initial)
        solve_seconds = time.perf_counter() - stage_started
        stage = "native"
        stage_started = time.perf_counter()
        receipt = validate_candidate_native(
            problem, result, model_path=model_path, substeps=substeps, gate=gate
        )
        native_seconds = time.perf_counter() - stage_started
        reason = receipt.failure_reason
    except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
        if stage == "solve":
            solve_seconds = time.perf_counter() - stage_started
        else:
            native_seconds = time.perf_counter() - stage_started
        reason = str(exc)
    digest, export_seconds = _export_attempt(
        receipt_path,
        backend=backend.name,
        hardware=hardware,
        start=start,
        reason=reason,
        receipt=receipt,
        timing=_AttemptTiming(preparation_seconds, solve_seconds, native_seconds),
    )
    total_seconds = time.perf_counter() - began
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return NativeBenchmarkAttempt(
        backend=backend.name,
        start=start.name,
        warm=start.warm,
        accepted=reason is None,
        failure_reason=reason,
        preparation_seconds=preparation_seconds,
        solve_seconds=solve_seconds,
        native_seconds=native_seconds,
        export_seconds=export_seconds,
        total_seconds=total_seconds,
        peak_python_bytes=peak,
        receipt_sha256=digest,
        receipt_path=receipt_path,
        candidate_receipt=receipt,
    )


def benchmark_native_candidates(
    problem: SparseCollocationProblem,
    *,
    backends: Sequence[BenchmarkBackend],
    starts: Sequence[BenchmarkStart],
    gate: NativeCandidateGate,
    substeps: int,
    output_dir: Path,
) -> NativeBenchmarkReport:
    """Serial bounded benchmark with no hidden warmups or dropped failures."""
    if (
        not backends
        or not starts
        or len(backends) * len(starts) > _MAX_ATTEMPTS
        or len({backend.name for backend in backends}) != len(backends)
        or len({start.name for start in starts}) != len(starts)
        or any(
            start.initial_torque_nm.shape != (problem.intervals,) for start in starts
        )
        or not output_dir.is_dir()
    ):
        raise ValueError("native benchmark needs bounded unique starts and output dir")
    hardware = capture_benchmark_hardware()
    attempts = tuple(
        _run_native_attempt(
            problem,
            backend,
            hardware,
            start,
            gate,
            substeps,
            output_dir / f"native-model-{backend_index}-{start_index}.xml",
            output_dir / f"native-attempt-{backend_index}-{start_index}.json",
        )
        for backend_index, backend in enumerate(backends)
        for start_index, start in enumerate(starts)
    )
    return NativeBenchmarkReport(hardware=hardware, attempts=attempts)
