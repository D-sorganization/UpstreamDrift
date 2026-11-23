"""Restricted bridge from a Solution Bundle to fresh boundary-state replay.

This is not a complete Solution Bundle verifier: the generic backend reports
commanded controls, not post-actuator torque or contact. Model, runtime and
factory identities remain caller-attested. Only a strict no-impact, stateless,
fixed-rate ZOH subset is accepted, and comparison never uses markers.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.replay_optimized_candidate import Controller, replay_candidate
from src.shared.python.optimization.solution_bundle import load_solution_bundle
from src.shared.python.simulation_backends.protocol import (
    SimState,
    SimulationBackend,
    Trace,
)
from src.shared.python.simulation_backends.validation import (
    ReplayTolerance,
    ValidationReport,
    compare_replay_quantities,
)


@dataclass(frozen=True)
class ControllerInputs:
    """Recorded controller inputs only; rollout and measured torques are excluded."""

    joint_order: tuple[str, ...]
    time: np.ndarray
    q_ref: np.ndarray
    qd_ref: np.ndarray
    tau_ff_ref: np.ndarray
    knots: np.ndarray
    delta_tau_coeffs: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    control: dict[str, Any]


@dataclass(frozen=True)
class PartialReplayEvidence:
    """Boundary comparisons only; never a complete/independent replay verdict."""

    trace: Trace
    reports: dict[str, ValidationReport]
    applied_torque_verified: bool = False
    impact_verified: bool = False
    independently_verified: bool = False


@dataclass(frozen=True)
class MeasuredLocalEvidence:
    """Measured file identities; factories and full environment remain unverified."""

    replay: PartialReplayEvidence
    model_file_hash_verified: bool = True
    partial_runtime_verified: bool = True
    factory_model_verified: bool = False


def measured_replay_runtime() -> str:
    """Measure a declared runtime subset, not a complete environment lock.

    The executing Python binary, NumPy core binary and bridge/runner/protocol
    source files are measured. Backend and controller implementations, their
    native dependencies and factory construction are not attested.
    """
    import scripts.replay_optimized_candidate as runner
    import src.shared.python.optimization.solution_bundle as codec
    import src.shared.python.simulation_backends.protocol as protocol
    import src.shared.python.simulation_backends.validation as validation

    try:
        core = importlib.import_module("numpy._core._multiarray_umath")
    except ModuleNotFoundError:
        core = importlib.import_module("numpy.core._multiarray_umath")
    if core.__file__ is None:
        raise ValueError("NumPy core binary unavailable")
    modules = (runner, protocol, codec, validation)
    paths = [Path(sys.executable), Path(core.__file__), Path(__file__)]
    for module in modules:
        if module.__file__ is None:
            raise ValueError("runtime source file unavailable")
        paths.append(Path(module.__file__))
    digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "files": digests,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def verify_measured_local_replay(
    directory: str | Path,
    *,
    model_path: str | Path,
    backend_factory: Callable[[], SimulationBackend],
    controller_factory: Callable[[ControllerInputs], Controller],
    tolerances: Mapping[str, ReplayTolerance],
) -> MeasuredLocalEvidence:
    """Measure the model file and runtime subset before constructing factories.

    The model file digest is bound to the bundle, but the factory is not proven
    to have consumed that file. Applied torque and impact remain unverified.
    """
    model_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    replay = verify_bundle_replay(
        directory,
        expected_model_hash=model_hash,
        runtime_sha256=measured_replay_runtime(),
        backend_factory=backend_factory,
        controller_factory=controller_factory,
        tolerances=tolerances,
    )
    return MeasuredLocalEvidence(replay=replay)


def playback_bundle(directory: str | Path, *, expected_model_hash: str) -> Trace:
    """Return detached recorded arrays; no backend or markers are used."""
    bundle = load_solution_bundle(directory, expected_model_hash=expected_model_hash)
    a = bundle.arrays
    t = np.asarray(a["time"], dtype=float)
    if t.size < 2:
        raise ValueError("at least two playback samples are required")
    return Trace(
        t=t.copy(),
        q=a["q_rollout"].copy(),
        v=a["qd_rollout"].copy(),
        u=a["tau_cmd"].copy(),
        dt=float(t[1] - t[0]),
        backend="recorded",
        meta={"replay_mode": "recorded_playback", "control_quantity": "commanded"},
        torques=a["tau_applied"].copy(),
    )


def verify_bundle_replay(
    directory: str | Path,
    *,
    expected_model_hash: str,
    runtime_sha256: str,
    backend_factory: Callable[[], SimulationBackend],
    controller_factory: Callable[[ControllerInputs], Controller],
    tolerances: Mapping[str, ReplayTolerance],
) -> PartialReplayEvidence:
    """Run fresh plant/controller factories and compare selected boundary values.

    The caller must independently construct fresh factories from the declared
    model, environment, controller and runtime. Hash strings are caller
    assertions, not measurements; successful comparisons are not acceptance.
    """
    bundle = load_solution_bundle(directory, expected_model_hash=expected_model_hash)
    a, policy = bundle.arrays, bundle.integrator
    keys = {"time", "q", "qd", "tau_cmd"}
    if set(tolerances) != keys or any(
        not isinstance(value, ReplayTolerance) for value in tolerances.values()
    ):
        raise ValueError("explicit tolerances required for each boundary quantity")
    if (
        len(expected_model_hash) != 64
        or any(c not in "0123456789abcdef" for c in expected_model_hash)
        or len(runtime_sha256) != 64
        or any(c not in "0123456789abcdef" for c in runtime_sha256)
        or policy.get("runtime_sha256") != runtime_sha256
    ):
        raise ValueError("model and runtime hashes must be matching SHA-256 values")
    if bundle.asset_hashes:
        raise ValueError("external assets are unsupported by the restricted runner")
    if bundle.control != {
        "hold": "zoh",
        "terminal_command": "unused_zero",
        "saturation": "none",
    }:
        raise ValueError("unsupported controller hold or terminal policy")
    if (
        set(policy) != {"kind", "dt", "substeps", "runtime_sha256"}
        or policy.get("kind") != "fixed_zoh"
    ):
        raise ValueError("unsupported integrator policy")
    dt, substeps = policy["dt"], policy["substeps"]
    if (
        isinstance(dt, bool)
        or not isinstance(dt, int | float)
        or not np.isfinite(dt)
        or dt <= 0
        or isinstance(substeps, bool)
        or not isinstance(substeps, int)
        or substeps < 1
    ):
        raise ValueError("invalid fixed-step schedule")
    time = np.asarray(a["time"], dtype=float)
    if time.size < 2 or not np.allclose(
        time, time[0] + np.arange(time.size) * dt, rtol=0, atol=min(1e-12, dt * 1e-6)
    ):
        raise ValueError("recorded time must match the fixed control grid")
    if bundle.impact_event["t"] <= time[-1]:
        raise ValueError("impact at or before the terminal state is unsupported")
    if a["controller_state0"].size or np.any(a["tau_cmd"][-1] != 0):
        raise ValueError(
            "only stateless control and zero unused terminal command are supported"
        )
    inputs = ControllerInputs(
        joint_order=bundle.joint_order,
        **{
            name: a[name].copy()
            for name in (
                "time",
                "q_ref",
                "qd_ref",
                "tau_ff_ref",
                "knots",
                "delta_tau_coeffs",
                "kp",
                "kd",
            )
        },
        control=copy.deepcopy(bundle.control),
    )
    trace = replay_candidate(
        backend_factory=backend_factory,
        controller_factory=lambda: controller_factory(inputs),
        initial=SimState(a["q0"].copy(), a["qd0"].copy(), time=float(time[0])),
        steps=time.size - 1,
        dt=float(dt),
        substeps=substeps,
    )
    recorded = {
        "time": time,
        "q": a["q_rollout"],
        "qd": a["qd_rollout"],
        "tau_cmd": a["tau_cmd"][:-1],
    }
    replayed = {"time": trace.t, "q": trace.q, "qd": trace.v, "tau_cmd": trace.u[:-1]}
    for key in keys:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(replayed[key] - recorded[key])
            allowance = tolerances[key].atol + tolerances[key].rtol * np.abs(
                recorded[key]
            )
        if not np.isfinite(difference).all() or not np.isfinite(allowance).all():
            raise ValueError(f"{key}: comparison arithmetic is nonfinite")
    reports = compare_replay_quantities(
        recorded,
        replayed,
        tolerances,
        recorded_runtime=runtime_sha256,
        replay_runtime=runtime_sha256,
    )
    return PartialReplayEvidence(trace=trace, reports=reports)
