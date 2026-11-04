"""Run OpenSim, Drake, Pinocchio or MyoSuite and emit a run manifest (Issue #14435).

:func:`run_engine` drives any shared ``PhysicsEngine`` through one zero-control
rollout and writes it with :func:`~src.sim_run.trajectory_run.write_trajectory_run`.
An engine whose runtime is absent raises :class:`EngineUnavailable` with why.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.engine_availability import (
    EngineStatus,
    get_engine_error,
    get_engine_status,
)
from src.sim_run.manifest import RunManifest
from src.sim_run.trajectory_run import Rollout, TrajectoryRun, write_trajectory_run


class EngineUnavailable(RuntimeError):
    """Raised when an engine's runtime cannot be imported on this machine."""


@dataclass(frozen=True)
class EngineAdapter:
    """``"module:Class"`` of a ``PhysicsEngine``, its model format and the
    installed distributions its version may be read from, in order."""

    engine_class: str
    extension: str
    distributions: tuple[str, ...]

    def version(self) -> str | None:
        for distribution in self.distributions:
            try:
                return metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
        return None


_ENGINES = "src.engines.physics_engines"
ENGINE_ADAPTERS: dict[str, EngineAdapter] = {
    "pinocchio": EngineAdapter(
        f"{_ENGINES}.pinocchio.python.pinocchio_physics_engine:PinocchioPhysicsEngine",
        "urdf",
        ("pin", "pinocchio"),
    ),
    "drake": EngineAdapter(
        f"{_ENGINES}.drake.python.drake_physics_engine:DrakePhysicsEngine",
        "urdf",
        ("drake",),
    ),
    "opensim": EngineAdapter(
        f"{_ENGINES}.opensim.python.opensim_physics_engine:OpenSimPhysicsEngine",
        "osim",
        ("opensim",),
    ),
    "myosuite": EngineAdapter(
        f"{_ENGINES}.myosuite.python.myosuite_physics_engine:MyoSuitePhysicsEngine",
        "xml",
        ("myosuite",),
    ),
}


def _load_engine(name: str) -> tuple[Any, str, str]:
    """Return ``(engine, version, extension)``; the shared probe decides availability."""
    if name not in ENGINE_ADAPTERS:
        raise ValueError(f"unknown engine {name!r}; known: {sorted(ENGINE_ADAPTERS)}")
    status = get_engine_status(name)
    if status is not EngineStatus.AVAILABLE:
        reason = get_engine_error(name) or "no usable runtime"
        raise EngineUnavailable(f"{name} is not available ({status.value}: {reason})")
    adapter = ENGINE_ADAPTERS[name]
    version = adapter.version()
    if version is None:
        raise EngineUnavailable(f"{name} is installed but reports no version")
    module_name, class_name = adapter.engine_class.split(":")
    engine_type = getattr(importlib.import_module(module_name), class_name)
    return engine_type(), version, adapter.extension


def _row(engine: Any) -> np.ndarray:
    qpos, qvel = engine.get_state()
    return np.concatenate(([engine.get_time()], qpos, qvel))


def run_engine(
    name: str,
    model: str,
    *,
    model_id: str,
    output_dir: Path,
    rollout: Rollout,
    timestep: float,
    ud_commit: str | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    """Roll ``model`` out on engine ``name``; write ``[time, qpos, qvel]`` rows.

    Raises:
        ValueError: if ``name`` is unknown, ``timestep`` is not positive, or no
            UpstreamDrift commit resolves.
        EngineUnavailable: if the engine's runtime is not installed.
        RunManifestError: if the resulting manifest is invalid.
    """
    steps = rollout.steps(timestep)
    engine, version, extension = _load_engine(name)
    engine.load_from_string(model, extension)
    engine.reset()
    qpos, qvel = engine.get_state()
    engine.set_state(qpos, rollout.seeded_qvel(len(qvel)))
    engine.forward()
    initial_q, initial_v = engine.get_state()
    rows = [_row(engine)]
    for _ in range(steps):
        engine.step(timestep)
        rows.append(_row(engine))
    run = TrajectoryRun(
        model_id=model_id,
        model_text=model,
        rollout=rollout,
        engine=(name, version),
        initial=(np.asarray(initial_q).tolist(), np.asarray(initial_v).tolist()),
        parameters={"timestep_s": timestep, "steps": steps, "model_format": extension},
    )
    return write_trajectory_run(
        output_dir, rows, run, ud_commit=ud_commit, created_at=created_at
    )
