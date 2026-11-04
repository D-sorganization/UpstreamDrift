"""Shared zero-control rollout and run writer for every engine (Issue #14435).

Engine run paths share :class:`Rollout` (validation, seeded initial velocity)
and :func:`write_trajectory_run` (``trajectory.npy`` plus the run manifest).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.sim_run.manifest import RunManifest
from src.sim_run.manifest_emitter import RunProvenance, emit_run_manifest, sha256_json
from src.sim_run.worker import get_repo_sha

TRAJECTORY_FILENAME = "trajectory.npy"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Rollout:
    """One zero-control rollout; raises ValueError on a bad duration/seed/noise."""

    duration: float
    seed: int = 0
    qvel_noise: float = 0.0

    def __post_init__(self) -> None:
        _require(self.duration > 0, f"duration must be positive, got {self.duration}")
        _require(self.seed >= 0, f"seed must be non-negative, got {self.seed}")
        _require(
            self.qvel_noise >= 0,
            f"qvel_noise must be non-negative, got {self.qvel_noise}",
        )

    def steps(self, timestep: float) -> int:
        """Number of ``timestep`` steps covering ``duration`` (at least one)."""
        _require(timestep > 0, f"timestep must be positive, got {timestep}")
        return max(1, round(self.duration / timestep))

    def seeded_qvel(self, nv: int) -> np.ndarray:
        """Initial velocity drawn from ``N(0, qvel_noise^2)`` with ``seed``."""
        return np.random.default_rng(self.seed).normal(0.0, self.qvel_noise, nv)


@dataclass(frozen=True)
class TrajectoryRun:
    """What one rollout produced and how, ready to write as a run."""

    model_id: str
    model_text: str
    rollout: Rollout
    engine: tuple[str, str]
    initial: tuple[Sequence[float], Sequence[float]]
    parameters: Mapping[str, Any]


def write_trajectory_run(
    out_dir: Path,
    rows: Sequence[np.ndarray],
    run: TrajectoryRun,
    *,
    ud_commit: str | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    """Save ``rows`` as ``trajectory.npy`` and emit the run manifest beside it.

    Raises:
        ValueError: if no UpstreamDrift commit resolves.
        RunManifestError: if the resulting manifest is invalid.
    """
    commit = ud_commit or get_repo_sha()
    _require(commit != "0" * 40, "ud_commit could not be resolved")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trajectory = out / TRAJECTORY_FILENAME
    np.save(trajectory, np.asarray(rows))
    qpos, qvel = run.initial
    provenance = RunProvenance(
        model_id=run.model_id,
        revision="sha256:" + hashlib.sha256(run.model_text.encode("utf-8")).hexdigest(),
        parameters={
            "duration_s": run.rollout.duration,
            "controls": "zero",
            "qvel_noise": run.rollout.qvel_noise,
            "trajectory_columns": {"time": 1, "qpos": len(qpos), "qvel": len(qvel)},
            **run.parameters,
        },
        input_sha256=sha256_json({"qpos": list(qpos), "qvel": list(qvel)}),
        seed=run.rollout.seed,
        engine_name=run.engine[0],
        engine_version=run.engine[1],
        ud_commit=commit,
        input_label="initial condition: initial pose with seeded qvel",
    )
    return emit_run_manifest(
        out, outputs=[trajectory], provenance=provenance, created_at=created_at
    )
