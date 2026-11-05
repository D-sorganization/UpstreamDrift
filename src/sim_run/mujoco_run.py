"""MuJoCo run path that writes its trajectory and a run manifest (Issue #14432).

:func:`run_mujoco` compiles an MJCF model, rolls it out with zero controls from
``qpos0`` and a seeded initial velocity, writes the trajectory to
``trajectory.npy`` and emits ``run_manifest.json`` beside it through the shared
emitter. The model revision is the SHA-256 of the MJCF text, the input digest
identifies the initial condition, and the manifest records the trajectory hash.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from src.sim_run.manifest import RunManifest
from src.sim_run.trajectory_run import (
    TRAJECTORY_FILENAME,
    Rollout,
    TrajectoryRun,
    write_trajectory_run,
)

__all__ = ["TRAJECTORY_FILENAME", "run_mujoco"]


def run_mujoco(
    xml: str,
    *,
    model_id: str,
    output_dir: Path,
    duration: float,
    seed: int = 0,
    qvel_noise: float = 0.0,
    ud_commit: str | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    """Run ``xml`` for ``duration`` seconds and write the trajectory and manifest.

    ``trajectory.npy`` holds one row per state, ``[time, qpos..., qvel...]``,
    including the initial state. The initial velocity is drawn from
    ``N(0, qvel_noise^2)`` with ``numpy.random.default_rng(seed)``.

    Raises:
        ValueError: if ``duration`` is not positive, ``seed`` is negative,
            ``qvel_noise`` is negative, or no UpstreamDrift commit resolves.
        RunManifestError: if the resulting manifest is invalid.
    """
    rollout = Rollout(duration, seed, qvel_noise)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[:] = rollout.seeded_qvel(model.nv)
    mujoco.mj_forward(model, data)
    initial = (data.qpos.tolist(), data.qvel.tolist())

    steps = rollout.steps(model.opt.timestep)
    rows = [np.concatenate(([data.time], data.qpos, data.qvel))]
    for _ in range(steps):
        mujoco.mj_step(model, data)
        rows.append(np.concatenate(([data.time], data.qpos, data.qvel)))

    run = TrajectoryRun(
        model_id=model_id,
        model_text=xml,
        rollout=rollout,
        engine=("mujoco", mujoco.__version__),
        initial=initial,
        parameters={
            "timestep_s": float(model.opt.timestep),
            "steps": steps,
            "gravity_m_s2": model.opt.gravity.tolist(),
        },
    )
    return write_trajectory_run(
        output_dir, rows, run, ud_commit=ud_commit, created_at=created_at
    )
