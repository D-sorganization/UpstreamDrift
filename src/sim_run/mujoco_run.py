"""MuJoCo run path that writes its trajectory and a run manifest (Issue #14432).

:func:`run_mujoco` compiles an MJCF model, rolls it out with zero controls from
``qpos0`` and a seeded initial velocity, writes the trajectory to
``trajectory.npy`` and emits ``run_manifest.json`` beside it through the shared
emitter. The model revision is the SHA-256 of the MJCF text, the input digest
identifies the initial condition, and the manifest records the trajectory hash.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from src.sim_run.manifest import RunManifest
from src.sim_run.manifest_emitter import emit_run_manifest, sha256_json
from src.sim_run.worker import get_repo_sha

TRAJECTORY_FILENAME = "trajectory.npy"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    _require(duration > 0, f"duration must be positive, got {duration}")
    _require(seed >= 0, f"seed must be non-negative, got {seed}")
    _require(qvel_noise >= 0, f"qvel_noise must be non-negative, got {qvel_noise}")
    commit = ud_commit or get_repo_sha()
    _require(commit != "0" * 40, "ud_commit could not be resolved")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[:] = np.random.default_rng(seed).normal(0.0, qvel_noise, model.nv)
    mujoco.mj_forward(model, data)
    initial = {"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist()}

    steps = max(1, round(duration / model.opt.timestep))
    rows = [np.concatenate(([data.time], data.qpos, data.qvel))]
    for _ in range(steps):
        mujoco.mj_step(model, data)
        rows.append(np.concatenate(([data.time], data.qpos, data.qvel)))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trajectory = out / TRAJECTORY_FILENAME
    np.save(trajectory, np.asarray(rows))

    return emit_run_manifest(
        out,
        outputs=[trajectory],
        model_id=model_id,
        revision="sha256:" + hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        parameters={
            "duration_s": duration,
            "timestep_s": float(model.opt.timestep),
            "steps": steps,
            "gravity_m_s2": model.opt.gravity.tolist(),
            "controls": "zero",
            "qvel_noise": qvel_noise,
            "trajectory_columns": {"time": 1, "qpos": model.nq, "qvel": model.nv},
        },
        input_sha256=sha256_json(initial),
        seed=seed,
        engine_name="mujoco",
        engine_version=mujoco.__version__,
        ud_commit=commit,
        created_at=created_at,
        input_label="initial condition: qpos0 with seeded qvel",
    )
