"""Drake IK reference of a pipeline run on a migrated spec (#14727 step 2).

Owner ruling (2026-11-06): the D51 reference is the run's IK ``q_track``
itself, not a rollout of it, so it carries the face the capture's club triad
observed instead of drifting off it.  The track is trunk-migrated from the
run's v1 source spec onto the target spec; the impact sample is the track
sample nearest the run's scheduled ball impact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.full_body_migration import TrunkMigrator

Array = NDArray[np.float64]


@dataclass(frozen=True)
class DrakeReference:
    """IK track on the target spec: ``q[k]`` at ``time_s[k]`` in spec order."""

    time_s: Array
    q: Array
    coordinate_order: tuple[str, ...]
    impact_index: int


def drake_v2_reference(
    run_dir: Path, v1_spec_bytes: bytes, v2_spec_bytes: bytes
) -> DrakeReference:
    """Migrate ``run_dir``'s IK ``q_track`` from the v1 onto the v2 spec.

    Raises:
        ValueError: the track's width differs from the v1 coordinate order.
    """
    v1, v2 = json.loads(v1_spec_bytes), json.loads(v2_spec_bytes)
    with np.load(run_dir / "dynamics_record.npz") as record:
        time_s = np.asarray(record["track_time_s"], dtype=float)
        q_track = np.asarray(record["q_track"], dtype=float)
    order = tuple(v1["coordinate_order"])
    if q_track.ndim != 2 or q_track.shape[1] != len(order):
        raise ValueError(
            f"q_track {q_track.shape} does not match {len(order)} v1 coordinates"
        )
    q = TrunkMigrator(v1, v2).migrate_coordinates(q_track, order)
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    impact_s = float(receipt["dynamics"]["ball_impact"]["t_start_s"])
    return DrakeReference(
        time_s=time_s,
        q=np.asarray(q, dtype=float),
        coordinate_order=tuple(v2["coordinate_order"]),
        impact_index=int(np.argmin(np.abs(time_s - impact_s))),
    )


def impact_conditions(spec_bytes: bytes, time_s: Array, q: Array, index: int) -> Any:
    """Face, path, AoA and speed of the clubhead at sample ``index``.

    ``q`` is in ``spec_bytes``' coordinate order; velocities are the central
    differences of ``q`` on ``time_s``.
    """
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.impact_parameters import TargetFrame
    from src.shared.python.impact_parameters.adapters import club_face_spec_from_spec
    from src.shared.python.impact_parameters.conditions import Rollout, evaluate_rollout

    xml, meta = export_full_body_mjcf(spec_bytes)
    model = mujoco.MjModel.from_xml_string(xml)
    joints = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in meta["coordinate_order"]
    ]
    qpos = np.tile(model.qpos0, (len(time_s), 1))
    qvel = np.zeros((len(time_s), model.nv))
    qpos[:, model.jnt_qposadr[joints]] = q
    qvel[:, model.jnt_dofadr[joints]] = np.gradient(q, time_s, axis=0)
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"]["Clubhead"]
    )
    body = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site])
    )
    return evaluate_rollout(
        model,
        Rollout(time_s, qpos, qvel),
        body,
        TargetFrame(),
        spec=club_face_spec_from_spec(json.loads(spec_bytes)),
        impact_index=index,
    )
