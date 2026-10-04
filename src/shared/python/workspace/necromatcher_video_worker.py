"""Clean SDK worker for an owned source-overlay export request."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

import mujoco  # noqa: F401 -- native SDK loaded before workspace rendering helpers
import numpy as np

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
    MujocoForceTorqueSource,
)
from src.shared.python.force_overlay.contracts import ForceTorqueFrame
from .artifact_handoff import compute_file_sha256
from .necromatcher import NecromatcherLibrary
from .necromatcher_fit_jobs import fit_execution_stamp
from .necromatcher_native import NativeFitBinding
from .necromatcher_video import export_fit_video
from .necromatcher_video_forces import ForceLayer

logger = logging.getLogger(__name__)


class MujocoFitSampler:
    """Sample MuJoCo wrenches for one fitted state (q, v, a) of a bound native model."""

    def __init__(self, binding: NativeFitBinding) -> None:
        model = binding.plant.adapter.model
        self._model = model
        self._data = mujoco.MjData(model)
        self._source = MujocoForceTorqueSource(model)
        joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in binding.plant.coordinate_order
        ]
        if min(joints) < 0:
            raise ValueError("Native model lacks a declared coordinate joint")
        self._qpos = model.jnt_qposadr[joints]
        self._dof = model.jnt_dofadr[joints]

    def sample(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray, time_s: float
    ) -> ForceTorqueFrame:
        """Apply the inverse-dynamics generalized force so MuJoCo replays ``a``."""
        data = self._data
        data.time = float(time_s)
        data.qpos[self._qpos] = q
        data.qvel[self._dof] = v
        data.qacc[self._dof] = a
        data.qfrc_applied[:] = 0.0
        mujoco.mj_inverse(self._model, data)
        data.qfrc_applied[:] = data.qfrc_inverse
        return self._source.sample(data)


def mujoco_force_sampler(binding: NativeFitBinding) -> MujocoFitSampler:
    return MujocoFitSampler(binding)


def execute(request_path: Path) -> dict[str, Any]:
    """Validate a local request and export only under its owned run directory."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if (
        not isinstance(request, dict)
        or request.get("kind") != "necromatcher/video-job/1"
        or request_path.name != "request.json"
        or request_path.parent.name != request.get("run_id")
    ):
        raise ValueError("Invalid owned video worker request")
    library = NecromatcherLibrary(request["library_root"])
    if (
        request_path.resolve()
        != (library.root / "video-runs" / request["run_id"] / "request.json").resolve()
    ):
        raise ValueError("Video request lies outside its owned library run")
    stamp = fit_execution_stamp()
    expected = request["execution_stamp"]
    for key in ("source_sha256", "runtime_sha256"):
        if stamp[key] != expected[key]:
            raise ValueError("Video worker source or runtime differs from launch")
    if (
        library.load_asset(request["source_fit_id"]).metadata["hash"]
        != request["source_fit_hash"]
    ):
        raise ValueError("Video source fit changed before rendering")
    fit = library.load_fit(request["source_fit_id"])
    for kind in ("model", "capture"):
        if (
            fit[f"{kind}_id"] != request[f"{kind}_id"]
            or fit[f"{kind}_hash"] != request[f"{kind}_hash"]
        ):
            raise ValueError("Video parent differs from launch")
    destination = request_path.parent / "overlay"
    layer_arguments: dict[str, Any] = (
        {
            "force_layer": ForceLayer(**request["force_layer"]),
            "force_sampler_factory": mujoco_force_sampler,
        }
        if request.get("force_layer")
        else {}
    )
    export_fit_video(
        library,
        request["source_fit_id"],
        destination,
        selected_frames=tuple(request["selected_frames"]),
        **layer_arguments,
    )
    current = fit_execution_stamp()
    if any(
        current[key] != expected[key] for key in ("source_sha256", "runtime_sha256")
    ):
        raise ValueError("Video worker source or runtime changed during export")
    return {
        "manifest_sha256": compute_file_sha256(
            destination / "manifest.json"
        ).removeprefix("sha256:")
    }


def main() -> None:
    try:
        result = execute(Path(sys.argv[1]))
        sys.stdout.write(json.dumps(result, allow_nan=False))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        ImportError,
    ):
        logger.exception("Owned native video export failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
