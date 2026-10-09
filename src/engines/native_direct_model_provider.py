"""Source-bound native MuJoCo model loading for the existing direct replay loop.

This provider exposes model/data only. It has no SDK task lifecycle or stepping
implementation; resource/profile admission remains the replay executor's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.engines.native_replay_contracts import require_no_global_mujoco_callbacks


class NativeDirectModel:
    """A fresh unwrapped native model/data pair loaded from its original path."""

    def __init__(self, model_path: str) -> None:
        import mujoco as mj

        require_no_global_mujoco_callbacks(mj)
        path = Path(model_path).resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".xml":
            raise ValueError("direct native loading requires an existing MJCF XML file")
        self.model_path = str(path)
        self.model: Any = mj.MjModel.from_xml_path(self.model_path)
        self.data: Any = mj.MjData(self.model)
        require_no_global_mujoco_callbacks(mj)

    def close(self) -> None:
        """Native Python model/data ownership needs no external resource teardown."""


def create_native_direct_model(model_path: str) -> NativeDirectModel:
    """Load a fresh native pair without calling any Gym or MyoSuite hook."""
    return NativeDirectModel(model_path)
