"""MuJoCo adapter for the swing report (#12328)."""

from __future__ import annotations

from typing import ClassVar

from src.shared.python.engine_core.engine_registry import EngineType

from .base import NativeAdapter


class MujocoAdapter(NativeAdapter):
    """Maps recorded MuJoCo output (``site_xpos`` history) to report series."""

    engine_type: ClassVar[EngineType] = EngineType.MUJOCO
    module_name: ClassVar[str] = "mujoco"
    distributions: ClassVar[tuple[str, ...]] = ("mujoco",)
