"""OpenSim adapter for the swing report (#12331)."""

from __future__ import annotations

from typing import ClassVar

from src.shared.python.engine_core.engine_registry import EngineType

from .base import NativeAdapter


class OpensimAdapter(NativeAdapter):
    """Maps recorded OpenSim output (body/marker kinematics) to report series."""

    engine_type: ClassVar[EngineType] = EngineType.OPENSIM
    module_name: ClassVar[str] = "opensim"
    distributions: ClassVar[tuple[str, ...]] = ("opensim",)
