"""Simscape Multibody (MATLAB_3D) adapter for the swing report (#12331)."""

from __future__ import annotations

from typing import ClassVar

from src.shared.python.engine_core.engine_registry import EngineType

from .base import NativeAdapter


class Matlab3DAdapter(NativeAdapter):
    """Maps recorded Simscape Multibody 3D output to report series."""

    engine_type: ClassVar[EngineType] = EngineType.MATLAB_3D
    module_name: ClassVar[str] = "matlab.engine"
    distributions: ClassVar[tuple[str, ...]] = ("matlabengine", "matlab")
