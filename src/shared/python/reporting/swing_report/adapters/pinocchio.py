"""Pinocchio adapter for the swing report (#12331)."""

from __future__ import annotations

from typing import ClassVar

from src.shared.python.engine_core.engine_registry import EngineType

from .base import NativeAdapter


class PinocchioAdapter(NativeAdapter):
    """Maps recorded Pinocchio output (frame-placement history) to report series."""

    engine_type: ClassVar[EngineType] = EngineType.PINOCCHIO
    module_name: ClassVar[str] = "pinocchio"
    distributions: ClassVar[tuple[str, ...]] = ("pin", "pinocchio")
