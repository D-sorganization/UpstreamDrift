"""Drake adapter for the swing report (#12328)."""

from __future__ import annotations

from typing import ClassVar

from src.shared.python.engine_core.engine_registry import EngineType

from .base import NativeAdapter


class DrakeAdapter(NativeAdapter):
    """Maps recorded Drake output (frame-pose history) to report series."""

    engine_type: ClassVar[EngineType] = EngineType.DRAKE
    module_name: ClassVar[str] = "pydrake"
    distributions: ClassVar[tuple[str, ...]] = ("drake",)
