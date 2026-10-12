"""Shared adapter machinery: one implementation, per-engine attributes (DRY)."""

from __future__ import annotations

import importlib
from typing import ClassVar, Protocol, runtime_checkable

from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.motion_matching.provenance import engine_package_version

from ..adapters_base import (
    CLUBHEAD_SPEED,
    EngineResult,
    QuantityData,
    finite_or_none,
)
from .native import NativeSwingRun, derive_speed

CLUBHEAD_POSITION = "clubhead_position"
KINEMATIC_SEQUENCE = "kinematic_sequence"
WORLD_FRAME = "world_Zup"
_AXES = {"x": 0, "y": 1, "z": 2}


@runtime_checkable
class EngineResultAdapter(Protocol):
    """Anything that turns a recorded native run into an ``EngineResult``."""

    engine_type: ClassVar[EngineType]

    def adapt(self, native: NativeSwingRun, run_id: str) -> EngineResult: ...

    def unavailable(self, run_id: str, reason: str) -> EngineResult: ...


class NativeAdapter:
    """Maps a ``NativeSwingRun`` to the advertised report quantities.

    Subclasses only set ``engine_type``, ``module_name`` and ``distributions``.
    ``clubhead_position`` is carried as one scalar channel (``position_axis``,
    default world ``z``) because a contract series is scalar-valued.
    """

    engine_type: ClassVar[EngineType]
    module_name: ClassVar[str]
    distributions: ClassVar[tuple[str, ...]]

    def __init__(self, position_axis: str = "z") -> None:
        if position_axis not in _AXES:
            raise ValueError(f"position_axis must be one of {sorted(_AXES)}")
        self.position_axis = position_axis

    def engine_version(self, recorded: str | None = None) -> str:
        """Recorded version if given, else probe the installed package."""
        if recorded:
            return recorded
        try:
            module = importlib.import_module(self.module_name)
        except ImportError:
            module = None
        return engine_package_version(module, *self.distributions)

    def adapt(self, native: NativeSwingRun, run_id: str) -> EngineResult:
        """Contract: missing samples stay ``None``; only advertised ids emitted."""
        times = tuple(float(t) for t in native.time_s)
        axis = _AXES[self.position_axis]
        quantities: dict[str, QuantityData] = {
            CLUBHEAD_POSITION: QuantityData(
                units="m",
                frame=WORLD_FRAME,
                time_s=times,
                values=tuple(
                    finite_or_none(p[axis]) for p in native.clubhead_position_m
                ),
                channel=self.position_axis,
            )
        }
        speed = derive_speed(native)
        if speed is not None:
            quantities[CLUBHEAD_SPEED] = QuantityData(
                units="m/s",
                frame=WORLD_FRAME,
                time_s=times,
                values=tuple(finite_or_none(v) for v in speed),
            )
        if native.kinematic_sequence is not None:
            quantities[KINEMATIC_SEQUENCE] = native.kinematic_sequence
        return EngineResult(
            engine=self.engine_type.value,
            run_id=run_id,
            engine_version=self.engine_version(native.engine_version),
            model_file_hash=native.model_file_hash,
            run_class=native.run_class,
            quantities=quantities,
            force_provenance=native.force_provenance,
        )

    def unavailable(self, run_id: str, reason: str) -> EngineResult:
        """A recorded-but-failed run: no quantities, builder notes the reason."""
        return EngineResult(
            engine=self.engine_type.value,
            run_id=run_id,
            engine_version=self.engine_version(),
            model_file_hash=None,
            run_class="forward_dynamics",
            error=reason,
        )
