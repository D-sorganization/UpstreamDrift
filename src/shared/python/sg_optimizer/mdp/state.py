"""MDP state primitives."""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES


@dataclass(frozen=True)
class State:
    """Ball position (yards, hole-frame) and discrete lie class."""

    x: float
    y: float
    lie: int

    def __post_init__(self) -> None:
        require(self.lie in LIE_CODES.values(), f"invalid lie code {self.lie}")
