"""Joint angles shared by hand visual backends (epic #15150).

Angles are anatomical hinge flexion in radians, keyed by the joint's distal
bone: ``index_prox`` is the index knuckle (MCP), ``index_mid`` the PIP,
``index_dist`` the DIP; ``meta_ring`` is the ring metacarpal's arch;
``thumb_prox``/``thumb_dist`` are the thumb MCP and IP and ``thumb_press`` the
thumb base pressing onto the grip. Zero is a straight digit; positive curls
toward the palm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

HandSide = Literal["lead", "trail"]
FINGERS: tuple[str, ...] = ("index", "middle", "ring", "little")


@dataclass(frozen=True)
class HandJointAngles:
    """An immutable set of hinge angles (radians) for one hand."""

    hinges: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.hinges]
        if len(set(names)) != len(names):
            raise ValueError("duplicate joint in HandJointAngles")
        for name, angle in self.hinges:
            if not isinstance(name, str) or not math.isfinite(angle):
                raise ValueError(f"joint angle {name!r} must be finite")

    @classmethod
    def from_mapping(cls, angles: Mapping[str, float]) -> HandJointAngles:
        return cls(tuple(sorted((str(k), float(v)) for k, v in angles.items())))

    def as_mapping(self) -> dict[str, float]:
        return dict(self.hinges)

    def flexion_deg(self, joint: str) -> float:
        """Angle of ``joint`` in degrees. Raises ``KeyError`` if absent."""
        return math.degrees(self.as_mapping()[joint])


__all__ = ["FINGERS", "HandJointAngles", "HandSide"]
