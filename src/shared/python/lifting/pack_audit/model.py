"""Data model shared by the engine adapters and the baseline report."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .names import LIFTS
from .packs import PackLocation


@dataclass(frozen=True)
class Anthropometry:
    """Reference lifter and load used to generate every pack model.

    Defaults: 80 kg, 1.78 m, men's 20 kg bar loaded with 100 kg of plates
    (50 kg per side), i.e. 120 kg on the bar.
    """

    body_mass_kg: float = 80.0
    height_m: float = 1.78
    plate_mass_per_side_kg: float = 50.0

    def __post_init__(self) -> None:
        for name in ("body_mass_kg", "height_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if (
            not math.isfinite(self.plate_mass_per_side_kg)
            or self.plate_mass_per_side_kg < 0
        ):
            raise ValueError("plate_mass_per_side_kg must be >= 0 and finite")

    @property
    def bar_total_mass_kg(self) -> float:
        """Bar plus plates, using the men's 20 kg bar."""
        return 20.0 + 2.0 * self.plate_mass_per_side_kg


@dataclass(frozen=True)
class CoordinateInfo:
    """One generalized coordinate as the engine reports it.

    ``zero_offset`` is the native value of canonical angle zero (MuJoCo's
    ``ref``); the canonical angle is ``native - zero_offset``.
    """

    name: str
    lower: float | None
    upper: float | None
    default: float | None
    kind: str
    zero_offset: float = 0.0


@dataclass
class PoseEval:
    """Canonical-frame result of evaluating one configuration."""

    positions: dict[str, np.ndarray]
    com: np.ndarray
    total_mass: float
    closure: dict[str, float] | None = None
    notes: list[str] = field(default_factory=list)
    sole_z: float | None = None


def require_lift(lift: str) -> None:
    """Raise ``ValueError`` unless *lift* is one of the audited lifts."""
    if lift not in LIFTS:
        raise ValueError(f"unknown lift {lift!r}; use {list(LIFTS)}")


class EngineAdapter(ABC):
    """Uniform read-only view of one lift model loaded in its own engine."""

    engine: str = ""
    frame: str = "z_up"

    def __init__(self, pack: PackLocation, lift: str, anthro: Anthropometry) -> None:
        require_lift(lift)
        self.pack = pack
        self.lift = lift
        self.anthro = anthro

    @abstractmethod
    def model_text(self) -> str:
        """The generated model file content (MJCF, OSIM, SDF or URDF)."""

    @abstractmethod
    def coordinates(self) -> list[CoordinateInfo]:
        """All generalized coordinates with limits and the pack default."""

    @abstractmethod
    def structure(self) -> dict[str, Any]:
        """Bodies, DOF, root joint, barbell attachment, contacts, masses."""

    @abstractmethod
    def segment_masses(self) -> dict[str, float]:
        """Mass of every link by body name."""

    @abstractmethod
    def evaluate(self, q: Mapping[str, float] | None) -> PoseEval:
        """FK/CoM at canonical angles *q* (pelvis at (0, 0, 1)), or the pack start."""

    def start_contact_force(self) -> dict[str, Any]:
        """Vertical ground-contact force at the pack start pose, if obtainable.

        Unavailable is never zero: the default reports ``value_n=None`` with
        the reason the engine/pack cannot supply it.
        """
        return {"value_n": None, "reason": "not implemented for this engine"}

    @abstractmethod
    def smoke_step(self) -> dict[str, Any]:
        """Step the model from its start pose; report finiteness, not physics."""
