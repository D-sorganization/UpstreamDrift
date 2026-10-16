"""Muscle-level capability registry (NSX-94, #12723).

Which engines can expose what the co-contraction experiment needs -- a
:class:`~muscle_frame.MuscleFrame` (moment arms, tension table, stiffness) -- and
with what limits.  Same fail-closed contract as
:class:`derivative_capability.DerivativeCapabilityRegistry`: asking for a missing
capability raises :class:`MuscleCapabilityUnavailable` carrying the reason; there
is no silent fall-back to a torque model.

Declared levels:

* ``opensim`` -- ``FULL``.  Millard muscles expose fibre and tendon stiffness and
  a compliant tendon, so the series formula ``k_m k_T / (k_m + k_T)`` has both
  terms from the engine.  Each model is still qualified frame by frame.
* ``myosuite`` -- ``PARTIAL``.  MyoSuite muscles are MuJoCo muscle actuators on a
  *rigid* tendon: the series element is the ``k_T -> infinity`` limit, so the
  stiffness is the fibre stiffness only and no tendon compliance exists to
  qualify.  The provider also needs a MuJoCo model with muscle actuators.
* ``mujoco`` (spec full body), ``pinocchio``, ``drake``, ``simscape`` -- ``NONE``.
  They are torque-driven; co-contraction there is a declared joint-impedance
  hypothesis with its own feedback budget (``H-DECLARED-JOINT-IMPEDANCE``), the
  subject of NSX-93 (#12722), never a muscle fibre.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    CapabilityUnavailable,
    capability_level_supported,
)
from src.shared.python.loop_nullspace.muscle_frame import TendonKind

__all__ = [
    "MuscleCapabilityDeclaration",
    "MuscleCapabilityRegistry",
    "MuscleCapabilityUnavailable",
    "default_muscle_registry",
]

_OPENSIM_LOADER = (
    "src.engines.physics_engines.opensim.python.muscle_mechanics:opensim_muscle_frame"
)
_MYOSUITE_LOADER = (
    "src.engines.physics_engines.myosuite.python.muscle_mechanics:mujoco_muscle_frame"
)


class MuscleCapabilityUnavailable(CapabilityUnavailable):
    """A muscle-level capability is absent or below the required level."""


@dataclass(frozen=True)
class MuscleCapabilityDeclaration:
    """What one engine declares about muscle-level mechanics."""

    engine: str
    level: CapabilityLevel
    tendon: TendonKind | None = None
    reason: str = ""
    loader: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.engine:
            raise ValueError("engine must be a non-empty string")
        if self.level is CapabilityLevel.NONE and not self.reason.strip():
            raise ValueError("a NONE declaration needs a reason")
        if self.level is not CapabilityLevel.NONE and self.tendon is None:
            raise ValueError("a supported declaration must name its tendon kind")
        if self.loader is not None and ":" not in self.loader:
            raise ValueError(f"loader must look like 'module:attr', got {self.loader}")


class MuscleCapabilityRegistry:
    """Per-engine muscle declarations; a missing capability fails closed."""

    def __init__(self) -> None:
        self._entries: dict[str, MuscleCapabilityDeclaration] = {}

    def declare(
        self,
        engine: str,
        level: CapabilityLevel,
        *,
        tendon: TendonKind | None = None,
        reason: str = "",
        loader: str | None = None,
    ) -> MuscleCapabilityDeclaration:
        """Register a declaration; a repeated engine name raises ``ValueError``."""
        if engine in self._entries:
            raise ValueError(f"muscle capability for {engine!r} already declared")
        decl = MuscleCapabilityDeclaration(engine, level, tendon, reason, loader)
        self._entries[engine] = decl
        return decl

    def require(
        self, engine: str, *, minimum: CapabilityLevel = CapabilityLevel.PARTIAL
    ) -> MuscleCapabilityDeclaration:
        """Return the declaration or raise :class:`MuscleCapabilityUnavailable`."""
        decl = self._entries.get(engine)
        if decl is None:
            raise MuscleCapabilityUnavailable(
                f"muscle-level mechanics for {engine!r} are not declared"
            )
        if not capability_level_supported(decl.level, minimum=minimum):
            why = f": {decl.reason}" if decl.reason else ""
            raise MuscleCapabilityUnavailable(
                f"muscle-level mechanics for {engine!r} are {decl.level.name} but "
                f"{minimum.name} is required{why}"
            )
        return decl

    def require_series_tendon(self, engine: str) -> MuscleCapabilityDeclaration:
        """Like :meth:`require` but the engine must expose a compliant tendon."""
        decl = self.require(engine)
        if decl.tendon is not TendonKind.COMPLIANT:
            kind = decl.tendon.value if decl.tendon else "no"
            raise MuscleCapabilityUnavailable(
                f"{engine!r} has a {kind} tendon: the series musculotendon "
                "stiffness needs a compliant tendon"
            )
        return decl

    def load(
        self, engine: str, *, minimum: CapabilityLevel = CapabilityLevel.PARTIAL
    ) -> Any:
        """Import and return the declared frame provider (lazy, on demand)."""
        decl = self.require(engine, minimum=minimum)
        if decl.loader is None:
            raise MuscleCapabilityUnavailable(f"{engine!r} declares no frame provider")
        module_name, _, attr = decl.loader.partition(":")
        try:
            return getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise MuscleCapabilityUnavailable(
                f"cannot import muscle frame provider {decl.loader!r} for "
                f"{engine!r}: {exc}"
            ) from exc


_TORQUE_DRIVEN = (
    "torque-driven engine: co-contraction is the declared joint-impedance "
    "hypothesis H-DECLARED-JOINT-IMPEDANCE with its own feedback budget "
    "(NSX-93, #12722), not a muscle activation fibre"
)


def default_muscle_registry() -> MuscleCapabilityRegistry:
    """OpenSim ``FULL``, MyoSuite ``PARTIAL`` (rigid tendon), the rest ``NONE``."""
    reg = MuscleCapabilityRegistry()
    reg.declare(
        "opensim",
        CapabilityLevel.FULL,
        tendon=TendonKind.COMPLIANT,
        loader=_OPENSIM_LOADER,
    )
    reg.declare(
        "myosuite",
        CapabilityLevel.PARTIAL,
        tendon=TendonKind.RIGID,
        loader=_MYOSUITE_LOADER,
    )
    reg.declare(
        "mujoco",
        CapabilityLevel.NONE,
        reason=(
            "the spec full-body MuJoCo model has no muscle actuators; the "
            "MuJoCo muscle path is the 'myosuite' entry"
        ),
    )
    for engine in ("pinocchio", "drake", "simscape"):
        reg.declare(engine, CapabilityLevel.NONE, reason=_TORQUE_DRIVEN)
    return reg
