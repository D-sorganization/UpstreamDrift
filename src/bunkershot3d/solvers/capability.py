"""Fail-closed sand-motion capability register (issue #9688, ADR-0044).

Why this module exists
----------------------
ADR-0044 is still Proposed. No current pathway produces genuine 3-D sand-grain
motion:

* **F0** (``solvers/drft.py``) is Dynamic Resistive Force Theory: an analytic
  force model with no transported sand particles.
* **F1** (``solvers/mpm/``) is 2-D plane strain: its particles are continuum
  material points on a 2-D slice, and its ball is an infinite cylinder.
* **sandvolume_extruded** (``src/tools/bunker_shot_gui/sandvolume.py``) extrudes
  that 2-D plane across an assumed width: repeating identical slices does not
  add a resolved physical dimension.
* **backends.mpm_proxy** (``backends/mpm/driver.py``) is a capped MuJoCo sphere
  proxy (writes head pose and contact wrench; no grain trajectories).
* **backends.chrono** (``backends/chrono/driver.py``) is coarse-grained DEM debt
  exercised only against mocks.
* **tracers** are decorative rendering graphics with no equations of motion.

This module provides a machine-checkable capability register so that no caller
can present F0, F1, proxy or decorative tracer output as genuine 3-D
individual-grain trajectories or spherical ball spin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .exceptions import CapabilityError
from .protocol import FidelityTier

__all__ = [
    "SAND_MOTION_CAPABILITIES",
    "CapabilityError",
    "SandMotionCapability",
    "SandMotionKind",
    "capability",
    "require_ball_spin_3d",
    "require_grain_trajectories_3d",
    "require_physical",
]


class SandMotionKind(StrEnum):
    """Classification of sand-motion representation."""

    DISCRETE_GRAINS = "discrete_grains"
    COARSE_GRAINED_DEM = "coarse_grained_dem"
    CONTINUUM_MATERIAL_POINTS = "continuum_material_points"
    EULERIAN_FIELD = "eulerian_field"
    DECORATIVE_TRACERS = "decorative_tracers"
    NONE = "none"


@dataclass(frozen=True)
class SandMotionCapability:
    """The capability profile of one simulation or rendering pathway.

    Attributes:
        pathway: Identifier for the pathway or FidelityTier member.
        kind: Classification of how sand motion is represented.
        spatial_dims: Number of resolved spatial dimensions (0, 2, or 3).
        grain_trajectories_3d: Whether the pathway produces genuine 3-D
            individual-grain trajectories.
        ball_spin_3d: Whether the pathway resolves 3-D spherical ball spin.
        physical: Whether the pathway models physical equations rather than
            decorative graphics.
        notes: Documentation of assumptions, limitations, and citations.
    """

    pathway: str
    kind: SandMotionKind
    spatial_dims: int
    grain_trajectories_3d: bool
    ball_spin_3d: bool
    physical: bool
    notes: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SandMotionKind):
            object.__setattr__(self, "kind", SandMotionKind(self.kind))

        if self.spatial_dims not in (0, 2, 3):
            raise ValueError(
                f"spatial_dims must be in {{0, 2, 3}}, got {self.spatial_dims!r}"
            )

        if self.kind is SandMotionKind.NONE:
            if self.spatial_dims != 0:
                raise ValueError(
                    f"SandMotionKind.NONE requires spatial_dims == 0, got {self.spatial_dims!r}"
                )
            if self.grain_trajectories_3d:
                raise ValueError(
                    "SandMotionKind.NONE requires no trajectories (grain_trajectories_3d=False)"
                )

        if self.grain_trajectories_3d:
            if self.kind not in (
                SandMotionKind.DISCRETE_GRAINS,
                SandMotionKind.COARSE_GRAINED_DEM,
            ):
                raise ValueError(
                    "grain_trajectories_3d=True requires kind in "
                    "{SandMotionKind.DISCRETE_GRAINS, SandMotionKind.COARSE_GRAINED_DEM}, "
                    f"got {self.kind.value!r}"
                )
            if self.spatial_dims != 3:
                raise ValueError(
                    f"grain_trajectories_3d=True requires spatial_dims == 3, got {self.spatial_dims!r}"
                )

        if self.kind is SandMotionKind.DECORATIVE_TRACERS and self.physical:
            raise ValueError(
                "DECORATIVE_TRACERS requires physical is False; visual tracers "
                "carry no physical equations of motion"
            )


_CAPABILITIES: dict[str, SandMotionCapability] = {
    FidelityTier.F0.value: SandMotionCapability(
        pathway=FidelityTier.F0.value,
        kind=SandMotionKind.NONE,
        spatial_dims=0,
        grain_trajectories_3d=False,
        ball_spin_3d=False,
        physical=True,
        notes=(
            "Dynamic Resistive Force Theory (ADR-0032); continuum force "
            "model with no transported sand grains or resolved trajectories"
        ),
    ),
    FidelityTier.F1.value: SandMotionCapability(
        pathway=FidelityTier.F1.value,
        kind=SandMotionKind.CONTINUUM_MATERIAL_POINTS,
        spatial_dims=2,
        grain_trajectories_3d=False,
        ball_spin_3d=False,
        physical=True,
        notes=(
            "2-D plane-strain continuum material point method (ADR-0033); "
            "no out-of-plane flow, cylinder ball approximation, no individual "
            "grain trajectories"
        ),
    ),
    FidelityTier.F2.value: SandMotionCapability(
        pathway=FidelityTier.F2.value,
        kind=SandMotionKind.CONTINUUM_MATERIAL_POINTS,
        spatial_dims=3,
        grain_trajectories_3d=False,
        ball_spin_3d=False,
        physical=True,
        notes=(
            "Material Point Method, GPU reference truth (ADR-0032); 3-D "
            "continuum material points, no discrete grain trajectories or "
            "local CPU execution pathway"
        ),
    ),
    FidelityTier.F3.value: SandMotionCapability(
        pathway=FidelityTier.F3.value,
        kind=SandMotionKind.DISCRETE_GRAINS,
        spatial_dims=3,
        grain_trajectories_3d=False,
        ball_spin_3d=False,
        physical=True,
        notes=(
            "Discrete Element Method, grain-scale studies only (ADR-0032); "
            "no validated 3-D whole-shot production solver (#9688)"
        ),
    ),
    "sandvolume_extruded": SandMotionCapability(
        pathway="sandvolume_extruded",
        kind=SandMotionKind.CONTINUUM_MATERIAL_POINTS,
        spatial_dims=2,
        grain_trajectories_3d=False,
        ball_spin_3d=False,
        physical=True,
        notes=(
            "Extruded 2-D plane-strain continuum sheets (PR #9184); extrusion "
            "across assumed width does not add a resolved dimension (#9688)"
        ),
    ),
    "backends.mpm_proxy": SandMotionCapability(
        pathway="backends.mpm_proxy",
        kind=SandMotionKind.NONE,
        spatial_dims=0,
        grain_trajectories_3d=False,
        ball_spin_3d=False,
        physical=True,
        notes=(
            "Capped MuJoCo rigid-sphere contact proxy; writes head pose "
            "and contact wrench, no grain trajectories or sand continuum (#9688)"
        ),
    ),
    "backends.chrono": SandMotionCapability(
        pathway="backends.chrono",
        kind=SandMotionKind.COARSE_GRAINED_DEM,
        spatial_dims=3,
        grain_trajectories_3d=False,
        ball_spin_3d=False,
        physical=True,
        notes=(
            "Project Chrono coarse-grained DEM driver; grain_trajectories_3d=False "
            "until a real-backend smoke exists (#9688 / ADR-0044)"
        ),
    ),
    "tracers": SandMotionCapability(
        pathway="tracers",
        kind=SandMotionKind.DECORATIVE_TRACERS,
        spatial_dims=3,
        grain_trajectories_3d=False,
        ball_spin_3d=False,
        physical=False,
        notes=(
            "Decorative visual tracers for rendering; non-physical with no "
            "resolved equations of motion (#9688 / ADR-0044)"
        ),
    ),
}

SAND_MOTION_CAPABILITIES: Mapping[str, SandMotionCapability] = MappingProxyType(
    _CAPABILITIES
)


def capability(pathway: str | FidelityTier) -> SandMotionCapability:
    """Return the capability record for a simulation or visualisation pathway.

    Args:
        pathway: Pathway identifier or FidelityTier member.

    Returns:
        The immutable SandMotionCapability record.

    Raises:
        CapabilityError: If the pathway is unknown, listing all known ids.
    """
    key = pathway.value if isinstance(pathway, FidelityTier) else str(pathway)
    if key not in SAND_MOTION_CAPABILITIES:
        known = ", ".join(sorted(SAND_MOTION_CAPABILITIES.keys()))
        raise CapabilityError(f"Unknown pathway {key!r}; known pathways: {known}")
    return SAND_MOTION_CAPABILITIES[key]


def _require(
    pathway: str | FidelityTier, flag: str, claim: str
) -> SandMotionCapability:
    """Return the record for ``pathway`` if its boolean ``flag`` is set, else refuse.

    Raises:
        CapabilityError: naming the pathway, its kind, the claim and '#9688 / ADR-0044'.
    """
    cap = capability(pathway)
    if not getattr(cap, flag):
        raise CapabilityError(
            f"Pathway {cap.pathway!r} ({cap.kind.value}) {claim} "
            f"(#9688 / ADR-0044):\n  {cap.notes}"
        )
    return cap


def require_grain_trajectories_3d(
    pathway: str | FidelityTier,
) -> SandMotionCapability:
    """Require genuine 3-D individual-grain trajectories; raise CapabilityError otherwise."""
    return _require(
        pathway,
        "grain_trajectories_3d",
        "does not produce genuine 3-D individual-grain trajectories",
    )


def require_ball_spin_3d(pathway: str | FidelityTier) -> SandMotionCapability:
    """Require resolved 3-D spherical ball spin; raise CapabilityError otherwise."""
    return _require(pathway, "ball_spin_3d", "does not resolve 3-D spherical ball spin")


def require_physical(pathway: str | FidelityTier) -> SandMotionCapability:
    """Require physical equations of motion (not decorative graphics); raise otherwise."""
    return _require(pathway, "physical", "is not a physical simulation model")
