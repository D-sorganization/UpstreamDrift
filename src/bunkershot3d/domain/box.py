"""The numerical simulation domain (issue #8608, ADR-0032 decision 1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..exceptions import DomainInvariantError
from ._validate import require_positive

__all__ = ["BoundaryCondition", "DomainBox"]


class BoundaryCondition(str, Enum):
    """How the walls of the simulation box behave."""

    FIXED = "fixed"
    PERIODIC = "periodic"

    @classmethod
    def parse(cls, value: BoundaryCondition | str) -> BoundaryCondition:
        """Coerce an authored spelling to a member.

        Args:
            value: A member, or its string spelling as it appears in YAML.

        Returns:
            The matching member.

        Raises:
            DomainInvariantError: ``value`` names no implemented boundary. The
                package error rather than the enum's bare ``ValueError``, so a
                caller can catch every BunkerShot3D rejection in one clause.
        """
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in cls)
            raise DomainInvariantError(
                f"boundary must be one of {allowed}, got {value!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class DomainBox:
    """The axis-aligned box a grain-scale simulation is run inside.

    This is the *numerical container*, not the sand. The physical patch of
    bunker -- its depth, plan extents, surface slope and stance slope -- is
    :class:`~bunkershot3d.sand.bed.BunkerBedGeometry`, which also knows the USGA
    depth bands. The two are kept apart deliberately: a box can be larger than
    the sand it holds (headroom for the club, walls for the packing), and a bed
    can be modelled without any box at all by the analytic F0 solver. Do not
    add sand properties here or numerical properties there.

    Attributes:
        length_x_m: Extent along x, the swing direction.
        width_y_m: Extent along y.
        depth_z_m: Extent along z, upwards.
        boundary: Wall treatment; ``"fixed"`` or ``"periodic"``.
    """

    length_x_m: float
    width_y_m: float
    depth_z_m: float
    boundary: BoundaryCondition = BoundaryCondition.FIXED

    def __post_init__(self) -> None:
        """Validate and normalise.

        Raises:
            DomainInvariantError: An extent is not a positive finite length, or
                the boundary condition is not one this package implements.
        """
        for name in ("length_x_m", "width_y_m", "depth_z_m"):
            object.__setattr__(self, name, require_positive(getattr(self, name), name))
        object.__setattr__(self, "boundary", BoundaryCondition.parse(self.boundary))

    @property
    def extents_m(self) -> tuple[float, float, float]:
        """``(lx, ly, lz)`` in metres, in authoring order."""
        return (self.length_x_m, self.width_y_m, self.depth_z_m)

    @property
    def half_extents_m(self) -> tuple[float, float, float]:
        """Half-extents, the form most collision shapes are built from."""
        return (
            0.5 * self.length_x_m,
            0.5 * self.width_y_m,
            0.5 * self.depth_z_m,
        )

    @property
    def volume_m3(self) -> float:
        """Enclosed volume."""
        return self.length_x_m * self.width_y_m * self.depth_z_m

    @property
    def plan_area_m2(self) -> float:
        """Footprint in the x-y plane."""
        return self.length_x_m * self.width_y_m
