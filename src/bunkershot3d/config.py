"""Configuration loading and assembly for BunkerShot3D (issue #8608).

``BunkerShotConfig`` is a **loader/assembler**, not a god object. It knows the
on-disk YAML schema and nothing else; its job is to produce the narrow value
objects in :mod:`bunkershot3d.domain` and then get out of the way
(ADR-0032 decision 1).

It used to carry fifteen flat delegating accessors -- ``clubhead_width``,
``grain_density``, ``output_rate_hz`` and so on -- added to satisfy the Law of
Demeter gate. Those are gone. They passed the gate while leaving every backend
coupled to the *root* config rather than to the narrow thing it needed, which
is the coupling the rule exists to prevent. Consumers now take a
:class:`~bunkershot3d.domain.DomainBox`, a
:class:`~bunkershot3d.domain.GrainPopulation`, a
:class:`~bunkershot3d.domain.ContactMaterial`, a
:class:`~bunkershot3d.domain.SolverSettings` or a
:class:`~bunkershot3d.domain.SwingCondition`.

There is deliberately **no** ``to_wedge_geometry``. The ``clubhead`` block
carries five numbers (loft, bounce, width, height, mass);
:class:`~bunkershot3d.geometry.wedge.WedgeGeometry` needs nineteen. Synthesising
the missing fourteen and presenting the result as a measured design vector is
exactly the honesty failure of issue #7999. A wedge comes from
:func:`bunkershot3d.geometry.get_preset` or from an explicit design vector.

The nested pydantic models keep the authored (unsuffixed) field names, because
they mirror the YAML documents already on disk. Unit suffixes appear the moment
a value crosses into a value object, which is where the convention in
:mod:`bunkershot3d.units` applies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .domain import (
    BoundaryCondition,
    ContactMaterial,
    DomainBox,
    GrainPopulation,
    SolverSettings,
    SwingCondition,
    TrajectorySource,
)
from .exceptions import ConfigurationInvalidError
from .geometry.delivery import DeliveryCondition

__all__ = [
    "BunkerBedConfig",
    "BunkerShotConfig",
    "ClubheadConfig",
    "ContactModelConfig",
    "DomainConfig",
    "GrainPopulationConfig",
    "OutputConfig",
    "TrajectoryConfig",
]


class DomainConfig(BaseModel):
    """Domain dimensions."""

    length_x: float = Field(..., gt=0.0, description="Domain length (m)")
    width_y: float = Field(..., gt=0.0, description="Domain width (m)")
    depth_z: float = Field(..., gt=0.0, description="Domain depth (m)")


class BunkerBedConfig(BaseModel):
    """Bunker bed setup."""

    domain: DomainConfig
    boundary: str = Field(default="fixed", pattern="^(fixed|periodic)$")


class GrainPopulationConfig(BaseModel):
    """Grain population parameters."""

    count: int = Field(..., gt=0)
    diameter_mean: float = Field(..., gt=0.0)
    diameter_sigma_log: float = Field(..., ge=0.0)
    density: float = Field(..., gt=0.0)
    coarse_graining_factor: float = Field(default=1.0, ge=1.0)


class ContactModelConfig(BaseModel):
    """Base contact parameters."""

    friction_coefficient: float = Field(..., ge=0.0, le=1.0)
    restitution_coefficient: float = Field(..., ge=0.0, le=1.0)
    youngs_modulus: float = Field(..., gt=0.0)
    poisson_ratio: float = Field(..., gt=0.0, lt=0.5)


class ClubheadConfig(BaseModel):
    """Clubhead bounding dimensions used by the F3 DEM proxies.

    This is a box, not a wedge. It is enough to build the crude collision shape
    the grain-scale backends use and no more; the design vector lives in
    :mod:`bunkershot3d.geometry`.
    """

    loft_deg: float = Field(..., ge=0.0, le=90.0)
    bounce_deg: float = Field(..., ge=-10.0, le=30.0)
    width: float = Field(..., gt=0.0)
    height: float = Field(..., gt=0.0)
    mass: float = Field(..., gt=0.0)


class OutputConfig(BaseModel):
    """Output specifications."""

    downsample_grains: int = Field(default=1, gt=0)
    rate_hz: float = Field(default=1000.0, gt=0.0)


class TrajectoryConfig(BaseModel):
    """Trajectory source configuration."""

    file: str
    duration: float = Field(default=0.1, gt=0.0, description="Simulation duration (s)")


class BunkerShotConfig(BaseModel):
    """Root configuration document, and the assembler for its value objects."""

    bunker_bed: BunkerBedConfig
    grain_population: GrainPopulationConfig
    contact_model: ContactModelConfig
    clubhead: ClubheadConfig
    trajectory: TrajectoryConfig
    output: OutputConfig

    # -- loader ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str) -> BunkerShotConfig:
        """Load and validate a configuration document.

        Args:
            path: Path to the YAML document.

        Returns:
            The validated configuration.

        Raises:
            ConfigurationInvalidError: The file is missing or unreadable, is
                not valid YAML, or does not contain a top-level mapping.
        """
        location = Path(path)
        try:
            text = location.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationInvalidError(
                f"configuration file not found or unreadable: {location}"
            ) from exc
        try:
            data: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigurationInvalidError(
                f"could not parse {location} as YAML: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ConfigurationInvalidError(
                f"{location} must contain a top-level mapping of configuration "
                f"sections, got {type(data).__name__}"
            )
        return cls(**data)

    # -- assembler -------------------------------------------------------

    def to_domain_box(self) -> DomainBox:
        """Build the numerical simulation container."""
        domain = self.bunker_bed.domain
        return DomainBox(
            length_x_m=domain.length_x,
            width_y_m=domain.width_y,
            depth_z_m=domain.depth_z,
            boundary=BoundaryCondition.parse(self.bunker_bed.boundary),
        )

    def to_grain_population(self) -> GrainPopulation:
        """Build the DEM grain population."""
        grains = self.grain_population
        return GrainPopulation(
            count=grains.count,
            diameter_mean_m=grains.diameter_mean,
            diameter_sigma_log=grains.diameter_sigma_log,
            density_kg_m3=grains.density,
            coarse_graining_factor=grains.coarse_graining_factor,
        )

    def to_contact_material(self) -> ContactMaterial:
        """Build the Hertz-Mindlin contact material."""
        contact = self.contact_model
        return ContactMaterial(
            friction=contact.friction_coefficient,
            restitution=contact.restitution_coefficient,
            youngs_modulus_pa=contact.youngs_modulus,
            poisson_ratio=contact.poisson_ratio,
        )

    def to_solver_settings(self) -> SolverSettings:
        """Build the output sampling settings."""
        output = self.output
        return SolverSettings(
            output_rate_hz=output.rate_hz,
            downsample_grains=output.downsample_grains,
        )

    def to_trajectory_source(self) -> TrajectorySource:
        """Build the prescribed-trajectory source record."""
        return TrajectorySource(
            file=self.trajectory.file,
            duration_s=self.trajectory.duration,
        )

    def to_swing_condition(
        self,
        clubhead_speed_mps: float,
        delivery: DeliveryCondition | None = None,
    ) -> SwingCondition:
        """Build the kinematic delivery for this run.

        The speed is a required argument rather than a configuration field on
        purpose: the legacy schema has no swing speed, it lives in the
        trajectory file. Defaulting one here is how the package acquired its
        hard-coded 5 m/s (defects B9/B10), a speed below the DRFT inertial
        crossover and therefore in a regime with the wrong dominant physics.

        Args:
            clubhead_speed_mps: Head speed at impact, measured from the
                trajectory the run will actually use.
            delivery: Geometric delivery (#8609). ``None`` means square and
                level -- a documented neutral reference, not a measurement.

        Returns:
            The assembled swing condition.

        Raises:
            DomainInvariantError: The speed or duration is inadmissible.
        """
        return SwingCondition(
            clubhead_speed_mps=clubhead_speed_mps,
            duration_s=self.trajectory.duration,
            delivery=delivery if delivery is not None else DeliveryCondition(),
        )
