"""
Configuration and parameter models for BunkerShot3D using Pydantic for DbC.
"""

from typing import NamedTuple

import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class ContactParams(NamedTuple):
    """Flat contact-material parameters (issue #6937).

    Decouples backend drivers from the nested ``config.contact_model.*``
    shape so a schema restructure does not ripple into every driver.
    """

    friction: float
    restitution: float
    youngs_modulus: float
    poisson_ratio: float


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
    """Clubhead parametric dimensions."""

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
    """Root configuration object."""

    bunker_bed: BunkerBedConfig
    grain_population: GrainPopulationConfig
    contact_model: ContactModelConfig
    clubhead: ClubheadConfig
    trajectory: TrajectoryConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BunkerShotConfig":
        """Load and validate configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    # --- Flat delegating accessors (Law of Demeter, issue #6937) ---------
    #
    # Backend drivers previously reached two levels into the nested config
    # (``config.contact_model.friction_coefficient`` etc.) ~20 times. These
    # flat accessors are the single boundary so a schema restructure (CC-4)
    # touches only this file, not every driver.

    def contact_params(self) -> ContactParams:
        """Return the contact-material parameters as a flat tuple."""
        cm = self.contact_model
        return ContactParams(
            friction=cm.friction_coefficient,
            restitution=cm.restitution_coefficient,
            youngs_modulus=cm.youngs_modulus,
            poisson_ratio=cm.poisson_ratio,
        )

    def domain_extents(self) -> tuple[float, float, float]:
        """Return the bunker-bed domain extents ``(lx, ly, lz)`` in metres."""
        d = self.bunker_bed.domain
        return (d.length_x, d.width_y, d.depth_z)

    @property
    def grain_count(self) -> int:
        """Configured grain population count."""
        return self.grain_population.count

    @property
    def grain_diameter_mean(self) -> float:
        """Mean grain diameter (m)."""
        return self.grain_population.diameter_mean

    @property
    def grain_diameter_sigma_log(self) -> float:
        """Log-normal sigma of the grain-diameter distribution."""
        return self.grain_population.diameter_sigma_log

    @property
    def grain_density(self) -> float:
        """Grain material density (kg/m^3)."""
        return self.grain_population.density

    @property
    def grain_coarse_graining_factor(self) -> float:
        """Coarse-graining factor applied to the grain population."""
        return self.grain_population.coarse_graining_factor

    @property
    def clubhead_width(self) -> float:
        """Clubhead width (m)."""
        return self.clubhead.width

    @property
    def clubhead_height(self) -> float:
        """Clubhead height (m)."""
        return self.clubhead.height

    @property
    def clubhead_mass(self) -> float:
        """Clubhead mass (kg)."""
        return self.clubhead.mass

    @property
    def output_rate_hz(self) -> float:
        """Output sampling rate (Hz)."""
        return self.output.rate_hz

    @property
    def trajectory_duration(self) -> float:
        """Configured simulation duration (s)."""
        return self.trajectory.duration

    @property
    def trajectory_file(self) -> str:
        """Configured swing-trajectory source file path."""
        return self.trajectory.file
