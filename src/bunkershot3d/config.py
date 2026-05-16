"""
Configuration and parameter models for BunkerShot3D using Pydantic for DbC.
"""

import yaml
from pathlib import Path
from pydantic import BaseModel, Field


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
