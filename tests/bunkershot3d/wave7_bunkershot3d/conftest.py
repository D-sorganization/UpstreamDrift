"""Shared fixtures for wave7 bunkershot3d coverage tests."""

from __future__ import annotations

from pathlib import Path

import pytest

CANONICAL_YAML = """
bunker_bed:
  domain:
    length_x: 2.0
    width_y: 1.0
    depth_z: 0.5
  boundary: "fixed"
grain_population:
  count: 100
  diameter_mean: 0.002
  diameter_sigma_log: 0.1
  density: 2650.0
  coarse_graining_factor: 1.0
contact_model:
  friction_coefficient: 0.5
  restitution_coefficient: 0.3
  youngs_modulus: 1.0e7
  poisson_ratio: 0.25
clubhead:
  loft_deg: 56.0
  bounce_deg: 10.0
  width: 0.1
  height: 0.05
  mass: 0.3
trajectory:
  file: "swing_data.csv"
  duration: 0.05
output:
  downsample_grains: 1
  rate_hz: 500.0
"""


@pytest.fixture
def bunker_config_path(tmp_path: Path) -> Path:
    """A valid BunkerShotConfig YAML written to disk."""
    p = tmp_path / "bunker.yaml"
    p.write_text(CANONICAL_YAML)
    return p
