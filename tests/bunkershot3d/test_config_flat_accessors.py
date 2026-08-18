"""Flat config accessors + contact-material factory (issues #6936, #6937).

These pin the Law-of-Demeter boundary so backend drivers stop reaching two
levels into the nested config, and verify the chrono ``_make_contact_material``
helper is the single source of truth for SMC material setup.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import bunkershot3d.backends.chrono.driver as chrono_driver_mod
from bunkershot3d.backends.chrono.driver import ChronoDriver
from bunkershot3d.config import BunkerShotConfig, ContactParams

_YAML = """
bunker_bed:
  domain: {length_x: 2.0, width_y: 1.0, depth_z: 0.5}
  boundary: "fixed"
grain_population:
  count: 1234
  diameter_mean: 0.003
  diameter_sigma_log: 0.1
  density: 2650.0
  coarse_graining_factor: 2.0
contact_model:
  friction_coefficient: 0.42
  restitution_coefficient: 0.27
  youngs_modulus: 1.5e7
  poisson_ratio: 0.3
clubhead: {loft_deg: 56.0, bounce_deg: 10.0, width: 0.11, height: 0.06, mass: 0.31}
trajectory: {file: "swing.csv", duration: 0.07}
output: {downsample_grains: 5, rate_hz: 750.0}
"""


@pytest.fixture
def config(tmp_path: Path) -> BunkerShotConfig:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_YAML)
    return BunkerShotConfig.from_yaml(cfg)


def test_contact_params_flattens_contact_model(config: BunkerShotConfig) -> None:
    params = config.contact_params()
    assert isinstance(params, ContactParams)
    assert params.friction == 0.42
    assert params.restitution == 0.27
    assert params.youngs_modulus == 1.5e7
    assert params.poisson_ratio == 0.3


def test_domain_extents_returns_lx_ly_lz(config: BunkerShotConfig) -> None:
    assert config.domain_extents() == (2.0, 1.0, 0.5)


def test_grain_accessors(config: BunkerShotConfig) -> None:
    assert config.grain_count == 1234
    assert config.grain_diameter_mean == 0.003
    assert config.grain_diameter_sigma_log == 0.1
    assert config.grain_density == 2650.0
    assert config.grain_coarse_graining_factor == 2.0


def test_clubhead_and_output_and_trajectory_accessors(
    config: BunkerShotConfig,
) -> None:
    assert config.clubhead_width == 0.11
    assert config.clubhead_height == 0.06
    assert config.clubhead_mass == 0.31
    assert config.output_rate_hz == 750.0
    assert config.trajectory_duration == 0.07
    assert config.trajectory_file == "swing.csv"


def test_make_contact_material_is_single_source_of_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three SMC materials are built by one factory with identical params.

    Issue #6936: walls / grains / clubhead must not drift. We stub ``chrono``
    so each ``ChContactMaterialSMC()`` records its setter calls, then assert
    the helper applies exactly the four flat contact params.
    """
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_YAML)

    materials: list[MagicMock] = []

    def _make_material() -> MagicMock:
        m = MagicMock(name="ChContactMaterialSMC")
        materials.append(m)
        return m

    chrono = MagicMock(name="pychrono")
    chrono.ChContactMaterialSMC.side_effect = _make_material
    monkeypatch.setattr(chrono_driver_mod, "chrono", chrono, raising=False)
    monkeypatch.setattr(chrono_driver_mod, "_HAS_CHRONO", True)

    driver = ChronoDriver(cfg)
    mat = driver._make_contact_material()

    assert mat is materials[-1]
    mat.SetFriction.assert_called_once_with(0.42)
    mat.SetRestitution.assert_called_once_with(0.27)
    mat.SetYoungModulus.assert_called_once_with(1.5e7)
    mat.SetPoissonRatio.assert_called_once_with(0.3)
