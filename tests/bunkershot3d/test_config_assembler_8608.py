"""``BunkerShotConfig`` as a loader/assembler (issue #8608, ADR-0032 decision 1).

Replaces ``test_config_flat_accessors.py``. That file pinned the 15 flat
delegating accessors added for #6937; ADR-0032 decision 1 replaces them with
narrow value objects, because a forwarding accessor satisfies the Law of
Demeter's letter while leaving every consumer coupled to the root config.

Also retained here: the #6936 check that the Chrono driver builds all three
SMC materials through one factory, since that guarantee survives the change.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import bunkershot3d.backends.chrono.driver as chrono_driver_mod
from bunkershot3d.backends.chrono.driver import ChronoDriver
from bunkershot3d.config import BunkerShotConfig
from bunkershot3d.domain import (
    BoundaryCondition,
    ContactMaterial,
    DomainBox,
    GrainPopulation,
    SolverSettings,
    SwingCondition,
    TrajectorySource,
)
from bunkershot3d.exceptions import ConfigurationInvalidError
from bunkershot3d.geometry import DeliveryCondition

pytestmark = pytest.mark.unit

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

#: The accessors ADR-0032 decision 1 removes. Named explicitly so a future
#: convenience property cannot quietly reintroduce the god object.
REMOVED_ACCESSORS = (
    "clubhead_height",
    "clubhead_mass",
    "clubhead_width",
    "contact_params",
    "domain_extents",
    "grain_coarse_graining_factor",
    "grain_count",
    "grain_density",
    "grain_diameter_mean",
    "grain_diameter_sigma_log",
    "output_rate_hz",
    "trajectory_duration",
    "trajectory_file",
)


@pytest.fixture
def config(tmp_path: Path) -> BunkerShotConfig:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_YAML)
    return BunkerShotConfig.from_yaml(cfg)


class TestTheGodObjectIsGone:
    @pytest.mark.parametrize("name", REMOVED_ACCESSORS)
    def test_flat_delegating_accessor_is_removed(
        self, config: BunkerShotConfig, name: str
    ) -> None:
        assert not hasattr(config, name), (
            f"{name} is a forwarding accessor: it satisfies the LoD gate while "
            "keeping every consumer coupled to the root config (ADR-0032 "
            "decision 1). Return a value object instead."
        )

    def test_the_assembler_methods_are_the_public_surface(
        self, config: BunkerShotConfig
    ) -> None:
        assembled = {
            name
            for name in dir(config)
            if name.startswith("to_") and not name.startswith("to__")
        }
        assert assembled == {
            "to_contact_material",
            "to_domain_box",
            "to_grain_population",
            "to_solver_settings",
            "to_swing_condition",
            "to_trajectory_source",
        }


class TestAssembly:
    def test_domain_box(self, config: BunkerShotConfig) -> None:
        box = config.to_domain_box()
        assert isinstance(box, DomainBox)
        assert box.extents_m == (2.0, 1.0, 0.5)
        assert box.boundary is BoundaryCondition.FIXED

    def test_grain_population(self, config: BunkerShotConfig) -> None:
        grains = config.to_grain_population()
        assert isinstance(grains, GrainPopulation)
        assert grains.count == 1234
        assert grains.diameter_mean_m == 0.003
        assert grains.diameter_sigma_log == 0.1
        assert grains.density_kg_m3 == 2650.0
        assert grains.coarse_graining_factor == 2.0

    def test_contact_material(self, config: BunkerShotConfig) -> None:
        material = config.to_contact_material()
        assert isinstance(material, ContactMaterial)
        assert material.friction == 0.42
        assert material.restitution == 0.27
        assert material.youngs_modulus_pa == 1.5e7
        assert material.poisson_ratio == 0.3

    def test_solver_settings(self, config: BunkerShotConfig) -> None:
        settings = config.to_solver_settings()
        assert isinstance(settings, SolverSettings)
        assert settings.output_rate_hz == 750.0
        assert settings.downsample_grains == 5

    def test_trajectory_source(self, config: BunkerShotConfig) -> None:
        source = config.to_trajectory_source()
        assert isinstance(source, TrajectorySource)
        assert source.file == "swing.csv"
        assert source.duration_s == 0.07

    def test_swing_condition_takes_its_speed_from_the_caller(
        self, config: BunkerShotConfig
    ) -> None:
        """The legacy schema has no swing speed: it lives in the trajectory
        file. The assembler must not invent one."""
        swing = config.to_swing_condition(clubhead_speed_mps=24.0)
        assert isinstance(swing, SwingCondition)
        assert swing.clubhead_speed_mps == 24.0
        assert swing.duration_s == 0.07
        assert swing.delivery == DeliveryCondition()

    def test_swing_condition_accepts_an_explicit_delivery(
        self, config: BunkerShotConfig
    ) -> None:
        delivery = DeliveryCondition(face_open_deg=15.0, attack_angle_deg=-6.0)
        swing = config.to_swing_condition(clubhead_speed_mps=24.0, delivery=delivery)
        assert swing.delivery == delivery

    def test_assembled_objects_are_equal_across_calls(
        self, config: BunkerShotConfig
    ) -> None:
        """Value semantics: two assemblies of the same config compare equal."""
        assert config.to_domain_box() == config.to_domain_box()
        assert config.to_grain_population() == config.to_grain_population()
        assert config.to_contact_material() == config.to_contact_material()


class TestNoWedgeGeometryFromTheLegacyBlock:
    def test_the_assembler_refuses_to_synthesise_a_design_vector(
        self, config: BunkerShotConfig
    ) -> None:
        """The ``clubhead`` block has five numbers; ``WedgeGeometry`` needs
        nineteen. Filling the gap would repeat the #7999 honesty failure."""
        assert not hasattr(config, "to_wedge_geometry")


class TestTheLoaderHalf:
    """``from_yaml`` is the other half of "loader/assembler" and must fail
    with a package error, not a bare ``FileNotFoundError`` or ``TypeError``."""

    def test_a_missing_file_raises_a_package_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationInvalidError, match="not found"):
            BunkerShotConfig.from_yaml(tmp_path / "absent.yaml")

    def test_a_document_that_is_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("- one\n- two\n")
        with pytest.raises(ConfigurationInvalidError, match="mapping"):
            BunkerShotConfig.from_yaml(cfg)

    def test_an_empty_document_is_refused(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("")
        with pytest.raises(ConfigurationInvalidError, match="mapping"):
            BunkerShotConfig.from_yaml(cfg)

    def test_malformed_yaml_is_refused(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("bunker_bed: [unclosed\n")
        with pytest.raises(ConfigurationInvalidError, match="parse"):
            BunkerShotConfig.from_yaml(cfg)


class TestContactMaterialFactoryIsStillTheSingleSourceOfTruth:
    """Issue #6936, carried over: walls, grains and clubhead must not drift."""

    def test_make_contact_material_applies_exactly_the_four_parameters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
