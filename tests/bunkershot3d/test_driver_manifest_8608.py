"""Every driver attaches a run manifest to its result file (issue #8608, W1).

#8617 gave ``BunkerShotResultWriter`` a ``manifest=`` argument and a
``set_manifest()`` setter, but no driver passed one, so every result on disk
carried ``schema_version`` and no provenance: no config hash, no seeds, no
library versions, no fidelity tier, no validity verdict. A result that cannot
be traced to its inputs is not evidence (finding B18).

The RNG seeds recorded here are the ones the drivers actually use. They are
fixed literals and a ``PCG64`` stream (``np.random.default_rng``), and the
manifest says exactly that rather than claiming the ``PCG64DXSM`` discipline of
:mod:`bunkershot3d.provenance.rng`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _bunker_fixtures_8612 import write_config, write_straight_trajectory

from bunkershot3d.backends.chrono.driver import ChronoDriver
from bunkershot3d.backends.liggghts.driver import LiggghtsDriver
from bunkershot3d.backends.mpm.driver import MPMDriver
from bunkershot3d.io.schema import BunkerShotResultReader
from bunkershot3d.provenance import RunManifest, Validity, config_hash, physics_hash

pytestmark = pytest.mark.unit


_DRIVERS = (ChronoDriver, LiggghtsDriver, MPMDriver)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    write_straight_trajectory(tmp_path / "swing_data.csv", speed=1.0, duration=0.02)
    return write_config(
        tmp_path / "canonical.yaml",
        grain_count=10,
        diameter_mean=0.01,
        diameter_sigma_log=0.1,
        duration=0.005,
        rate_hz=1000.0,
        trajectory_file="swing_data.csv",
    )


@pytest.mark.parametrize("driver_cls", _DRIVERS, ids=lambda c: c.__name__)
class TestEveryDriverBuildsAManifest:
    def test_run_manifest_is_a_run_manifest(
        self, driver_cls: type, config_path: Path
    ) -> None:
        manifest = driver_cls(config_path).run_manifest()
        assert isinstance(manifest, RunManifest)

    def test_hashes_match_the_configuration(
        self, driver_cls: type, config_path: Path
    ) -> None:
        driver = driver_cls(config_path)
        manifest = driver.run_manifest()
        assert manifest.config_hash == config_hash(driver.config)
        assert manifest.physics_hash == physics_hash(driver.config)

    def test_at_least_one_seed_is_recorded(
        self, driver_cls: type, config_path: Path
    ) -> None:
        assert driver_cls(config_path).run_manifest().seeds

    def test_no_seed_claims_the_provenance_module_discipline(
        self, driver_cls: type, config_path: Path
    ) -> None:
        """These drivers do not use :mod:`bunkershot3d.provenance.rng`, so no
        seed may claim its ``PCG64DXSM`` generator."""
        for seed in driver_cls(config_path).run_manifest().seeds:
            assert seed.generator != "PCG64DXSM"
            assert seed.generator

    def test_the_fidelity_tier_is_f3(self, driver_cls: type, config_path: Path) -> None:
        """ADR-0032: Chrono, LIGGGHTS and the MuJoCo proxy are all F3."""
        assert driver_cls(config_path).run_manifest().fidelity_tier == "F3"

    def test_the_verdict_is_not_valid_and_says_why(
        self, driver_cls: type, config_path: Path
    ) -> None:
        """These tiers are intractable at true grain scale, so a result from
        them must not be labelled a valid answer (ADR-0032)."""
        manifest = driver_cls(config_path).run_manifest()
        assert manifest.validity is not Validity.VALID
        assert manifest.validity_reason.strip()

    def test_library_versions_are_captured(
        self, driver_cls: type, config_path: Path
    ) -> None:
        assert "numpy" in driver_cls(config_path).run_manifest().library_versions

    def test_the_solver_is_named(self, driver_cls: type, config_path: Path) -> None:
        assert driver_cls(config_path).run_manifest().solver


class TestSeedsNameTheirRealGenerator:
    """A manifest that names the wrong generator is worse than none."""

    def test_numpy_seeded_drivers_record_pcg64(self, config_path: Path) -> None:
        for driver_cls in (ChronoDriver, MPMDriver):
            for seed in driver_cls(config_path).run_manifest().seeds:
                assert seed.generator == "PCG64", (
                    "these drivers seed np.random.default_rng, which is PCG64"
                )

    def test_liggghts_records_the_lammps_generator(self, config_path: Path) -> None:
        """LIGGGHTS deck seeds drive LAMMPS' own Park/Miller generator, not
        numpy, so they must not be labelled a numpy stream."""
        seeds = LiggghtsDriver(config_path).run_manifest().seeds
        assert seeds
        for seed in seeds:
            assert seed.generator == "LIGGGHTS-RanPark"


class TestSolverNames:
    @pytest.mark.parametrize(
        ("driver_cls", "expected"),
        [
            (ChronoDriver, "chrono"),
            (LiggghtsDriver, "liggghts"),
            (MPMDriver, "mujoco"),
        ],
        ids=lambda value: getattr(value, "__name__", value),
    )
    def test_solver_identifier(
        self, driver_cls: type, expected: str, config_path: Path
    ) -> None:
        assert driver_cls(config_path).run_manifest().solver == expected


class TestTheManifestReachesTheFile:
    """The wiring, not just the builder: a written result must carry it."""

    def _write_dump(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for step in (0, 200):
                atoms = [
                    [0.1 * j, 0.05 * j, 0.02 * j, 0.0, 0.0, -0.01] for j in range(3)
                ]
                f.write(f"ITEM: TIMESTEP\n{step}\n")
                f.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0.0 2.0\n0.0 1.0\n0.0 0.5\n")
                f.write("ITEM: ATOMS id type x y z vx vy vz\n")
                f.writelines(
                    f"{i} 1 " + " ".join(f"{v:.4f}" for v in row) + "\n"
                    for i, row in enumerate(atoms, start=1)
                )

    def test_liggghts_dump_conversion_carries_provenance(
        self, config_path: Path, tmp_path: Path
    ) -> None:
        driver = LiggghtsDriver(config_path)
        self._write_dump(tmp_path / "dump.bunkershot")
        output = tmp_path / "result.h5"
        driver._parse_and_write(tmp_path, output)

        with BunkerShotResultReader(output) as reader:
            manifest = reader.manifest
        assert manifest is not None
        assert manifest.config_hash == config_hash(driver.config)
        assert manifest.solver == "liggghts"

    def test_a_mujoco_run_carries_provenance(
        self, config_path: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("mujoco", reason="mujoco not installed")
        driver = MPMDriver(config_path)
        output = tmp_path / "result.h5"
        driver.run(output)

        with BunkerShotResultReader(output) as reader:
            manifest = reader.manifest
        assert manifest is not None
        assert manifest.solver == "mujoco"
        assert manifest.config_hash == config_hash(driver.config)
        assert manifest.wall_clock_s > 0.0
