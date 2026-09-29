"""
Tests for the LIGGGHTS backend driver.

The backend refuses to run (ADR-0032 / #8612 finding B2: the deck contains no
clubhead), so these tests cover the deck writer and the dump parser, which are
retained. The refusal itself is pinned in ``test_liggghts_nonviable_8612.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _bunker_fixtures_8612 import write_config
from bunkershot3d.backends.liggghts.driver import LiggghtsDriver, _iter_dump_frames
from bunkershot3d.exceptions import BackendNotImplementedError

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path / "canonical.yaml",
        grain_count=500,
        diameter_mean=0.002,
        diameter_sigma_log=0.1,
        duration=0.005,
        rate_hz=500.0,
    )


# ---------------------------------------------------------------------------
# Unit tests — no LIGGGHTS binary required
# ---------------------------------------------------------------------------


class TestLiggghtsDriverInit:
    def test_init_loads_config(self, dummy_config: Path) -> None:
        driver = LiggghtsDriver(dummy_config)
        assert driver.config is not None
        assert driver.config.bunker_bed.domain.length_x == 2.0

    def test_init_stores_config_path(self, dummy_config: Path) -> None:
        driver = LiggghtsDriver(dummy_config)
        assert driver.config_path == dummy_config


class TestSetupAndRunRefuse:
    """setup() and run() raise BackendNotImplementedError unconditionally."""

    def test_setup_raises_backend_not_implemented_error(
        self, dummy_config: Path
    ) -> None:
        driver = LiggghtsDriver(dummy_config)
        with pytest.raises(BackendNotImplementedError):
            driver.setup()

    def test_setup_error_message_explains_the_missing_intruder(
        self, dummy_config: Path
    ) -> None:
        driver = LiggghtsDriver(dummy_config)
        with pytest.raises(BackendNotImplementedError, match="no clubhead"):
            driver.setup()

    def test_run_raises_backend_not_implemented_error(
        self, dummy_config: Path, tmp_path: Path
    ) -> None:
        driver = LiggghtsDriver(dummy_config)
        with pytest.raises(BackendNotImplementedError):
            driver.run(tmp_path / "out.h5")

    def test_run_writes_no_output(self, dummy_config: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.h5"
        with pytest.raises(BackendNotImplementedError):
            LiggghtsDriver(dummy_config).run(out)
        assert not out.exists()


class TestInputDeckGeneration:
    """_generate_input_deck() must produce a syntactically reasonable script."""

    def test_deck_contains_hertz_mindlin(
        self, dummy_config: Path, tmp_path: Path
    ) -> None:
        driver = LiggghtsDriver(dummy_config)
        deck = driver._generate_input_deck(tmp_path)
        content = deck.read_text()
        assert "hertz" in content

    def test_deck_contains_domain_dimensions(
        self, dummy_config: Path, tmp_path: Path
    ) -> None:
        driver = LiggghtsDriver(dummy_config)
        deck = driver._generate_input_deck(tmp_path)
        content = deck.read_text()
        assert "2" in content  # length_x = 2.0
        assert "region" in content

    def test_deck_contains_gravity(self, dummy_config: Path, tmp_path: Path) -> None:
        driver = LiggghtsDriver(dummy_config)
        deck = driver._generate_input_deck(tmp_path)
        content = deck.read_text()
        assert "gravity" in content

    def test_deck_contains_dump_command(
        self, dummy_config: Path, tmp_path: Path
    ) -> None:
        driver = LiggghtsDriver(dummy_config)
        deck = driver._generate_input_deck(tmp_path)
        content = deck.read_text()
        assert "dump" in content

    def test_deck_is_written_to_work_dir(
        self, dummy_config: Path, tmp_path: Path
    ) -> None:
        driver = LiggghtsDriver(dummy_config)
        deck = driver._generate_input_deck(tmp_path)
        assert deck.parent == tmp_path
        assert deck.exists()


class TestDumpParser:
    """_iter_dump_frames() must parse the standard LIGGGHTS custom dump format."""

    def _write_dump(
        self, path: Path, frames: list[tuple[int, list[list[float]]]]
    ) -> None:
        """Write a minimal LIGGGHTS custom dump file."""
        with open(path, "w", encoding="utf-8") as f:
            for step, atoms in frames:
                n = len(atoms)
                f.write(f"ITEM: TIMESTEP\n{step}\n")
                f.write(f"ITEM: NUMBER OF ATOMS\n{n}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0.0 2.0\n0.0 1.0\n0.0 0.5\n")
                f.write("ITEM: ATOMS id type x y z vx vy vz\n")
                f.writelines(
                    f"{i} 1 " + " ".join(f"{v:.4f}" for v in row) + "\n"
                    for i, row in enumerate(atoms, start=1)
                )

    def test_parses_single_frame(self, tmp_path: Path) -> None:
        dump = tmp_path / "dump.test"
        atoms = [[0.1, 0.2, 0.3, 0.01, 0.02, 0.03]]
        self._write_dump(dump, [(0, atoms)])
        frames = list(_iter_dump_frames(dump))
        assert len(frames) == 1
        step, pos, vel = frames[0]
        assert step == 0
        np.testing.assert_allclose(pos[0], [0.1, 0.2, 0.3], atol=1e-4)
        np.testing.assert_allclose(vel[0], [0.01, 0.02, 0.03], atol=1e-4)

    def test_parses_multiple_frames(self, tmp_path: Path) -> None:
        dump = tmp_path / "dump.test"
        frame0 = [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.4, 0.5, 0.1, 0.1, 0.0, 0.0]]
        frame1 = [[0.2, 0.2, 0.3, 0.0, 0.0, -0.1], [0.4, 0.5, 0.15, 0.1, 0.0, -0.05]]
        self._write_dump(dump, [(0, frame0), (100, frame1)])
        frames = list(_iter_dump_frames(dump))
        assert len(frames) == 2
        assert frames[0][0] == 0
        assert frames[1][0] == 100
        assert frames[0][1].shape == (2, 3)
        assert frames[1][2].shape == (2, 3)

    def test_positions_shape_correct(self, tmp_path: Path) -> None:
        dump = tmp_path / "dump.test"
        atoms = [[float(i), 0.1, 0.2, 0.0, 0.0, 0.0] for i in range(5)]
        self._write_dump(dump, [(0, atoms)])
        _, pos, vel = list(_iter_dump_frames(dump))[0]
        assert pos.shape == (5, 3)
        assert vel.shape == (5, 3)


class TestParseAndWrite:
    """_parse_and_write() must produce a valid HDF5 file with grain states."""

    def _write_dump(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for step in [0, 200]:
                atoms = [
                    [0.1 * j, 0.05 * j, 0.02 * j, 0.0, 0.0, -0.01] for j in range(3)
                ]
                n = len(atoms)
                f.write(f"ITEM: TIMESTEP\n{step}\n")
                f.write(f"ITEM: NUMBER OF ATOMS\n{n}\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0.0 2.0\n0.0 1.0\n0.0 0.5\n")
                f.write("ITEM: ATOMS id type x y z vx vy vz\n")
                f.writelines(
                    f"{i} 1 " + " ".join(f"{v:.4f}" for v in row) + "\n"
                    for i, row in enumerate(atoms, start=1)
                )

    def test_creates_hdf5_output(self, dummy_config: Path, tmp_path: Path) -> None:
        driver = LiggghtsDriver(dummy_config)
        dump = tmp_path / "dump.bunkershot"
        output = tmp_path / "result.h5"
        self._write_dump(dump)
        driver._parse_and_write(tmp_path, output)
        assert output.exists()

    def test_hdf5_contains_grain_group(
        self, dummy_config: Path, tmp_path: Path
    ) -> None:
        import h5py

        driver = LiggghtsDriver(dummy_config)
        dump = tmp_path / "dump.bunkershot"
        output = tmp_path / "result.h5"
        self._write_dump(dump)
        driver._parse_and_write(tmp_path, output)

        with h5py.File(output, "r") as f:
            assert "grains" in f
