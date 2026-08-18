"""Regression tests for the LIGGGHTS backend (#8612).

Covers baseline findings:

- **B2** — the generated deck contains a sand box, gravity, insertion and a run
  command, and **no clubhead at all**. Nothing strikes the sand. ADR-0032
  records LIGGGHTS as a non-viable tier; the driver must therefore refuse to
  run rather than silently simulate a clubless sand box.
- **B11** — a config declaring a *log-normal* diameter distribution was emitted
  as ``radius gaussian <mean> <sigma_log>``: the wrong distribution, with sigma
  interpreted in metres rather than in log-space.
- **B12** — ``dt = 1.0e-5`` was hard-coded in two places (deck writer and dump
  parser) with no stability criterion, and ``total_steps`` came from a
  hard-coded 0.5 s that ignored ``trajectory.duration``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from _bunker_fixtures_8612 import QUARTZ_DENSITY, rayleigh_time, write_config
from bunkershot3d.backends.liggghts.driver import LiggghtsDriver
from bunkershot3d.io.schema import BunkerShotResultReader

pytestmark = pytest.mark.unit


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path / "c.yaml",
        grain_count=500,
        diameter_mean=0.002,
        diameter_sigma_log=0.25,
        duration=0.02,
        rate_hz=500.0,
    )


def _deck_text(config: Path, work_dir: Path) -> str:
    return LiggghtsDriver(config)._generate_input_deck(work_dir).read_text()


class TestNonViabilityIsExplicit:
    """B2: the deck has no clubhead, so the backend must refuse to run."""

    def test_setup_refuses(self, config: Path) -> None:
        with pytest.raises(NotImplementedError, match="clubhead"):
            LiggghtsDriver(config).setup()

    def test_run_refuses(self, config: Path, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError, match="clubhead"):
            LiggghtsDriver(config).run(tmp_path / "out.h5")

    def test_refusal_does_not_depend_on_the_binary_being_absent(
        self, config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with a binary on PATH the backend must not run."""
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/liggghts")
        with pytest.raises(NotImplementedError, match="clubhead"):
            LiggghtsDriver(config).run(tmp_path / "out.h5")

    def test_refusal_cites_the_adr(self, config: Path) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            LiggghtsDriver(config).setup()
        assert "0032" in str(excinfo.value)

    def test_deck_still_has_no_clubhead(self, config: Path, tmp_path: Path) -> None:
        """Pin the premise of B2 so a future clubhead lands with a test."""
        text = _deck_text(config, tmp_path).lower()
        assert "clubhead" not in text
        assert "mesh/surface" not in text


class TestLognormalDistribution:
    """B11: the log-normal spec must not be emitted as a Gaussian."""

    def test_no_gaussian_keyword(self, config: Path, tmp_path: Path) -> None:
        assert "gaussian" not in _deck_text(config, tmp_path)

    def test_radii_follow_lognormal_quantiles(
        self, config: Path, tmp_path: Path
    ) -> None:
        text = _deck_text(config, tmp_path)
        radii = [
            float(match)
            for match in re.findall(r"radius\s+constant\s+([0-9.eE+-]+)", text)
        ]
        assert len(radii) >= 3, "expected a discrete log-normal size distribution"

        median = 0.002 / 2.0
        sigma = 0.25
        # Symmetric equal-probability bins: the geometric mean is the median.
        geometric_mean = math.exp(sum(math.log(r) for r in radii) / len(radii))
        assert geometric_mean == pytest.approx(median, rel=1e-6)
        # Spread is set in log-space, not metres.
        assert max(radii) / min(radii) > 1.0
        assert max(radii) < median * math.exp(4.0 * sigma)

    def test_number_fractions_sum_to_one(self, config: Path, tmp_path: Path) -> None:
        text = _deck_text(config, tmp_path)
        match = re.search(r"fix\s+pdd1[^\n]*particledistribution/discrete[^\n]*", text)
        assert match is not None
        fractions = [
            float(token) for token in re.findall(r"pts\d+\s+([0-9.]+)", match.group(0))
        ]
        assert fractions
        assert sum(fractions) == pytest.approx(1.0, abs=1e-9)


class TestTimestepAndDuration:
    """B12: one timestep, derived from a stability criterion."""

    def test_timestep_respects_the_rayleigh_limit(
        self, config: Path, tmp_path: Path
    ) -> None:
        text = _deck_text(config, tmp_path)
        match = re.search(r"^timestep\s+([0-9.eE+-]+)", text, flags=re.MULTILINE)
        assert match is not None
        dt = float(match.group(1))

        r_min = (0.002 / 2.0) * math.exp(-3.0 * 0.25)
        limit = 0.2 * rayleigh_time(r_min, QUARTZ_DENSITY, 7.0e10, 0.17)
        assert dt <= limit, f"dt={dt:.3e} exceeds the 0.2-Rayleigh limit {limit:.3e}"
        assert dt != 1.0e-5

    def test_run_length_follows_trajectory_duration(
        self, config: Path, tmp_path: Path
    ) -> None:
        text = _deck_text(config, tmp_path)
        dt = float(
            re.search(r"^timestep\s+([0-9.eE+-]+)", text, flags=re.MULTILINE).group(1)  # type: ignore[union-attr]
        )
        steps = int(
            re.search(r"^run\s+(\d+)", text, flags=re.MULTILINE).group(1)  # type: ignore[union-attr]
        )
        assert steps * dt == pytest.approx(0.02, rel=1e-3), (
            "run length still comes from a hard-coded 0.5 s"
        )

    def test_dump_parser_uses_the_same_timestep(
        self, config: Path, tmp_path: Path
    ) -> None:
        driver = LiggghtsDriver(config)
        text = driver._generate_input_deck(tmp_path).read_text()
        dt = float(
            re.search(r"^timestep\s+([0-9.eE+-]+)", text, flags=re.MULTILINE).group(1)  # type: ignore[union-attr]
        )

        dump = tmp_path / "dump.bunkershot"
        dump.write_text(
            "ITEM: TIMESTEP\n1000\n"
            "ITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
            "ITEM: ATOMS id type x y z vx vy vz\n"
            "1 1 0.1 0.1 0.1 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        out = tmp_path / "result.h5"
        driver._parse_and_write(tmp_path, out)

        reader = BunkerShotResultReader(out)
        times, _positions, _velocities = reader.read_grain_states()
        reader.close()
        assert float(times[0]) == pytest.approx(1000 * dt, rel=1e-6), (
            "the parser's hard-coded dt disagrees with the deck's"
        )

    def test_no_hard_coded_timestep_in_source(self) -> None:
        import bunkershot3d.backends.liggghts.driver as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "dt = 1.0e-5" not in source
        assert "0.5 / dt" not in source
