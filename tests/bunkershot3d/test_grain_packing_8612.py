"""Regression tests for grain initialisation (#8612, finding B16).

- MuJoCo drew every grain from ``rng.uniform`` over the whole domain, so grains
  started interpenetrating and exploded on the first step; only the 500-step
  settle phase masked it.
- Chrono placed grains on ``np.linspace`` z-layers, which for realistic counts
  collapses the layer spacing far below one grain diameter.

Both are replaced by a jittered bottom-up lattice whose minimum separation is
guaranteed by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _bunker_fixtures_8612 import make_mock_chrono, write_config
from bunkershot3d.backends.packing import lattice_positions

pytestmark = pytest.mark.unit

# Dense enough that random placement overlaps with certainty:
# 250 grains of d = 10 mm in a 0.1 x 0.1 x 0.05 m box.
_DENSE = {
    "grain_count": 250,
    "diameter_mean": 0.01,
    "diameter_sigma_log": 0.0,
    "length_x": 0.1,
    "width_y": 0.1,
    "depth_z": 0.05,
}


def _min_separation(positions: np.ndarray) -> float:
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


class TestLatticePositions:
    def test_no_two_grains_overlap(self) -> None:
        positions = lattice_positions(
            count=250,
            extents=(0.1, 0.1, 0.05),
            diameter=0.01,
            rng=np.random.default_rng(seed=42),
        )
        assert positions.shape == (250, 3)
        assert _min_separation(positions) >= 0.01

    def test_grains_stay_inside_the_domain(self) -> None:
        positions = lattice_positions(
            count=250,
            extents=(0.1, 0.1, 0.05),
            diameter=0.01,
            rng=np.random.default_rng(seed=7),
        )
        assert np.all(np.abs(positions[:, 0]) <= 0.05 - 0.005 + 1e-12)
        assert np.all(np.abs(positions[:, 1]) <= 0.05 - 0.005 + 1e-12)
        assert np.all(positions[:, 2] >= 0.005 - 1e-12)
        assert np.all(positions[:, 2] <= 0.05 - 0.005 + 1e-12)

    def test_fills_from_the_bottom_up(self) -> None:
        """A bunker is a bed, not a cloud: grains fill the lowest sites first."""
        positions = lattice_positions(
            count=81,
            extents=(0.1, 0.1, 0.05),
            diameter=0.01,
            rng=np.random.default_rng(seed=3),
        )
        # 9 x 9 sites per layer, so 81 grains occupy exactly the bottom layer.
        assert float(np.ptp(positions[:, 2])) < 0.01

    def test_is_deterministic_for_a_given_seed(self) -> None:
        first = lattice_positions(
            count=50,
            extents=(0.1, 0.1, 0.05),
            diameter=0.01,
            rng=np.random.default_rng(seed=1),
        )
        second = lattice_positions(
            count=50,
            extents=(0.1, 0.1, 0.05),
            diameter=0.01,
            rng=np.random.default_rng(seed=1),
        )
        np.testing.assert_array_equal(first, second)

    def test_overfull_domain_is_refused(self) -> None:
        with pytest.raises(ValueError, match="grain"):
            lattice_positions(
                count=100_000,
                extents=(0.1, 0.1, 0.05),
                diameter=0.01,
                rng=np.random.default_rng(seed=1),
            )


class TestMujocoGrainPlacement:
    def test_generated_grains_do_not_interpenetrate(self, tmp_path: Path) -> None:
        mujoco = pytest.importorskip("mujoco")
        from bunkershot3d.backends.mpm.driver import MPMDriver

        config = write_config(tmp_path / "c.yaml", **_DENSE)
        driver = MPMDriver(config)
        model = mujoco.MjModel.from_xml_string(driver._generate_xml())

        grain_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"g{i}")
            for i in range(_DENSE["grain_count"])
        ]
        assert all(bid >= 0 for bid in grain_ids)
        positions = np.array([model.body_pos[bid] for bid in grain_ids])

        assert _min_separation(positions) >= _DENSE["diameter_mean"] - 1e-12, (
            "grains are initialised interpenetrating; the settle phase only "
            "masks the resulting explosion"
        )


class TestChronoGrainPlacement:
    def test_generated_grains_do_not_interpenetrate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import bunkershot3d.backends.chrono.driver as chrono_mod
        from bunkershot3d.backends.chrono.driver import ChronoDriver

        chrono = make_mock_chrono()
        monkeypatch.setattr(chrono_mod, "chrono", chrono, raising=False)
        monkeypatch.setattr(chrono_mod, "_HAS_CHRONO", True)

        config = write_config(tmp_path / "c.yaml", **_DENSE)
        driver = ChronoDriver(config)
        driver.setup()

        # 5 walls are created first, then the grains, then the clubhead.
        grain_bodies = chrono.created_bodies[5 : 5 + _DENSE["grain_count"]]
        positions = np.array(
            [
                [
                    body.SetPos.call_args[0][0].x,
                    body.SetPos.call_args[0][0].y,
                    body.SetPos.call_args[0][0].z,
                ]
                for body in grain_bodies
            ]
        )

        assert positions.shape == (_DENSE["grain_count"], 3)
        assert _min_separation(positions) >= _DENSE["diameter_mean"] - 1e-12, (
            "linspace z-layers collapse below one grain diameter"
        )
