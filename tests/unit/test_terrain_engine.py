"""Tests for physics.terrain_engine (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.terrain import ElevationMap, Terrain
from src.shared.python.physics.terrain_engine import (
    TerrainAwareEngine,
    TerrainContactModel,
)


def _flat_terrain(height: float = 0.0) -> Terrain:
    data = np.full((10, 10), height)
    elev = ElevationMap(data=data, resolution=1.0, width=9.0, length=9.0)
    return Terrain(name="flat", elevation=elev)


class TestTerrainAwareEngine:
    def test_construction_no_terrain(self) -> None:
        eng = TerrainAwareEngine()
        assert eng is not None
        assert eng.terrain is None

    def test_construction_with_terrain(self) -> None:
        t = _flat_terrain()
        eng = TerrainAwareEngine(terrain=t)
        assert eng.terrain is not None

    def test_get_ground_height_no_terrain(self) -> None:
        eng = TerrainAwareEngine()
        h = eng.get_ground_height(0.0, 0.0)
        assert h == pytest.approx(0.0)

    def test_get_ground_height_with_flat_terrain(self) -> None:
        t = _flat_terrain(0.0)
        eng = TerrainAwareEngine(terrain=t)
        h = eng.get_ground_height(3.0, 3.0)
        assert h == pytest.approx(0.0)

    def test_set_terrain(self) -> None:
        eng = TerrainAwareEngine()
        t = _flat_terrain()
        eng.set_terrain(t)
        assert eng.terrain is t

    def test_elevated_terrain(self) -> None:
        t = _flat_terrain(2.0)
        eng = TerrainAwareEngine(terrain=t)
        h = eng.get_ground_height(4.0, 4.0)
        assert h == pytest.approx(2.0)


class TestTerrainContactModel:
    def setup_method(self) -> None:
        self.t = _flat_terrain(0.0)
        self.model = TerrainContactModel(terrain=self.t)

    def test_above_terrain_not_in_contact(self) -> None:
        assert not self.model.is_in_contact(3.0, 3.0, 5.0)

    def test_at_terrain_in_contact(self) -> None:
        assert self.model.is_in_contact(3.0, 3.0, 0.0)

    def test_below_terrain_in_contact(self) -> None:
        assert self.model.is_in_contact(3.0, 3.0, -1.0)

    def test_penetration_above_zero(self) -> None:
        # Above terrain → no penetration (0)
        pen = self.model.compute_penetration(3.0, 3.0, 2.0)
        assert pen == pytest.approx(0.0)

    def test_penetration_below_positive(self) -> None:
        # Below terrain → positive penetration
        pen = self.model.compute_penetration(3.0, 3.0, -1.0)
        assert pen > 0.0
