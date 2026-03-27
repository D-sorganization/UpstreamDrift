"""Tests for src.shared.python.physics.terrain_mixin (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np

from src.shared.python.physics.terrain import ElevationMap, Terrain, TerrainType
from src.shared.python.physics.terrain_mixin import TerrainMixin

# ---------------------------------------------------------------------------
# Minimal concrete class implementing TerrainMixin
# ---------------------------------------------------------------------------


class _SimpleMixin(TerrainMixin):
    """Minimal concrete implementation for testing TerrainMixin methods."""

    def __init__(self) -> None:
        super().__init__()


def _make_terrain() -> Terrain:
    elevation = ElevationMap(
        data=np.zeros((10, 10)),
        resolution=1.0,
        width=10.0,
        length=10.0,
    )
    return Terrain("test", elevation)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestTerrainMixinInit:
    def test_terrain_none_initially(self) -> None:
        mixin = _SimpleMixin()
        assert mixin.terrain is None

    def test_terrain_enabled_initially(self) -> None:
        mixin = _SimpleMixin()
        assert mixin.terrain_enabled is True


# ---------------------------------------------------------------------------
# set_terrain / enable_terrain
# ---------------------------------------------------------------------------


class TestTerrainMixinSetTerrain:
    def test_set_terrain_stores_terrain(self) -> None:
        mixin = _SimpleMixin()
        t = _make_terrain()
        mixin.set_terrain(t)
        assert mixin.terrain is t

    def test_enable_terrain_false(self) -> None:
        mixin = _SimpleMixin()
        mixin.enable_terrain(False)
        assert mixin.terrain_enabled is False

    def test_enable_terrain_true(self) -> None:
        mixin = _SimpleMixin()
        mixin.enable_terrain(False)
        mixin.enable_terrain(True)
        assert mixin.terrain_enabled is True


# ---------------------------------------------------------------------------
# get_ground_height / get_terrain_normal / get_terrain_type
# ---------------------------------------------------------------------------


class TestTerrainMixinQueries:
    def _mixin_with_terrain(self) -> _SimpleMixin:
        mixin = _SimpleMixin()
        mixin.set_terrain(_make_terrain())
        return mixin

    def test_get_ground_height_returns_float(self) -> None:
        mixin = self._mixin_with_terrain()
        h = mixin.get_ground_height(0.0, 0.0)
        assert isinstance(h, (int, float))

    def test_get_ground_height_no_terrain_returns_zero(self) -> None:
        mixin = _SimpleMixin()
        h = mixin.get_ground_height(0.0, 0.0)
        assert h == 0.0

    def test_get_terrain_normal_returns_array(self) -> None:
        mixin = self._mixin_with_terrain()
        n = mixin.get_terrain_normal(0.0, 0.0)
        assert n.shape == (3,)

    def test_get_terrain_normal_default_points_up(self) -> None:
        mixin = self._mixin_with_terrain()
        n = mixin.get_terrain_normal(0.0, 0.0)
        # Should be roughly pointing up (z ≈ 1)
        assert n[2] > 0.5

    def test_get_terrain_type_returns_terrain_type(self) -> None:
        mixin = self._mixin_with_terrain()
        tt = mixin.get_terrain_type(0.0, 0.0)
        assert isinstance(tt, TerrainType)


# ---------------------------------------------------------------------------
# get_terrain_friction / get_terrain_restitution
# ---------------------------------------------------------------------------


class TestTerrainMixinFriction:
    def _mixin_with_terrain(self) -> _SimpleMixin:
        mixin = _SimpleMixin()
        mixin.set_terrain(_make_terrain())
        return mixin

    def test_friction_is_positive(self) -> None:
        mixin = self._mixin_with_terrain()
        f = mixin.get_terrain_friction(0.0, 0.0)
        assert f > 0.0

    def test_restitution_between_zero_and_one(self) -> None:
        mixin = self._mixin_with_terrain()
        r = mixin.get_terrain_restitution(0.0, 0.0)
        assert 0.0 <= r <= 1.0
