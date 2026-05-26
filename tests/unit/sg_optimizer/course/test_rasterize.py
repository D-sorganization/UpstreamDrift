"""Rasterizer — lie-priority resolution and thin-hazard preservation."""

from __future__ import annotations

import numpy as np

from src.shared.python.sg_optimizer.course.rasterize import (
    LIE_CODES,
    CircleFeature,
    RectFeature,
    SyntheticHole,
    rasterize_synthetic,
)


def test_priority_water_overrides_fairway():
    hole = SyntheticHole(
        name="t",
        par=4,
        tee=(0.0, 0.0),
        pin=(100.0, 0.0),
        bbox=(0.0, 110.0, -20.0, 20.0),
        features=(
            RectFeature("fairway", 0.0, 110.0, -10.0, 10.0),
            RectFeature("water", 40.0, 60.0, -5.0, 5.0),
        ),
    )
    raster = rasterize_synthetic(hole, resolution_yd=1.0)
    assert raster.lie_at(50.0, 0.0) == LIE_CODES["water"]
    assert raster.lie_at(20.0, 0.0) == LIE_CODES["fairway"]


def test_green_overrides_rough():
    hole = SyntheticHole(
        name="t",
        par=4,
        tee=(0.0, 0.0),
        pin=(100.0, 0.0),
        bbox=(0.0, 120.0, -25.0, 25.0),
        features=(CircleFeature("green", 100.0, 0.0, 8.0),),
    )
    raster = rasterize_synthetic(hole, resolution_yd=1.0)
    assert raster.lie_at(100.0, 0.0) == LIE_CODES["holed"]  # pin cell
    assert raster.lie_at(95.0, 0.0) == LIE_CODES["green"]
    assert raster.lie_at(50.0, 0.0) == LIE_CODES["rough"]


def test_off_grid_is_ob():
    hole = SyntheticHole(
        name="t",
        par=3,
        tee=(0.0, 0.0),
        pin=(50.0, 0.0),
        bbox=(0.0, 60.0, -10.0, 10.0),
    )
    raster = rasterize_synthetic(hole, resolution_yd=1.0)
    assert raster.lie_at(-5.0, 0.0) == LIE_CODES["ob"]
    assert raster.lie_at(200.0, 0.0) == LIE_CODES["ob"]


def test_thin_water_strip_one_yard_wide_present():
    # Pitfall: at 1-yard grid, a 1-yard-wide water strip must survive.
    hole = SyntheticHole(
        name="t",
        par=4,
        tee=(0.0, 0.0),
        pin=(150.0, 0.0),
        bbox=(0.0, 200.0, -20.0, 20.0),
        features=(
            RectFeature("fairway", 0.0, 200.0, -10.0, 10.0),
            RectFeature("water", 80.0, 81.0, -10.0, 10.0),
        ),
    )
    raster = rasterize_synthetic(hole, resolution_yd=1.0)
    assert np.any(raster.codes == LIE_CODES["water"])
