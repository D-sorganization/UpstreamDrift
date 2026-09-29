"""Unit tests for the marker shape registry."""

from __future__ import annotations

import pytest

from src.shared.python.plot_style import MarkerShape, MarkerShapeRenderer
from src.shared.python.plot_style.shapes import (
    SHAPE_REGISTRY,
    CrossMarker,
    CubeMarker,
    DiamondMarker,
    SphereMarker,
    StarMarker,
    default_marker_for,
)


def test_registry_keys() -> None:
    assert set(SHAPE_REGISTRY.keys()) == {
        MarkerShape.SPHERE,
        MarkerShape.CUBE,
        MarkerShape.CROSS,
        MarkerShape.STAR,
        MarkerShape.DIAMOND,
    }


def test_registry_factories_match_classes() -> None:
    assert SHAPE_REGISTRY[MarkerShape.SPHERE] is SphereMarker
    assert SHAPE_REGISTRY[MarkerShape.CUBE] is CubeMarker
    assert SHAPE_REGISTRY[MarkerShape.CROSS] is CrossMarker
    assert SHAPE_REGISTRY[MarkerShape.STAR] is StarMarker
    assert SHAPE_REGISTRY[MarkerShape.DIAMOND] is DiamondMarker


def test_default_marker_for_each_shape() -> None:
    for shape in SHAPE_REGISTRY:
        m = default_marker_for(shape)
        assert isinstance(m, MarkerShapeRenderer)
        assert m.shape_id == shape.value


def test_default_marker_for_custom_mesh_raises() -> None:
    with pytest.raises(KeyError):
        default_marker_for(MarkerShape.CUSTOM_MESH)


def test_default_marker_for_wrong_type() -> None:
    with pytest.raises(TypeError):
        default_marker_for("sphere")  # type: ignore[arg-type]
