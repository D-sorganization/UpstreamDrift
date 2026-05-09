"""Unit tests for ``body_part_viz.shapes._mesh_decimation``."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from src.shared.python.body_part_viz.shapes._mesh_decimation import decimate


def test_quadric_reduces_icosphere() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4)
    assert len(sphere.faces) >= 5000
    v, f = decimate(
        np.asarray(sphere.vertices),
        np.asarray(sphere.faces),
        max_vertices=500,
        strategy="quadric",
    )
    assert len(v) <= 500
    assert f.shape[1] == 3


def test_uniform_strategy() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4)
    v, f = decimate(
        np.asarray(sphere.vertices),
        np.asarray(sphere.faces),
        max_vertices=500,
        strategy="uniform",
    )
    assert len(v) <= 500


def test_no_decimation_when_under_budget() -> None:
    box = trimesh.creation.box()
    v_in = np.asarray(box.vertices)
    f_in = np.asarray(box.faces)
    v, f = decimate(v_in, f_in, max_vertices=1000, strategy="quadric")
    assert v is v_in
    assert f is f_in


def test_uniform_fallback_on_quadric_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """When quadric raises, the helper must fall back to uniform."""
    sphere = trimesh.creation.icosphere(subdivisions=4)

    def _raise(self: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
        raise RuntimeError("synthetic non-manifold failure")

    monkeypatch.setattr(
        trimesh.Trimesh, "simplify_quadric_decimation", _raise, raising=True
    )

    v, f = decimate(
        np.asarray(sphere.vertices),
        np.asarray(sphere.faces),
        max_vertices=500,
        strategy="quadric",
    )
    assert len(v) <= 500
    assert len(v) > 0


def test_uniform_fallback_when_quadric_overshoots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4)

    def _overshoot(self: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
        # Returns the original mesh unchanged → overshoots the budget.
        return self

    monkeypatch.setattr(
        trimesh.Trimesh, "simplify_quadric_decimation", _overshoot, raising=True
    )

    v, f = decimate(
        np.asarray(sphere.vertices),
        np.asarray(sphere.faces),
        max_vertices=500,
        strategy="quadric",
    )
    assert len(v) <= 500


def test_validates_max_vertices() -> None:
    box = trimesh.creation.box()
    with pytest.raises(ValueError, match="max_vertices"):
        decimate(
            np.asarray(box.vertices),
            np.asarray(box.faces),
            max_vertices=2,
        )


def test_validates_vertices_shape() -> None:
    with pytest.raises(ValueError, match="vertices"):
        decimate(
            np.zeros((4, 2)),
            np.zeros((1, 3), dtype=np.int64),
            max_vertices=100,
        )


def test_validates_faces_shape() -> None:
    with pytest.raises(ValueError, match="faces"):
        decimate(
            np.zeros((4, 3)),
            np.zeros((1, 4), dtype=np.int64),
            max_vertices=100,
        )
