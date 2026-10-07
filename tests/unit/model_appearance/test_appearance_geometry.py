"""Smooth segment meshes: closed, outward-wound and finite."""

import numpy as np
import pytest

from src.shared.python.model_appearance import ellipsoid_mesh, lofted_segment
from src.shared.python.model_appearance.geometry import LoftOptions

pytestmark = pytest.mark.unit
Z = np.array([0.0, 0.0, 0.4])


def test_spindle_is_closed_with_positive_volume() -> None:
    mesh = lofted_segment(np.zeros(3), Z, 0.05)
    assert mesh.volume() > 0
    assert np.isfinite(mesh.vertices).all()
    assert mesh.faces.max() == len(mesh.vertices) - 1
    edges = np.sort(
        np.concatenate(
            [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()  # watertight


def test_garment_band_is_thicker_than_the_skin_it_covers() -> None:
    skin = lofted_segment(np.zeros(3), Z, 0.05, LoftOptions(coverage=(0.0, 0.6)))
    cloth = lofted_segment(
        np.zeros(3), Z, 0.05, LoftOptions(coverage=(0.0, 0.6), thickness=1.1)
    )
    assert cloth.volume() > skin.volume() > 0


def test_ellipsoid_volume_matches_the_analytic_value() -> None:
    half = np.array([0.05, 0.04, 0.03])
    mesh = ellipsoid_mesh(
        np.zeros(3), half, np.array([1.0, 1.0, 0.0]), rings=40, sides=60
    )
    assert mesh.volume() == pytest.approx(4 / 3 * np.pi * half.prod(), rel=0.02)


@pytest.mark.parametrize(
    "call",
    [
        lambda: lofted_segment(np.zeros(3), np.zeros(3), 0.05),
        lambda: lofted_segment(np.zeros(3), Z, -0.1),
        lambda: lofted_segment(np.zeros(3), Z, 0.05, LoftOptions(coverage=(0.6, 0.2))),
        lambda: lofted_segment(np.zeros(3), np.array([np.nan, 0, 1]), 0.05),
        lambda: ellipsoid_mesh(np.zeros(3), np.array([1, 1, 0.0]), Z),
        lambda: ellipsoid_mesh(np.zeros(3), np.ones(3), np.zeros(3)),
    ],
)
def test_preconditions_raise_value_error(call) -> None:
    with pytest.raises(ValueError):
        call()
