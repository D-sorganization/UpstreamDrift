"""Metric scene references are portable, bounded and independent of pixels."""

from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.coaching import (
    History,
    ReferenceGeometry,
    ReferencePlane,
    ReferencePoint,
)


pytestmark = pytest.mark.unit


def plane(**changes: object) -> ReferencePlane:
    return ReferencePlane(
        **dict(
            id="swing",
            title="Swing Plane",
            origin_m=(0, 0, 1),
            along_m=(1, 0, 1),
            across_m=(0, 1, 1),
            **changes,
        )
    )


def test_plane_mesh_and_signed_metres() -> None:
    reference = plane()
    vertices = reference.vertices()
    assert vertices.shape == (4, 3)
    np.testing.assert_allclose(vertices[:, 2], 1)
    np.testing.assert_allclose(reference.distances([(0, 0, 3), (5, 2, 0)]), [2, -1])


@pytest.mark.parametrize("across", [(2, 0, 1), (0, 0, 1), (float("nan"), 0, 1)])
def test_reject_degenerate_plane(across: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        ReferencePlane(origin_m=(0, 0, 1), along_m=(1, 0, 1), across_m=across)


def test_missing_samples_stay_missing_and_shape_is_checked() -> None:
    result = plane().distances([[0, 0, 2], [np.nan, np.nan, np.nan]])
    assert result[0] == 1
    assert np.isnan(result[1])
    with pytest.raises(ValueError):
        plane().distances([[0, 2]])
    with pytest.raises(ValueError):
        plane().distances([[np.inf, 0, 0]])


def test_scene_binding_and_roundtrip(tmp_path: Path) -> None:
    geometry = ReferenceGeometry(
        scene_id="capture-a",
        planes=(plane(),),
        points=(ReferencePoint(title="Ball", position_m=(1, 2, 0)),),
    )
    path = tmp_path / "references.json"
    geometry.save(path)
    assert ReferenceGeometry.load(path, scene_id="capture-a") == geometry
    with pytest.raises(ValueError, match="scene"):
        ReferenceGeometry.load(path, scene_id="capture-b")


def test_visibility_edits_validate_and_duplicate_ids_fail() -> None:
    reference = plane(first_s=1, last_s=2)
    assert reference.at(1) and reference.at(2)
    assert not reference.at(0)
    assert not reference.changed(visible=False).at(1)
    with pytest.raises(ValueError):
        reference.changed(opacity=2)
    with pytest.raises(ValueError):
        reference.changed(last_s=0)
    with pytest.raises(ValueError):
        ReferenceGeometry(scene_id="a", planes=(reference, reference))
    with pytest.raises(ValueError):
        ReferenceGeometry(scene_id="a", units="pixels")


def test_point_contract_and_plane_winding() -> None:
    point = ReferencePoint(position_m=(1, 2, 3))
    assert point.changed(visible=False).visible is False
    assert point.at(0)
    with pytest.raises(ValueError):
        point.changed(position_m=(0, 0, float("inf")))
    reverse = ReferencePlane(origin_m=(0, 0, 1), along_m=(0, 1, 1), across_m=(1, 0, 1))
    assert reverse.distances([[0, 0, 2]])[0] == -1


def test_geometry_reuses_bounded_drawing_history() -> None:
    original = ReferenceGeometry(scene_id="scene")
    edited = ReferenceGeometry(scene_id="scene", planes=(plane(),))
    history = History(original)
    history.apply(edited)
    assert history.current == edited and history.can_undo
    history.undo()
    assert history.current == original and history.can_redo
    history.redo()
    assert history.current == edited
