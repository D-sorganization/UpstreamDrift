"""Portrait detail must change visible landmarks without changing anatomy."""

from dataclasses import fields

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, document_to_dict
from src.shared.python.model_appearance.head_mesh import (
    build_mesh_parts,
    load_head_asset,
)
from src.shared.python.model_appearance.head_morph import HeadMorph, warp_vertices
from src.shared.python.model_appearance.schema import HeadSettings

pytestmark = pytest.mark.unit
CONTROLS = (
    "cheeks",
    "chin",
    "brow",
    "nose_width",
    "nose_bridge",
    "nose_length",
    "eye_opening",
    "face_length",
    "ears",
)


@pytest.mark.parametrize("name", CONTROLS)
def test_portrait_control_roundtrip_and_bounds(name):
    """Every published control is effective, finite, bounded and serializable."""
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"morph": {name: 0.18}}}
    )
    assert document_from_dict(document_to_dict(doc)) == doc
    skin = load_head_asset().skin
    changed = warp_vertices(skin.vertices, doc.head.morph)
    assert np.isfinite(changed).all()
    assert np.max(np.abs(changed - skin.vertices)) > 0.001
    for value in (float("nan"), float("inf"), -0.201, 0.201):
        with pytest.raises(ValueError):
            HeadMorph(**{name: value})


def test_legacy_serialization_and_exact_head_envelope():
    """Legacy payloads stay compact; all new controls anchor neck and skull top."""
    legacy = {"width": 0.1, "jaw": 0.0, "nose": 0.0, "mouth": 0.0, "eyes": 0.0}
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"morph": legacy}}
    )
    assert document_to_dict(doc)["head"]["morph"] == legacy
    skin = load_head_asset().skin
    for sign in (-1, 1):
        morph = HeadMorph(**{f.name: sign * 0.2 for f in fields(HeadMorph)})
        changed = warp_vertices(skin.vertices, morph)
        assert changed[:, 2].min() == skin.vertices[:, 2].min()
        assert changed[:, 2].max() == skin.vertices[:, 2].max()
        assert np.array_equal(
            changed[skin.vertices[:, 2] <= 0], skin.vertices[skin.vertices[:, 2] <= 0]
        )


def test_nose_and_eye_controls_have_distinct_directional_effects():
    """A longer bridge and narrower eye opening are not global head scaling."""
    points = np.array(
        [[0.5, 0.0, 0.51], [0.55, 0.03, 0.41], [0.5, 0.12, 0.558], [0.5, 0.12, 0.518]]
    )
    bridge = warp_vertices(points, HeadMorph(nose_bridge=0.2))
    length = warp_vertices(points, HeadMorph(nose_length=0.2))
    eyes = warp_vertices(points, HeadMorph(eye_opening=-0.2))
    assert bridge[0, 0] > points[0, 0] + 0.01
    assert length[1, 2] < points[1, 2] - 0.005
    assert np.ptp(eyes[2:, 2]) < 0.8 * np.ptp(points[2:, 2])


def test_moustache_is_opt_in_closed_visual_geometry():
    """Facial hair adds only closed meshes; old head parts retain exact vertices."""
    before = build_mesh_parts(0.24, "none", HeadSettings())
    after = build_mesh_parts(0.24, "none", HeadSettings(facial_hair="moustache"))
    assert len(after) == len(before) + 2
    for old, new in zip(before, after, strict=False):
        assert old[0] == new[0]
        np.testing.assert_array_equal(old[1].vertices, new[1].vertices)
    for name, mesh, _ in after[len(before) :]:
        assert name.startswith("moustache_") and mesh.volume() > 0
        e = np.concatenate(
            [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]
        )
        assert sorted(map(tuple, e)) == sorted(map(tuple, e[:, ::-1]))
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"facial_hair": "moustache"}}
    )
    assert document_from_dict(document_to_dict(doc)) == doc
    with pytest.raises(ValueError):
        HeadSettings(model="parametric", facial_hair="moustache")
