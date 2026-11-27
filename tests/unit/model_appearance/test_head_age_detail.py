"""Optional age details are bounded, closed and remain visual-only."""

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    document_from_dict,
    document_to_dict,
)
from src.shared.python.model_appearance.schema import HeadSettings

pytestmark = pytest.mark.unit


def _document(amount):
    """Build a head with a requested age-detail strength."""
    return document_from_dict(
        {"schema_version": "appearance-v1", "head": {"age_detail": amount}}
    )


def test_default_and_roundtrip():
    """Zero stays out of serialized settings; positive strength roundtrips."""
    base = document_from_dict({"schema_version": "appearance-v1"})
    assert "age_detail" not in document_to_dict(base)["head"]
    assert _document(0) == base
    aged = _document(0.8)
    assert document_from_dict(document_to_dict(aged)) == aged
    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            HeadSettings(age_detail=invalid)
        with pytest.raises(ValueError):
            _document(invalid)


def test_age_details_are_closed_skin_meshes():
    """Each added fold is finite, closed and receives the existing skin material."""
    base = {p.name: p for p in build_head_parts(1.0, _document(0).head)}
    aged = {p.name: p for p in build_head_parts(1.0, _document(1).head)}
    added = set(aged) - set(base)
    assert added == {"age_eye_l", "age_eye_r", "age_cheek_l", "age_cheek_r"}
    for name in added:
        part = aged[name]
        assert part.material == "skin"
        mesh = part.mesh
        edges = np.concatenate(
            (mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]])
        )
        assert sorted(map(tuple, edges)) == sorted(map(tuple, edges[:, ::-1]))
        assert np.isfinite(mesh.vertices).all() and mesh.volume() > 0
    for name in base:
        assert np.array_equal(base[name].mesh.vertices, aged[name].mesh.vertices)


def test_age_folds_follow_the_same_face_warp():
    """Morphed folds remain registered to their corresponding facial surface."""
    from src.shared.python.model_appearance.head_morph import HeadMorph, warp_vertices

    plain = _document(1)
    data = document_to_dict(plain)
    data["head"]["morph"] = {"nose": 0.15, "cheeks": 0.1}
    morphed = document_from_dict(data)
    source = {p.name: p.mesh for p in build_head_parts(1.0, plain.head)}
    target = {p.name: p.mesh for p in build_head_parts(1.0, morphed.head)}
    transform = HeadMorph(nose=0.15, cheeks=0.1)
    for name in ("age_eye_l", "age_cheek_r"):
        np.testing.assert_allclose(
            target[name].vertices, warp_vertices(source[name].vertices, transform)
        )
        np.testing.assert_array_equal(target[name].faces, source[name].faces)


def test_cheek_fold_tracks_nose_to_mouth():
    """Cheek folds descend beside the nose toward the mouth, not across cheeks."""
    parts = {p.name: p.mesh for p in build_head_parts(1.0, _document(1).head)}
    fold = parts["age_cheek_l"].vertices
    assert np.ptp(fold[:, 2]) > 2 * np.ptp(fold[:, 1])
    assert np.min(fold[:, 1]) < 0.10 and np.max(fold[:, 1]) < 0.15
