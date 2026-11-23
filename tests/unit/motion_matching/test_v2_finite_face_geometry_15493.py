"""Finite rendered face geometry is not contact or feasibility evidence."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import club_face_marking as marking
from src.shared.python.motion_matching import club_face_target as target

pytestmark = pytest.mark.unit


def _spec():
    path = Path(__file__).resolve().parents[3] / (
        "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_surface_matches_rendered_patch_after_transform():
    spec = _spec()
    assembly = ca.assembly_from_spec(spec)
    assert assembly is not None
    settings = dataclasses.replace(
        marking.load_face_marking_settings(), face_marking=True
    )
    drawn = ca.assembly_meshes(dataclasses.replace(assembly, face_marking=settings))[
        "head_face"
    ]
    frame = next(f for f in spec["frames"] if f["name"] == target.FACE_FRAME)
    transform = np.asarray(frame["placement"], dtype=float)
    face = target.face_triangles_in_frame(spec)
    assert face.ndim == 3 and face.shape[1:] == (3, 3) and len(face) > 0
    assert np.isfinite(face).all()
    restored = face @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(restored, drawn.vertices[drawn.faces], atol=1e-12)


def test_face_geometry_is_copied_and_rejects_unrelated_frame():
    spec = _spec()
    original = target.face_triangles_in_frame(spec)
    changed = target.face_triangles_in_frame(spec)
    changed[0] = 100.0
    np.testing.assert_array_equal(target.face_triangles_in_frame(spec), original)
    frame = next(f for f in spec["frames"] if f["name"] == target.FACE_FRAME)
    frame["body"] = "unrelated"
    with pytest.raises(ValueError, match="club body"):
        target.face_triangles_in_frame(spec)


def test_empty_face_selection_fails_closed(monkeypatch):
    from src.shared.python.model_appearance.club_head_mesh import load_club_head

    mesh = load_club_head("driver").head_mesh
    monkeypatch.setattr(
        marking,
        "face_triangle_mask",
        lambda mesh, frame, threshold: np.zeros(len(mesh.faces), dtype=bool),
    )
    with pytest.raises(ValueError, match="triangles"):
        marking.face_surface_triangles(mesh)
