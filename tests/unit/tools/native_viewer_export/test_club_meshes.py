"""Native viewer backends draw the shared club meshes, not the ellipsoid hint."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.native_viewer_export.backends._club import club_parts, write_obj

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def _spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def test_club_parts_are_shaft_grip_head_in_the_club_body() -> None:
    result = club_parts(_spec())
    assert result is not None
    body, parts = result
    assert body.endswith("Clubface Vector")
    assert [p.name for p in parts] == ["club_shaft", "club_grip", "club_head"]
    for part in parts:
        assert part.mesh.volume() > 0.0  # closed, outward wound
        assert len(part.rgba) == 4


def test_grip_uses_rubber_and_shaft_the_club_finish() -> None:
    _, parts = club_parts(_spec())  # type: ignore[misc]
    by_name = {p.name: p.rgba for p in parts}
    assert by_name["club_grip"] != by_name["club_shaft"]
    assert by_name["club_head"] == by_name["club_shaft"]


def test_spec_without_a_club_returns_none() -> None:
    spec = _spec()
    spec["bodies"] = [b for b in spec["bodies"] if "Clubface" not in b["name"]]
    assert club_parts(spec) is None


def test_unknown_finish_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown club finish"):
        club_parts(_spec(), finish="no_such_finish")


def test_write_obj_round_trips_vertex_and_face_counts(tmp_path: Path) -> None:
    _, parts = club_parts(_spec())  # type: ignore[misc]
    head = parts[-1].mesh
    path = Path(write_obj(head, str(tmp_path), "head"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("v ") for line in lines) == len(head.vertices)
    faces = [line.split()[1:] for line in lines if line.startswith("f ")]
    assert len(faces) == len(head.faces)
    assert np.asarray(faces, dtype=int).min() == 1


def test_pinocchio_triangle_mesh_matches_the_head() -> None:
    coal = pytest.importorskip("coal")
    from src.tools.native_viewer_export.backends.pinocchio_meshcat import _bvh_mesh

    _, parts = club_parts(_spec())  # type: ignore[misc]
    head = parts[-1].mesh
    bvh = _bvh_mesh(coal, head)
    assert bvh.num_tris == len(head.faces)
