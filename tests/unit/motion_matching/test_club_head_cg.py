"""Head-CG offset behind the face (SHOW-C4 follow-up, #14600)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.club_head_cg import (
    MAX_BEHIND_FACE_M,
    apply_head_cg_offset,
    face_normal_in_club_frame,
    toe_axis_in_club_frame,
)

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _head(document: dict) -> dict:
    heads = [
        s
        for b in document["bodies"]
        for s in b["solids"]
        if s["name"].endswith("/Clubhead")
    ]
    assert len(heads) == 1
    return heads[0]


def test_default_document_is_unchanged_by_a_zero_offset(spec):
    out = apply_head_cg_offset(spec, behind_face_m=0.0)
    assert out == spec


def test_cg_moves_behind_the_face_along_minus_the_rendered_normal(spec):
    out = apply_head_cg_offset(spec, behind_face_m=0.035)
    com = np.asarray(_head(out)["com_m"])
    normal = face_normal_in_club_frame(spec)
    assert np.linalg.norm(com) == pytest.approx(0.035)
    assert com @ normal == pytest.approx(-0.035)  # behind the face


def test_toe_offset_is_in_the_face_plane_and_perpendicular_to_the_shaft(spec):
    com = np.asarray(
        _head(apply_head_cg_offset(spec, behind_face_m=0.0, toward_toe_m=0.03))["com_m"]
    )
    assert abs(com @ face_normal_in_club_frame(spec)) < 1e-12
    assert abs(com[1]) < 1e-12  # shaft is along y
    assert com @ toe_axis_in_club_frame(spec) == pytest.approx(0.03)


def test_only_the_head_com_changes_mass_and_inertia_are_kept(spec):
    out = apply_head_cg_offset(spec, behind_face_m=0.035, toward_toe_m=0.02)
    before, after = _head(spec), _head(out)
    assert after["mass_kg"] == before["mass_kg"]
    assert after["inertia_com_kg_m2"] == before["inertia_com_kg_m2"]
    stripped = json.loads(json.dumps(out))
    _head(stripped)["com_m"] = before["com_m"]
    assert stripped == spec


def test_input_document_is_not_mutated(spec):
    snapshot = json.dumps(spec, sort_keys=True)
    apply_head_cg_offset(spec, behind_face_m=0.035)
    assert json.dumps(spec, sort_keys=True) == snapshot


@pytest.mark.parametrize("bad", [-0.001, MAX_BEHIND_FACE_M + 1e-3, float("nan")])
def test_rejects_out_of_range_depth(spec, bad):
    with pytest.raises(ValueError, match="behind_face_m"):
        apply_head_cg_offset(spec, behind_face_m=bad)


def test_rejects_a_document_without_a_single_head_solid(spec):
    bare = json.loads(json.dumps(spec))
    for body in bare["bodies"]:
        body["solids"] = [
            s for s in body["solids"] if not s["name"].endswith("/Clubhead")
        ]
    with pytest.raises(ValueError, match="Clubhead"):
        apply_head_cg_offset(bare, behind_face_m=0.01)


def test_chain_shaft_carries_the_same_head_cg_and_keeps_total_com(spec):
    from src.shared.python.motion_matching.club_models import DRIVER
    from src.shared.python.motion_matching.club_shaft_chain import (
        apply_shaft_chain,
        build_shaft_chain,
    )

    rigid = apply_head_cg_offset(spec, behind_face_m=0.035)
    chain_doc = apply_shaft_chain(rigid, build_shaft_chain(DRIVER, 8))
    assert _head(chain_doc)["com_m"] == _head(rigid)["com_m"]
