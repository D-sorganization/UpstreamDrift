"""The drawn head is sized from the subject's stature, visual only (#15299).

Bands are adult male means (Gordon et al. 1989, the 1988 US Army survey:
menton to top of head 232 mm, head breadth 152 mm, head length 197 mm at a
1.756 m stature), read as fractions of stature with a few percent of slack.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, document_to_dict
from src.shared.python.model_appearance import head as head_model
from src.shared.python.model_appearance import head_mesh
from src.shared.python.model_appearance.schema import HeadSettings

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPEC = json.loads((MODELS / "full_body_spec_anthro_driver.json").read_text())
BARE = {"schema_version": "appearance-v1"}
LEGACY = {"sizing": "segment_length"}
STATURES_M = (1.71, 1.82)  # the committed driver subject and the tour-average preset
HEIGHT_OVER_STATURE = (0.128, 0.136)  # menton to vertex; 1 : 7.8 to 1 : 7.35
BREADTH_OVER_STATURE = (0.080, 0.090)  # 152 mm of 1.756 m is 0.0866
LENGTH_OVER_STATURE = (0.104, 0.116)  # 197 mm of 1.756 m is 0.1122
BREADTH_OVER_HEIGHT = (0.60, 0.70)  # 152 / 232 is 0.655
LENGTH_OVER_HEIGHT = (0.80, 0.88)  # 197 / 232 is 0.849


def _spec(stature_m: float) -> dict:
    spec = copy.deepcopy(SPEC)
    spec["subject"]["stature_m"] = stature_m
    return spec


def _drawn(stature_m: float, head: dict | None = None) -> float:
    doc = document_from_dict({**BARE, "head": head or {}})
    spec = _spec(stature_m)
    anchor = head_model.resolve_head_anchor(spec)
    assert anchor is not None
    return head_model.drawn_head_length(spec, anchor, doc.head)


def test_the_asset_proportions_are_measured_from_the_mesh() -> None:
    measured = head_mesh.head_proportions()
    skin = head_mesh.load_head_asset().skin.vertices
    assert measured.menton_to_vertex == pytest.approx(0.898, abs=0.005)
    assert 0.0 < 1.0 - measured.menton_to_vertex < 0.2  # the chin is above the neck
    assert measured.breadth < np.ptp(skin[:, 1])  # the skull, not the ears
    assert measured.length < np.ptp(skin[:, 0])  # glabella, not the nose tip
    low, high = BREADTH_OVER_HEIGHT
    assert low <= measured.breadth / measured.menton_to_vertex <= high
    low, high = LENGTH_OVER_HEIGHT
    assert low <= measured.length / measured.menton_to_vertex <= high


@pytest.mark.parametrize("stature_m", STATURES_M)
def test_the_default_head_is_in_proportion_to_the_stature(stature_m: float) -> None:
    length = _drawn(stature_m)
    measured = head_mesh.head_proportions()
    for fraction, (low, high) in (
        (measured.menton_to_vertex, HEIGHT_OVER_STATURE),
        (measured.breadth, BREADTH_OVER_STATURE),
        (measured.length, LENGTH_OVER_STATURE),
    ):
        assert low <= fraction * length / stature_m <= high, (fraction, length)


def test_the_head_grows_with_the_subject() -> None:
    short, tall = (_drawn(s) for s in STATURES_M)
    assert tall / short == pytest.approx(STATURES_M[1] / STATURES_M[0])
    ratio = HeadSettings().height_over_stature
    fraction = head_mesh.head_proportions().menton_to_vertex
    assert short * fraction == pytest.approx(ratio * STATURES_M[0])


def test_the_drawn_meshes_carry_the_stature_size() -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

    spec = _spec(STATURES_M[1])
    xml, _ = build_scene_xml(json.dumps(spec).encode(), BARE)
    model = mujoco.MjModel.from_xml_string(xml)
    mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "vmesh_head_skin_mesh")
    first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    unit = head_mesh.load_head_asset().skin.vertices
    extent = np.ptp(model.mesh_vert[first : first + count], axis=0).max()
    assert extent / np.ptp(unit, axis=0).max() == pytest.approx(
        _drawn(STATURES_M[1]), rel=1e-4
    )


def test_the_old_size_is_one_setting_away() -> None:
    for stature_m in STATURES_M:
        assert _drawn(stature_m, LEGACY) == head_model.DEFAULT_HEAD_LENGTH_M
    doc = document_from_dict({**BARE, "head": LEGACY})
    assert document_to_dict(doc)["head"]["sizing"] == "segment_length"
    assert "sizing" not in document_to_dict(document_from_dict(BARE))["head"]


def test_the_parametric_head_keeps_its_segment_length() -> None:
    assert _drawn(1.82, {"model": "parametric"}) == head_model.DEFAULT_HEAD_LENGTH_M


def test_a_spec_without_a_stature_keeps_its_segment_length() -> None:
    spec = copy.deepcopy(SPEC)
    del spec["subject"]["stature_m"]
    anchor = head_model.resolve_head_anchor(spec)
    assert anchor is not None
    assert head_model.drawn_head_length(spec, anchor, HeadSettings()) == anchor.length_m


def test_the_ratio_is_a_bounded_setting() -> None:
    wide = {"height_over_stature": 0.14}
    assert _drawn(1.8, wide) / _drawn(1.8) == pytest.approx(0.14 / 0.132)
    assert document_from_dict({**BARE, "head": wide}).head.height_over_stature == 0.14
    for bad in ({"height_over_stature": 0.3}, {"sizing": "guess"}):
        with pytest.raises(ValueError, match="head|appearance"):
            document_from_dict({**BARE, "head": bad})
    anchor = head_model.resolve_head_anchor(SPEC)
    assert anchor is not None
    with pytest.raises(ValueError, match="stature"):
        head_model.drawn_head_length(_spec(-1.0), anchor, HeadSettings())


def test_resizing_the_head_changes_nothing_kinematic() -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

    raw = json.dumps(_spec(STATURES_M[1])).encode()
    new, old = (
        mujoco.MjModel.from_xml_string(build_scene_xml(raw, {**BARE, "head": h})[0])
        for h in ({}, LEGACY)
    )
    assert (new.nbody, new.nq, new.nv, new.nsite) == (
        old.nbody,
        old.nq,
        old.nv,
        old.nsite,
    )
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "site_pos",
        "site_quat",
    ):
        assert np.array_equal(getattr(new, field), getattr(old, field)), field
    assert (
        head_model.resolve_head_anchor(_spec(STATURES_M[1])).length_m
        == (SPEC["anthropometry"]["segments"]["head"]["length_m"])
    )
