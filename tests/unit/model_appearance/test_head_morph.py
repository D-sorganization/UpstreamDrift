"""Visual head morphs preserve topology and attach every accessory."""

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, document_to_dict
from src.shared.python.model_appearance.head_mesh import build_mesh_parts
from src.shared.python.model_appearance.head_morph import HeadMorph, warp_vertices
from src.shared.python.model_appearance.schema import HeadSettings

pytestmark = pytest.mark.unit


def test_identity_and_neck_anchor():
    points = np.array([[0.2, 0.1, 0.0], [0.6, 0.1, 0.5], [0.0, 0.0, 1.0]])
    assert np.array_equal(warp_vertices(points, HeadMorph()), points)
    changed = warp_vertices(points, HeadMorph(width=0.1, nose=0.1))
    assert np.array_equal(changed[0], points[0])
    assert np.array_equal(changed[:, 2], points[:, 2])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.201, 0.201])
def test_bounds(value):
    with pytest.raises(ValueError):
        HeadMorph(nose=value)
    with pytest.raises(ValueError):
        document_from_dict(
            {"schema_version": "appearance-v1", "head": {"morph": {"nose": value}}}
        )


def test_appearance_roundtrip():
    doc = document_from_dict(
        {
            "schema_version": "appearance-v1",
            "head": {
                "morph": {
                    "width": 0.12,
                    "jaw": -0.08,
                    "nose": 0.06,
                    "mouth": 0.04,
                    "eyes": -0.03,
                }
            },
        }
    )
    assert document_from_dict(document_to_dict(doc)) == doc


def test_parts_share_warp_and_keep_closed_topology():
    morph = HeadMorph(width=0.1, jaw=-0.1, nose=0.1, mouth=0.1, eyes=-0.1)
    original = build_mesh_parts(0.24, "hair", HeadSettings(cap_style="baseball_cap"))
    changed = build_mesh_parts(
        0.24, "hair", HeadSettings(cap_style="baseball_cap", morph=morph)
    )
    assert len(original) == len(changed)
    for (name, before, role), (name2, after, role2) in zip(
        original, changed, strict=True
    ):
        assert (name, role) == (name2, role2)
        assert np.array_equal(before.faces, after.faces)
        assert np.allclose(
            after.vertices, warp_vertices(before.vertices / 0.24, morph) * 0.24
        )
        assert after.volume() > 0, name
        edges = np.concatenate(
            [after.faces[:, [0, 1]], after.faces[:, [1, 2]], after.faces[:, [2, 0]]]
        )
        assert sorted(map(tuple, edges)) == sorted(map(tuple, edges[:, ::-1]))


def test_parametric_head_cannot_silently_ignore_morph():
    with pytest.raises(ValueError, match="cc0_mesh"):
        document_from_dict(
            {
                "schema_version": "appearance-v1",
                "head": {"model": "parametric", "morph": {"nose": 0.1}},
            }
        )


@pytest.mark.timeout(180)
def test_morph_preserves_current_mujoco_reference_physics():
    import json

    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    raw = json.dumps(reference_spec("driver")).encode()
    docs = [
        document_from_dict({"schema_version": "appearance-v1"}),
        document_from_dict(
            {
                "schema_version": "appearance-v1",
                "head": {
                    "morph": {"width": 0.15, "jaw": -0.12, "nose": 0.12, "eyes": 0.08}
                },
            }
        ),
    ]
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=doc)[0])
        for doc in docs
    ]
    a, b = models
    assert not np.array_equal(a.mesh_vert, b.mesh_vert)
    assert (a.nbody, a.njnt, a.nq, a.nv) == (b.nbody, b.njnt, b.nq, b.nv)
    assert [a.joint(i).name for i in range(a.njnt)] == [
        b.joint(i).name for i in range(b.njnt)
    ]
    for key in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "qpos0",
    ):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=key)
    rng = np.random.default_rng(41)
    for _ in range(3):
        q = a.qpos0 + rng.normal(0, 0.03, a.nq)
        v = rng.normal(0, 0.05, a.nv)
        states = []
        for model in models:
            data = mujoco.MjData(model)
            data.qpos[:] = q
            data.qvel[:] = v
            mujoco.mj_forward(model, data)
            states.append(data)
        for key in ("qM", "qfrc_bias", "qacc"):
            np.testing.assert_array_equal(
                getattr(states[0], key), getattr(states[1], key), err_msg=key
            )


def test_direct_head_settings_rejects_invalid_morph():
    with pytest.raises(ValueError, match="HeadMorph"):
        HeadSettings(morph={"nose": 0.1})  # type: ignore[arg-type]
