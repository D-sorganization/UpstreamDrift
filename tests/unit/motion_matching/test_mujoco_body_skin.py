"""The skinned CC0 body in the MuJoCo export: default off, appearance only (#15299)."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python import body_skin_layer as layer
from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import body_skin_fit as fit
from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import skinned_wrist_sections as sws
from src.shared.python.model_appearance import body_skin_export as export
from src.shared.python.model_appearance import document_from_dict, library

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (
        ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_text(encoding="utf-8")
)
ADDRESS = json.loads(
    (ROOT / "tests/fixtures/model_appearance/address_pose.json").read_text()
)
SETTINGS = export.load_export_settings()
DOCUMENT = document_from_dict({"schema_version": "appearance-v1"})
LAYERS = ("skin", "shirt", "trousers")


def _export(backend: str | None) -> tuple[str, dict[str, Any]]:
    spec = copy.deepcopy(SPEC)
    if backend is not None:
        spec["subject"]["body_visual_backend"] = backend
    return exporter.export_full_body_mjcf(
        json.dumps(spec).encode(), appearance=DOCUMENT
    )


@pytest.fixture(scope="module")
def loft() -> Any:
    return mujoco.MjModel.from_xml_string(_export(None)[0])


@pytest.fixture(scope="module")
def skinned() -> tuple[Any, dict[str, Any]]:
    xml, summary = _export(export.SKINNED_BACKEND)
    return mujoco.MjModel.from_xml_string(xml), summary["visual_layer"]["body_skin"]


def test_the_loft_body_is_the_default_and_has_no_skin() -> None:
    xml, summary = _export(None)
    assert not export.body_skin_enabled(SPEC)
    assert "<deformable" not in xml and "<skin" not in xml
    assert "body_skin" not in summary["visual_layer"]
    assert _export(export.LOFT_BACKEND)[0] == xml
    with pytest.raises(ValueError, match="body_visual_backend"):
        _export("marble")


def test_the_skin_changes_no_physics(loft: Any, skinned: tuple[Any, Any]) -> None:
    model = skinned[0]
    sizes = ("nq", "nv", "nbody", "njnt", "nu", "neq")
    assert [getattr(model, n) for n in sizes] == [getattr(loft, n) for n in sizes]
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos"):
        assert np.array_equal(getattr(model, name), getattr(loft, name)), name


def test_three_layers_follow_the_bodies(skinned: tuple[Any, Any]) -> None:
    model, summary = skinned
    assert model.nskin == len(LAYERS)
    assert [model.skin(i).name for i in range(3)] == [f"body_skin_{n}" for n in LAYERS]
    assert summary["bones"] == {"skin": 12, "shirt": 11, "trousers": 6}
    assert model.nskinbone == sum(summary["bones"].values())
    assert model.nskinvert == sum(summary["vertices"].values())
    roles = summary["bodies"]
    assert library.classify_body(roles["thorax"]) == "torso"
    assert roles["upper_arm_l"].endswith("LUpperArm")
    assert roles["shank_r"] == "tibia_r"


def test_the_skin_replaces_the_loft_and_the_neck(
    loft: Any, skinned: tuple[Any, Any]
) -> None:
    def visuals(model: Any) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for index in range(model.ngeom):
            geom = model.geom(index)
            part = library.classify_body(model.body(geom.bodyid[0]).name or "world")
            if geom.name.startswith(("visual_vmesh", "visual_head")):
                # the number in a mesh geom's name counts the drawn bodies
                out.setdefault(part, []).append(re.sub(r"vmesh_\d+_", "", geom.name))
        return out

    before, after = visuals(loft), visuals(skinned[0])
    for part in SETTINGS.covered_parts:
        assert before[part] and part not in after, part
    for part in ("foot", "club"):
        assert after[part] == before[part]
    necks = [n for n in before["head"] if n not in after["head"]]
    assert necks and all("neck" in name for name in necks)
    assert len(after["head"]) == len(before["head"]) - len(necks)


def test_the_skinned_model_steps_at_swing_poses(skinned: tuple[Any, Any]) -> None:
    model = skinned[0]
    data = mujoco.MjData(model)
    for pose in ({}, SETTINGS.bind_pose, ADDRESS):
        mujoco.mj_resetData(model, data)
        for name, value in pose.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.xpos).all()
        assert not data.warning.number.any()


def test_the_forearms_end_under_the_skinned_hands_wrists() -> None:
    spec = copy.deepcopy(SPEC)
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    club = ca.assembly_from_spec(spec)
    by_body = layer.reference_frames(spec, SETTINGS.bind_pose)
    roles = export.body_roles(spec, {b: f.translation for b, f in by_body.items()})
    frames = {role: by_body[body] for role, body in roles.items()}
    settings = fit.load_fit_settings()
    skin = fit.load_body_skin()
    fitted = fit.fit_body_skin(skin, export.subject_lengths(frames), settings)
    binds = export.bind_poses(frames, fitted)
    assert layer.wrist_rings(SPEC, club, roles, binds) == {}  # other hands
    rings = layer.wrist_rings(spec, club, roles, binds)
    sections = sws.forearm_wrist_sections(spec, club)
    assert sorted(rings) == ["l", "r"] and len(sections) == 2
    matched = fit.match_wrists(skin, fitted, rings, settings)
    for side, ring in rings.items():
        section = sections[roles[f"forearm_{side}"]]
        assert ring.half_width_m == section.half_width_m > ring.half_depth_m
        elbow, wrist = (fitted.joints[f"{j}_{side}"] for j in ("elbow", "wrist"))
        bone = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        assert abs(float(ring.width_axis @ bone)) < 0.2  # across the forearm
        columns = [g == f"forearm_distal_{side}" for g in skin.groups]
        end = skin.weights[:, columns].sum(axis=1) > 0.5
        end &= (matched.vertices - wrist) @ bone > -settings.wrist_match_full_m
        off = matched.vertices[end] - wrist - ring.centre_m
        radius = np.linalg.norm(off - np.outer(off @ bone, bone), axis=1)
        # The ring lies inside the hand's section, by the named 2 mm at most
        # off its two half axes.
        assert radius.max() <= ring.half_width_m - settings.wrist_under_m + 3e-4
        assert radius.min() >= ring.half_depth_m - settings.wrist_under_m - 3e-4
