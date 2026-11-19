"""Trunk solids follow a profile, and shoulders are not balls (#15279)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.body_meshes import BodyMesh, body_meshes
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.model_appearance.torso_profile import (
    TorsoFrame,
    TorsoProfile,
    TorsoStation,
    torso_mesh,
)
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPEC = json.loads((MODELS / "full_body_spec_anthro_driver.json").read_text("utf-8"))
DOC = document_from_dict({"schema_version": "appearance-v1"})
_FRAME = TorsoFrame(
    np.zeros(3), np.array([0.0, 0.0, 0.25]), np.array([0.0, 1.0, 0.0]), 0.16, 0.10
)
_PROFILE = TorsoProfile(
    (
        TorsoStation(-0.1, 0.8, 0.9),
        TorsoStation(0.5, 1.0, 1.0),
        TorsoStation(1.1, 0.5, 0.6),
    ),
    squareness=2.6,
)


def test_the_solid_has_the_profiles_width_and_depth_at_every_ring() -> None:
    mesh = torso_mesh(_FRAME, _PROFILE)
    heights = np.unique(np.round(mesh.vertices[:-2, 2], 9))  # the poles are last
    assert len(heights) == _PROFILE.rings
    assert heights[0] == pytest.approx(-0.1 * 0.25)
    assert heights[-1] == pytest.approx(1.1 * 0.25)
    widths, depths = _PROFILE.scales_at(heights / 0.25)
    for height, width, depth in zip(heights, widths, depths, strict=True):
        ring = mesh.vertices[np.isclose(mesh.vertices[:, 2], height, atol=1e-8)]
        assert np.abs(ring[:, 1]).max() == pytest.approx(0.16 * width, rel=1e-6)
        assert np.abs(ring[:, 0]).max() == pytest.approx(0.10 * depth, rel=1e-6)
    first, last = _PROFILE.scales_at([-0.1, 1.1])
    assert first == pytest.approx([0.8, 0.5]) and last == pytest.approx([0.9, 0.6])
    assert np.abs(mesh.vertices[:, 1]).max() == pytest.approx(0.16, rel=1e-3)
    thick = torso_mesh(_FRAME, _PROFILE, thickness=1.08)
    assert np.abs(thick.vertices[:, 1]).max() == pytest.approx(0.16 * 1.08, rel=1e-3)


def test_the_solid_is_closed_and_mirrors_left_to_right() -> None:
    mesh = torso_mesh(_FRAME, _PROFILE)
    edges = np.sort(
        np.vstack([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), "an edge is not shared by exactly two faces"
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    assert float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) > 0.0
    mirrored = mesh.vertices * np.array([1.0, -1.0, 1.0])
    for point in mirrored[::7]:
        assert np.linalg.norm(mesh.vertices - point, axis=1).min() < 1e-9


@pytest.mark.parametrize(
    "stations, squareness",
    [
        (((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)), 2.0),
        (((0.0, 1.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0)), 2.0),
        (((0.0, 1.0, 1.0), (0.5, -1.0, 1.0), (1.0, 1.0, 1.0)), 2.0),
        (((0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)), 1.5),
    ],
)
def test_bad_profiles_are_refused(stations: Any, squareness: float) -> None:
    with pytest.raises(ValueError, match="torso profile"):
        TorsoProfile(tuple(TorsoStation(*row) for row in stations), squareness)


def test_bad_frames_are_refused() -> None:
    along = TorsoFrame(_FRAME.start, _FRAME.end, np.array([0.0, 0.0, 1.0]), 0.16, 0.1)
    with pytest.raises(ValueError, match="width axis"):
        torso_mesh(along, _PROFILE)
    flat = TorsoFrame(_FRAME.start, _FRAME.end, _FRAME.width_axis, 0.16, 0.0)
    with pytest.raises(ValueError, match="positive finite"):
        torso_mesh(flat, _PROFILE)
    with pytest.raises(ValueError, match="positive finite"):
        torso_mesh(_FRAME, _PROFILE, thickness=0.0)


def _meshes(part: str, label: str) -> list[BodyMesh]:
    return [
        m
        for m in body_meshes(SPEC, DOC)
        if library.classify_body(m.body) == part and m.label == label
    ]


def _shoulder_line_height(body: str, up: np.ndarray) -> float:
    """Height of the shoulder joints along ``up`` in the thorax body frame."""
    poses = spec_body_poses(SPEC)
    joints = [
        j for j in SPEC["joints"] if library.classify_body(j["child"]) == "upper_arm"
    ]
    world = np.mean(
        [
            (poses[j["parent"]] @ np.asarray(j["parent_to_base"], float))[:3, 3]
            for j in joints
        ],
        axis=0,
    )
    pose = poses[body]
    return float((pose[:3, :3].T @ (world - pose[:3, 3])) @ up)


def test_the_trunk_keeps_its_width_at_its_joints() -> None:
    """A spindle pinches to nothing at both ends: two of them are a snowman."""
    sections = body_sections(SPEC)
    spines = {
        cap.body: cap
        for cap in derive_visual_skeleton(SPEC).capsules
        if library.classify_body(cap.body) == "torso"
    }
    torsos = _meshes("torso", "skin0")
    assert len(torsos) == 2
    for skin in torsos:
        verts = skin.mesh_in_body_frame.vertices
        section, cap = sections[skin.body], spines[skin.body]
        up = section.trunk_up_axis()
        height = verts @ up
        low, high = np.asarray(cap.start_m) @ up, np.asarray(cap.end_m) @ up
        widest = np.abs(verts @ section.ml_axis).max()
        # The upper plane is the shoulder line (the shoulder joints' height):
        # the collar is slanted now, so the spine-end plane is no longer a
        # width plane of the thorax (#15292).
        planes = [min(low, high), max(low, high)]
        if skin.body.endswith("COMRod"):
            planes[1] = _shoulder_line_height(skin.body, up)
        for plane in planes:
            band = np.abs(height - plane) < 0.02
            assert band.any(), f"{skin.body}: no skin at its joint plane"
            width = np.abs(verts[band] @ section.ml_axis).max()
            assert width > 0.8 * widest, (skin.body, width, widest)


def test_the_thorax_narrows_into_the_neck_and_the_abdomen_does_not() -> None:
    sections = body_sections(SPEC)
    tops = {}
    for skin in _meshes("torso", "skin0"):
        verts = skin.mesh_in_body_frame.vertices
        height = verts @ sections[skin.body].trunk_up_axis()
        top = verts[height > height.max() - 0.01]
        tops[skin.body] = (
            np.abs(top @ sections[skin.body].ml_axis).max()
            / np.abs(verts @ sections[skin.body].ml_axis).max()
        )
    thorax = next(body for body in tops if body.endswith("COMRod"))
    abdomen = next(body for body in tops if body != thorax)
    assert tops[thorax] < 0.4, "the thorax ends as wide as the chest"
    assert tops[abdomen] > 0.5, "the abdomen is drawn with the thorax profile"


def test_the_shoulder_link_is_a_yoke_flatter_than_it_is_deep() -> None:
    links = _meshes("shoulder", "skin0")
    assert len(links) == 2
    up = np.asarray(library.SHOULDER_UP_AXIS)
    for link in links:
        verts = link.mesh_in_body_frame.vertices
        # Its own axis frame: the droop tilts the whole yoke, not its section
        # (the tilt is the longest principal axis, so it is excluded here).
        _, _, axes = np.linalg.svd(verts - verts.mean(axis=0), full_matrices=False)
        deep_axis = axes[1 + int(np.argmax(np.abs(axes[1:, 0])))]
        tall_axis = axes[1 + int(np.argmin(np.abs(axes[1:, 0])))]
        tall = float(np.ptp(verts @ tall_axis))
        deep = float(np.ptp(verts @ deep_axis))
        assert tall == pytest.approx(library.SHOULDER_ASPECT * deep, rel=0.05)
        # It starts beside the neck, not at the middle of the chest.
        assert np.abs(verts[:, 1]).min() > 0.02


def test_the_shoulder_blend_is_longer_down_the_arm_than_across_it() -> None:
    blends = _meshes("upper_arm", "blend0")
    assert len(blends) == 2
    for blend in blends:
        verts = blend.mesh_in_body_frame.vertices
        down, across = float(np.ptp(verts[:, 2])), float(np.ptp(verts[:, 1]))
        ratio = library.DELTOID_HALF_SIZES[1] / library.DELTOID_HALF_SIZES[0]
        assert down == pytest.approx(ratio * across, rel=0.05)
        assert ratio > 1.3


def test_a_profile_clipped_at_its_joint_plane_keeps_its_section_there() -> None:
    clipped = _PROFILE.clipped(0.0)
    assert clipped.stations[0].along == 0.0
    assert [s.along for s in clipped.stations[1:]] == [0.5, 1.1]
    width, depth = _PROFILE.scales_at([0.0])
    assert clipped.stations[0].width == pytest.approx(float(width[0]))
    assert clipped.stations[0].depth == pytest.approx(float(depth[0]))
    assert _PROFILE.clipped(-0.5) is _PROFILE


def test_a_station_slant_drops_the_front_and_keeps_the_back() -> None:
    flat = torso_mesh(_FRAME, _PROFILE)
    slanted_profile = TorsoProfile(
        (*_PROFILE.stations[:2], dataclasses.replace(_PROFILE.stations[2], slant=0.4)),
        squareness=2.6,
    )
    frame = dataclasses.replace(_FRAME, front_axis=np.array([1.0, 0.0, 0.0]))
    slanted = torso_mesh(frame, slanted_profile)
    top = slanted.vertices[:-2][-slanted_profile.sides :]
    flat_top = flat.vertices[:-2][-slanted_profile.sides :]
    drop = flat_top[:, 2] - top[:, 2]
    assert drop[np.argmax(top[:, 0])] == pytest.approx(0.4 * 0.25, rel=1e-6)
    assert drop[np.argmin(top[:, 0])] == pytest.approx(0.0, abs=1e-9)
    assert np.all(drop >= -1e-12)
    mirrored = slanted.vertices * np.array([1.0, -1.0, 1.0])
    for point in mirrored[::7]:
        assert np.linalg.norm(slanted.vertices - point, axis=1).min() < 1e-9
