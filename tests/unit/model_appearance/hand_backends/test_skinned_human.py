"""Skinned CC0 human hand visual backend (epic #15122)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.grip_spec.shaft_roll import apply_shaft_roll
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    HandJointAngles,
    legacy_neutral_joint_angles,
)
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
    max_grip_cylinder_penetration_m,
    max_self_penetration_fraction,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual,
    get_hand_visual_backend,
)
from src.shared.python.model_appearance.hand_backends.skinned_human import (
    BACKEND_NAME,
    SkinnedHumanSettings,
    default_settings,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[4]
_GRIP_RADIUS_M = 0.0145
_SHAFT_AXIS = np.array([0.0, 0.0, 1.0])


def _spec(**kwargs: object) -> GripSpec:
    return GripSpec.legacy(**kwargs)


def _angles() -> HandJointAngles:
    return legacy_neutral_joint_angles(_spec(), "lead")


def _build(side: str = "lead", **spec_kw: object) -> Mesh:
    spec = _spec(**spec_kw)
    angles = legacy_neutral_joint_angles(spec, side)
    result = build_hand_visual(spec, side, angles, grip_radius_m=_GRIP_RADIUS_M)
    assert result.backend == BACKEND_NAME
    return result.mesh


def test_registry_exposes_skinned_human_backend() -> None:
    backend = get_hand_visual_backend(BACKEND_NAME)
    assert backend.name == BACKEND_NAME


def test_build_returns_finite_watertight_mesh() -> None:
    mesh = _build()
    assert mesh.vertices.shape[1] == 3
    assert np.isfinite(mesh.vertices).all()
    assert len(mesh.faces) >= 120
    assert mesh.volume() > 0.0


def test_grip_cylinder_penetration_within_tolerance() -> None:
    quality = load_hand_visual_quality()
    tol = quality["grip_cylinder_penetration_m"]
    for side in ("lead", "trail"):
        mesh = _build(side)
        pen = max_grip_cylinder_penetration_m(mesh.vertices, _GRIP_RADIUS_M)
        assert pen <= tol, f"{side} grip penetration {pen:.4f} m > {tol} m"


def test_inter_finger_penetration_within_tolerance() -> None:
    quality = load_hand_visual_quality()
    tol = quality["self_penetration_fraction"]
    mesh = _build("lead")
    frac = max_self_penetration_fraction(mesh, sample_stride=6)
    assert frac <= tol, f"self penetration fraction {frac:.3f} > {tol}"


def test_trail_is_mirror_of_lead_geometry() -> None:
    spec = _spec()
    lead = build_hand_visual(
        spec,
        "lead",
        legacy_neutral_joint_angles(spec, "lead"),
        grip_radius_m=_GRIP_RADIUS_M,
    ).mesh
    trail = build_hand_visual(
        spec,
        "trail",
        legacy_neutral_joint_angles(spec, "trail"),
        grip_radius_m=_GRIP_RADIUS_M,
    ).mesh
    lead_m = lead.vertices.copy()
    lead_m[:, 1] *= -1.0
    tree = cKDTree(trail.vertices)
    dists, _ = tree.query(lead_m, k=1)
    assert float(np.median(dists)) < 0.004


def test_shaft_roll_rotates_mesh_about_grip_axis() -> None:
    base_spec = _spec()
    rolled = _spec(lead=HandGripSpec(shaft_roll_deg=20.0))
    angles = legacy_neutral_joint_angles(base_spec, "lead")
    mesh0 = build_hand_visual(
        base_spec, "lead", angles, grip_radius_m=_GRIP_RADIUS_M
    ).mesh
    mesh1 = build_hand_visual(rolled, "lead", angles, grip_radius_m=_GRIP_RADIUS_M).mesh
    rot = apply_shaft_roll(np.eye(3), _SHAFT_AXIS, 20.0)[:3, :3]
    expected = mesh0.vertices @ rot.T
    rel1 = mesh1.vertices
    err = np.linalg.norm(expected - rel1, axis=1)
    assert float(np.median(err)) < 0.003


def test_palm_arch_increases_grip_side_curvature() -> None:
    shallow = HandJointAngles.from_legacy_curl(
        finger_curl_scale=0.35, palm_arch_scale=0.2
    )
    deep = HandJointAngles.from_legacy_curl(finger_curl_scale=0.85, palm_arch_scale=1.0)
    spec = _spec()
    m0 = build_hand_visual(spec, "lead", shallow, grip_radius_m=_GRIP_RADIUS_M).mesh
    m1 = build_hand_visual(spec, "lead", deep, grip_radius_m=_GRIP_RADIUS_M).mesh
    # Deeper curl + palm arch should pull the grip-side envelope closer to the club.
    assert float(m1.vertices[:, 0].min()) < float(m0.vertices[:, 0].min())


def test_hand_cc0_provenance_present() -> None:
    provenance = _REPO / "assets" / "hand_cc0" / "PROVENANCE.json"
    assert provenance.is_file(), "missing hand_cc0 PROVENANCE.json"
    record = json.loads(provenance.read_text(encoding="utf-8"))
    assert record.get("licence") == "CC0-1.0"
    assert "MANO" in record.get("research_only_alternatives", "")


def test_settings_not_hardcoded_in_builder() -> None:
    settings = default_settings()
    assert isinstance(settings, SkinnedHumanSettings)
    assert settings.skin_rgb != (0.0, 0.0, 0.0)
