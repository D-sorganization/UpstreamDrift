"""Procedural feet and shoes (HRG-F1, #13188, epic #13160).

The foot used to be three generic capsule lofts (radius 0.040 m, aspect 0.7).
These tests pin the replacement: a dimensioned shoe last sized from the spec's
foot length, a cited breadth ratio and the physics contact spheres, which are
never moved to fit it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.shared.python.grip_contact.grip_mesh import validate_closed_mesh
from src.shared.python.model_appearance import document_from_dict, foot, library
from src.shared.python.model_appearance.body_meshes import BodyMesh, body_meshes
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.skinned_body import spec_body_poses

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
DOC = document_from_dict({"schema_version": "appearance-v1"})
SIDES = ("r", "l")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def meshes(spec: dict[str, Any]) -> list[BodyMesh]:
    return body_meshes(spec, DOC)


def _parts(meshes: list[BodyMesh], body: str, prefix: str) -> list[BodyMesh]:
    return [m for m in meshes if m.body == body and m.label.startswith(prefix)]


class _Frame:
    """Foot basis (forward, left, up) of a calcn body and its coordinates."""

    def __init__(self, spec: dict[str, Any], side: str) -> None:
        self.spec = spec
        self.side = side
        self.fwd, self.up, self.left = foot.sole_frame(
            spec, body_sections(spec)[f"calcn_{side}"], f"calcn_{side}"
        )
        self.poses = spec_body_poses(spec)

    def to_calcn(self, body: str, points: np.ndarray) -> np.ndarray:
        """Points of ``body`` expressed in the calcn spec frame."""
        calcn = self.poses[f"calcn_{self.side}"]
        rel = np.linalg.inv(calcn) @ self.poses[body]
        return points @ rel[:3, :3].T + rel[:3, 3]

    def local(self, points: np.ndarray) -> np.ndarray:
        """(forward, up, left) coordinates of calcn-frame points."""
        return np.stack([points @ self.fwd, points @ self.up, points @ self.left], 1)


def _shoe_points(
    spec: dict[str, Any], meshes: list[BodyMesh], side: str
) -> tuple[_Frame, np.ndarray, list[Any]]:
    frame = _Frame(spec, side)
    tris, verts = [], []
    for body in (f"calcn_{side}", f"toes_{side}"):
        for m in _parts(meshes, body, "garment"):
            v = frame.to_calcn(body, m.mesh_in_body_frame.vertices)
            tris.append(v[m.mesh_in_body_frame.faces])
            verts.append(v)
    return frame, np.vstack(verts), tris


def _winding_inside(tris: np.ndarray, point: np.ndarray) -> bool:
    """Generalised winding number of one closed shell about ``point``."""
    a, b, c = (tris[:, i] - point for i in range(3))
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    return bool(abs(2.0 * np.arctan2(num, den).sum() / (4.0 * np.pi)) > 0.5)


def _spheres(spec: dict[str, Any], side: str) -> list[dict[str, Any]]:
    return [s for s in spec["contact"]["spheres"] if s["body"] == f"calcn_{side}"]


def test_foot_model_default_is_procedural_and_cc0_is_unavailable(
    spec: dict[str, Any],
) -> None:
    assert DOC.foot.model == "procedural"
    reserved = document_from_dict(
        {"schema_version": "appearance-v1", "foot": {"model": "cc0_mesh"}}
    )
    with pytest.raises(foot.FootModelUnavailableError, match="cc0_mesh"):
        body_meshes(spec, reserved)


def test_foot_meshes_are_not_the_generic_capsule_loft(
    meshes: list[BodyMesh],
) -> None:
    for side in SIDES:
        for body in (f"calcn_{side}", f"toes_{side}"):
            assert _parts(meshes, body, "skin"), body
        assert _parts(meshes, f"talus_{side}", ""), "ankle stub missing"
        assert _parts(meshes, f"calcn_{side}", "garment")
        assert _parts(meshes, f"toes_{side}", "garment")


@pytest.mark.parametrize("side", SIDES)
def test_every_foot_mesh_is_closed_with_outward_normals(
    meshes: list[BodyMesh], side: str
) -> None:
    checked = 0
    for body in (f"talus_{side}", f"calcn_{side}", f"toes_{side}"):
        for m in _parts(meshes, body, ""):
            mesh = m.mesh_in_body_frame
            validate_closed_mesh(mesh.vertices, mesh.faces)
            checked += 1
    assert checked >= 5


@pytest.mark.parametrize("side", SIDES)
def test_shoe_length_matches_the_spec_foot_length(
    spec: dict[str, Any], meshes: list[BodyMesh], side: str
) -> None:
    frame, verts, _ = _shoe_points(spec, meshes, side)
    length = float(np.ptp(frame.local(verts)[:, 0]))
    target = spec["anthropometry"]["segments"]["foot"]["length_m"]
    assert abs(length - target) < 1e-3, f"{1e3 * (length - target):.2f} mm"


@pytest.mark.parametrize("side", SIDES)
def test_shoe_breadth_follows_the_cited_length_ratio(
    spec: dict[str, Any], meshes: list[BodyMesh], side: str
) -> None:
    frame, verts, _ = _shoe_points(spec, meshes, side)
    breadth = float(np.ptp(frame.local(verts)[:, 2]))
    target = (
        foot.BREADTH_OVER_LENGTH * spec["anthropometry"]["segments"]["foot"]["length_m"]
    )
    assert abs(breadth - target) < 1e-3, f"{1e3 * (breadth - target):.2f} mm"


@pytest.mark.parametrize("side", SIDES)
def test_each_contact_sphere_centre_is_inside_the_shoe(
    spec: dict[str, Any], meshes: list[BodyMesh], side: str
) -> None:
    _, _, tris = _shoe_points(spec, meshes, side)
    for sph in _spheres(spec, side):
        centre = np.asarray(sph["position_m"], float)
        # A centre inside either shell (calcn upper or toe box) is inside.
        assert any(_winding_inside(t, centre) for t in tris), sph["name"]


@pytest.mark.parametrize("side", SIDES)
def test_sphere_nadirs_lie_within_2mm_of_the_sole_plane(
    spec: dict[str, Any], meshes: list[BodyMesh], side: str
) -> None:
    frame, verts, _ = _shoe_points(spec, meshes, side)
    loc = frame.local(verts)
    sole = loc  # the lowest point of a thin band under the sphere is the sole
    for sph in _spheres(spec, side):
        centre = np.asarray(sph["position_m"], float)
        a, h = centre @ frame.fwd, centre @ frame.up
        near = sole[(abs(sole[:, 0] - a) < 0.006) & (abs(sole[:, 2]) < 0.012)]
        assert len(near), sph["name"]
        gap = float(near[:, 1].min()) - (h - sph["radius_m"])
        assert abs(gap) <= 2e-3, f"{sph['name']}: {1e3 * gap:.2f} mm"


@pytest.mark.parametrize("side", SIDES)
def test_neutral_stance_sole_clearance_is_zero_to_two_mm(
    spec: dict[str, Any], meshes: list[BodyMesh], side: str
) -> None:
    frame, verts, _ = _shoe_points(spec, meshes, side)
    ground = min(
        float(np.asarray(s["position_m"]) @ frame.up) - s["radius_m"]
        for s in _spheres(spec, side)
    )
    clearance = float((verts @ frame.up).min()) - ground
    assert 0.0 <= clearance <= 2e-3, f"{1e3 * clearance:.2f} mm"


def test_left_shoe_is_the_mirror_of_the_right(spec: dict[str, Any]) -> None:
    # The spec's two MTP joints differ by calibration (1.9 mm); equalise them so
    # only the shoe's own handedness is under test.
    spec = copy.deepcopy(spec)
    joints = {j["name"]: j for j in spec["joints"]}
    joints["mtp_l"]["parent_to_base"][0][3] = joints["mtp_r"]["parent_to_base"][0][3]
    meshes = body_meshes(spec, DOC)
    loc = {}
    for side in SIDES:
        frame, verts, _ = _shoe_points(spec, meshes, side)
        loc[side] = frame.local(verts)
    mirrored = loc["l"] * np.array([1.0, 1.0, -1.0])
    gaps, _ = cKDTree(loc["r"]).query(mirrored)
    assert gaps.max() < 1e-5, f"{1e3 * gaps.max():.4f} mm"


def test_shoe_has_a_medial_arch_waist_and_a_lateral_ball(
    spec: dict[str, Any], meshes: list[BodyMesh]
) -> None:
    frame, verts, _ = _shoe_points(spec, meshes, "r")
    loc = frame.local(verts)
    # Right foot: medial is the subject's left (+left axis).
    a0, a1 = loc[:, 0].min(), loc[:, 0].max()

    def extent(frac: float) -> tuple[float, float]:
        band = loc[abs(loc[:, 0] - (a0 + frac * (a1 - a0))) < 0.004]
        return float(band[:, 2].max()), float((-band[:, 2]).max())

    waist_med, waist_lat = extent(0.38)
    ball_med, ball_lat = extent(0.70)
    assert waist_med < 0.85 * ball_med, "medial arch waist missing"
    assert waist_med < waist_lat, "arch must indent the medial side more"


def test_shoe_rises_to_an_instep_ramp_then_falls_to_the_toe_box(
    spec: dict[str, Any], meshes: list[BodyMesh]
) -> None:
    frame, verts, _ = _shoe_points(spec, meshes, "r")
    loc = frame.local(verts)
    a0, a1 = loc[:, 0].min(), loc[:, 0].max()

    def top(frac: float) -> float:
        band = loc[abs(loc[:, 0] - (a0 + frac * (a1 - a0))) < 0.004]
        return float(band[:, 1].max() - band[:, 1].min())

    assert top(0.42) > top(0.70) > top(0.93), (top(0.42), top(0.70), top(0.93))
    assert top(0.08) > top(0.70), "heel counter must stand above the ball"


def test_toe_box_overlaps_the_calcn_shell_across_the_mtp_hinge(
    spec: dict[str, Any], meshes: list[BodyMesh]
) -> None:
    frame = _Frame(spec, "r")
    calcn = _parts(meshes, "calcn_r", "garment")[0].mesh_in_body_frame.vertices
    toes = frame.to_calcn(
        "toes_r", _parts(meshes, "toes_r", "garment")[0].mesh_in_body_frame.vertices
    )
    mtp = (np.linalg.inv(frame.poses["calcn_r"]) @ frame.poses["toes_r"])[:3, 3]
    a_mtp = float(mtp @ frame.fwd)
    assert (calcn @ frame.fwd).max() > a_mtp + 0.005
    assert (toes @ frame.fwd).min() < a_mtp - 0.005


def test_physics_and_spec_are_untouched(spec: dict[str, Any]) -> None:
    before = copy.deepcopy(spec)
    body_meshes(spec, DOC)
    assert spec == before


def test_unavailable_foot_length_keeps_the_legacy_loft(spec: dict[str, Any]) -> None:
    legacy = copy.deepcopy(spec)
    del legacy["anthropometry"]["segments"]["foot"]
    out = body_meshes(legacy, DOC)
    assert [m.label for m in out if m.body == "calcn_r"] == ["skin0", "garment0"]


def test_foot_parts_reject_a_bad_length(spec: dict[str, Any]) -> None:
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="length"):
            foot.foot_parts(spec, body_sections(spec), length_m=bad)


def test_measured_length_overrides_the_spec_length(spec: dict[str, Any]) -> None:
    parts = foot.foot_parts(spec, body_sections(spec), length_m=0.27)
    frame = _Frame(spec, "r")
    pts = np.vstack(
        [
            frame.to_calcn(b, p.mesh.vertices)
            for b in ("calcn_r", "toes_r")
            for p in parts[b]
            if p.role == "garment"
        ]
    )
    assert abs(float(np.ptp(frame.local(pts)[:, 0])) - 0.27) < 1e-3


def test_shoe_colour_stays_a_garment(meshes: list[BodyMesh]) -> None:
    garment = library.CLOTHING[DOC.clothing]["foot"]
    for m in _parts(meshes, "calcn_r", "garment") + _parts(meshes, "toes_r", ""):
        if m.label.startswith("garment"):
            assert m.material_name == garment
        else:
            assert m.material_name == DOC.skin_tone


def test_ankle_stub_wears_the_trouser_leg_or_shows_skin(
    spec: dict[str, Any], meshes: list[BodyMesh]
) -> None:
    (stub,) = _parts(meshes, "talus_r", "")
    assert stub.material_name == library.CLOTHING[DOC.clothing]["shin"]
    shorts = document_from_dict(
        {"schema_version": "appearance-v1", "clothing": "golf_polo_shorts"}
    )
    (bare,) = [m for m in body_meshes(spec, shorts) if m.body == "talus_r"]
    assert bare.label.startswith("skin") and bare.material_name == shorts.skin_tone
