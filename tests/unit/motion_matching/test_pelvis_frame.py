"""HRG-C3 (#13172): pelvis frame re-expression and documented HJC relocation.

Operation (a) is a pure change of coordinates and must leave every physical
quantity unchanged to numerical precision, including MuJoCo inverse dynamics.
Operation (b) is a physical change and must move exactly what it documents.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import pelvis_frame as module
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
SPECS = {
    "anthro_driver": MODELS / "full_body_spec_anthro_driver.json",
    "v1": MODELS / "full_body_spec_v1.json",
}
PELVIS = module.PELVIS_BODY
REL_TOL = 1e-9
# Estimated centres in the Hip frame: lower and narrower than the legacy fit.
CENTRE_R = (-0.0087, -0.0968, -0.0479)
CENTRE_L = (-0.0117, 0.0890, -0.0277)

needs_mujoco = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None, reason="mujoco not installed"
)


def _load(name: str) -> dict:
    return json.loads(SPECS[name].read_text(encoding="utf-8"))


def _rigid(euler_deg: tuple[float, float, float], origin: tuple[float, ...]):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    out[:3, 3] = origin
    return out


def _without_pelvis_shape(spec: dict) -> dict:
    out = copy.deepcopy(spec)
    shapes = out.get("visual_hints", {}).get("shapes", {})
    for key in [k for k in shapes if k.startswith(PELVIS + "/")]:
        del shapes[key]
    return out


# ---------------------------------------------------------------- MuJoCo helpers


def _compiled(spec: dict):
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, meta = export_full_body_mjcf(json.dumps(spec).encode("utf-8"))
    return mujoco.MjModel.from_xml_string(xml), meta


def _state(model, seed: int):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.25, model.nq)
    v = rng.normal(0.0, 1.0, model.nv)
    a = rng.normal(0.0, 3.0, model.nv)
    return q, v, a


SPINE = ("TorsoInput", "SpineInputX", "SpineInputY")


def _snapshot(spec: dict, seed: int = 3, zero: tuple[str, ...] = ()) -> dict:
    """World quantities of a compiled spec at a fixed generalised state."""
    import mujoco

    model, meta = _compiled(spec)
    # Coordinates are matched by name, so spec ordering cannot hide an error.
    data = mujoco.MjData(model)
    q, v, a = _state(model, seed)
    for name in zero:  # the upper body only keeps its pose at zero spine angles
        q[model.joint(name).qposadr[0]] = 0.0
    data.qpos[:], data.qvel[:], data.qacc[:] = q, v, a
    mujoco.mj_inverse(model, data)
    mujoco.mj_comPos(model, data)
    inertia = np.array(
        [
            data.ximat[i].reshape(3, 3)
            @ np.diag(model.body_inertia[i])
            @ data.ximat[i].reshape(3, 3).T
            for i in range(model.nbody)
        ]
    )
    sites = {
        name: data.site_xpos[model.site(site).id].copy()
        for name, site in {**meta["frame_sites"], **meta["contact_sites"]}.items()
    }
    return {
        "sites": sites,
        "com": data.xipos.copy(),
        "mass": model.body_mass.copy(),
        "inertia_world": inertia,
        "subtree_com": data.subtree_com[1].copy(),
        "qfrc_inverse": data.qfrc_inverse.copy(),
        "xanchor": {
            model.joint(j).name: data.xanchor[j].copy() for j in range(model.njnt)
        },
        "model": model,
        "data": data,
    }


def _assert_close(a: np.ndarray, b: np.ndarray, label: str) -> None:
    scale = max(float(np.max(np.abs(b))), 1.0e-12)
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / scale
    assert err <= REL_TOL, f"{label}: relative difference {err:.3e} > {REL_TOL}"


def _assert_invariant(before: dict, after: dict) -> None:
    for key in sorted(before["sites"]):
        _assert_close(after["sites"][key], before["sites"][key], f"site {key}")
    _assert_close(after["com"], before["com"], "body COM")
    _assert_close(after["mass"], before["mass"], "body mass")
    _assert_close(after["inertia_world"], before["inertia_world"], "world inertia")
    _assert_close(after["subtree_com"], before["subtree_com"], "whole-body COM")
    _assert_close(after["qfrc_inverse"], before["qfrc_inverse"], "qfrc_inverse")


# ------------------------------------------------------- (a) pure data transform


def test_reexpress_round_trip_and_composition_are_exact() -> None:
    spec = _load("anthro_driver")
    t1 = _rigid((10.0, -20.0, 35.0), (0.011, -0.004, -0.038))
    t2 = _rigid((-5.0, 8.0, 100.0), (-0.02, 0.03, 0.01))
    stripped = _without_pelvis_shape(spec)
    two_step = module.reexpress_body_frame(
        module.reexpress_body_frame(stripped, PELVIS, t1), PELVIS, t2
    )
    one_step = module.reexpress_body_frame(stripped, PELVIS, t1 @ t2)
    assert _max_doc_difference(two_step, one_step) < 1e-12
    back = module.reexpress_body_frame(
        module.reexpress_body_frame(stripped, PELVIS, t1), PELVIS, np.linalg.inv(t1)
    )
    assert _max_doc_difference(back, stripped) < 1e-12


def _max_doc_difference(a, b) -> float:
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((_max_doc_difference(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list):
        assert len(a) == len(b)
        return max(
            (_max_doc_difference(x, y) for x, y in zip(a, b, strict=True)), default=0.0
        )
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    assert a == b
    return 0.0


def test_reexpress_does_not_mutate_the_input() -> None:
    spec = _load("anthro_driver")
    frozen = copy.deepcopy(spec)
    module.reexpress_body_frame(spec, PELVIS, _rigid((0, 0, 0), (0.0, 0.0, 0.05)))
    assert spec == frozen


def test_reexpress_preconditions() -> None:
    spec = _load("anthro_driver")
    good = _rigid((0, 0, 0), (0.0, 0.0, 0.01))
    with pytest.raises(ValueError, match="unknown body"):
        module.reexpress_body_frame(spec, "nope", good)
    with pytest.raises(ValueError, match="4x4"):
        module.reexpress_body_frame(spec, PELVIS, np.eye(3))
    bad = good.copy()
    bad[:3, :3] *= 1.2
    with pytest.raises(ValueError, match="rotation"):
        module.reexpress_body_frame(spec, PELVIS, bad)
    nan = good.copy()
    nan[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module.reexpress_body_frame(spec, PELVIS, nan)
    with pytest.raises(ValueError, match="axis-aligned"):
        module.reexpress_body_frame(spec, PELVIS, _rigid((0, 0, 20.0), (0, 0, 0)))


# ------------------------------------------------------------ (a) invariance


@needs_mujoco
@pytest.mark.parametrize("name", sorted(SPECS))
@pytest.mark.parametrize(
    "transform",
    [
        _rigid((0, 0, 0), (-0.0102, -0.0039, -0.0378)),
        _rigid((12.0, -25.0, 140.0), (0.021, 0.034, -0.057)),
    ],
    ids=["translation", "rotation_and_translation"],
)
def test_reexpression_leaves_physics_invariant_to_numerical_precision(
    name: str, transform: np.ndarray
) -> None:
    spec = _load(name)
    rotates = not np.allclose(transform[:3, :3], np.eye(3))
    if rotates:
        # Axis-aligned ellipsoids cannot be rotated; see the precondition test.
        spec = _without_pelvis_shape(spec)
    before = _snapshot(spec)
    after = _snapshot(module.reexpress_body_frame(spec, PELVIS, transform))
    assert float(np.max(np.abs(before["qfrc_inverse"]))) > 1.0  # non-trivial load
    _assert_invariant(before, after)


@needs_mujoco
def test_reexpression_actually_changes_the_stored_pelvis_frame() -> None:
    spec = _load("anthro_driver")
    t = _rigid((0, 0, 0), (0.0, 0.0, 0.05))
    out = module.reexpress_body_frame(spec, PELVIS, t)
    hip_old = next(j for j in spec["joints"] if j["name"] == "hip_r")
    hip_new = next(j for j in out["joints"] if j["name"] == "hip_r")
    delta = (
        np.asarray(hip_new["parent_to_base"])[:3, 3]
        - np.asarray(hip_old["parent_to_base"])[:3, 3]
    )
    np.testing.assert_allclose(delta, [0.0, 0.0, -0.05], atol=1e-15)
    assert canonical_sha256(out) != canonical_sha256(spec)


# ------------------------------------------------------ (b) physical relocation


def _revision(name: str = "anthro_driver") -> module.PelvisRevision:
    return module.relocate_pelvis(_load(name), CENTRE_R, CENTRE_L)


def test_relocation_returns_a_new_revision_and_leaves_the_parent_alone() -> None:
    spec = _load("anthro_driver")
    frozen = copy.deepcopy(spec)
    rev = module.relocate_pelvis(spec, CENTRE_R, CENTRE_L)
    assert spec == frozen
    assert rev.canonical_sha256 != canonical_sha256(spec)
    assert rev.parent_canonical_sha256 == canonical_sha256(spec)
    assert rev.spec["schema_version"] == spec["schema_version"]
    block = rev.spec["spec_revision"]
    assert block["issue"] == 13172 and "hip_centres_hip_frame_m" in block
    assert "HRG-C3" in rev.spec["provenance"]


def test_relocation_refuses_a_pelvis_whose_l5s1_joints_are_not_colocated() -> None:
    # The native-derived v1 spine pivots are offset from the torso twist pivot;
    # the design offsets do not describe it, so the revision fails closed.
    with pytest.raises(ValueError, match="colocated"):
        module.relocate_pelvis(_load("v1"), CENTRE_R, CENTRE_L)


@needs_mujoco
def test_relocation_places_hjcs_pelvis_com_and_l5s1_relative_to_the_midpoint() -> None:
    spec = _load("anthro_driver")
    rev = module.relocate_pelvis(spec, CENTRE_R, CENTRE_L)
    after = _snapshot(rev.spec, seed=5)
    model, data = after["model"], after["data"]
    # Zero state: the pelvis is at the root pose, so world = root frame.
    import mujoco

    data.qpos[:] = 0.0
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    r = _anchor(model, data, "hip_flexion_r")
    left = _anchor(model, data, "hip_flexion_l")
    mid = 0.5 * (r + left)
    pelvis = model.body(PELVIS).id
    com = data.xipos[pelvis]
    l5s1 = _anchor(model, data, "SpineInputX")
    hip_frame_origin = data.site_xpos[model.site("native_frame_0").id]
    # HJCs are at the supplied estimate, expressed in the unchanged Hip frame.
    np.testing.assert_allclose(r - hip_frame_origin, CENTRE_R, atol=1e-12)
    np.testing.assert_allclose(left - hip_frame_origin, CENTRE_L, atol=1e-12)
    design = module.PelvisDesign.from_spec(spec)
    # Pelvis COM and L5/S1 sit at their design offsets above the HJC midpoint.
    np.testing.assert_allclose(com - mid, design.com_m, atol=1e-12)
    np.testing.assert_allclose(l5s1 - mid, design.l5s1_m, atol=1e-12)
    assert 0.13 <= float(np.linalg.norm(l5s1 - mid)) <= 0.15


def _anchor(model, data, joint: str) -> np.ndarray:
    return data.xanchor[model.joint(joint).id].copy()


@needs_mujoco
def test_relocation_keeps_markers_upper_body_and_masses_where_the_fit_put_them() -> (
    None
):
    import mujoco

    spec = _load("anthro_driver")
    rev = module.relocate_pelvis(spec, CENTRE_R, CENTRE_L)
    a, b = _snapshot(spec, seed=9, zero=SPINE), _snapshot(rev.spec, seed=9, zero=SPINE)
    names = {
        mujoco.mj_id2name(a["model"], mujoco.mjtObj.mjOBJ_BODY, i): i
        for i in range(a["model"].nbody)
    }
    # Every site on a body above the hips or on the pelvis frame is unchanged:
    # the revision moves joint centres, not the fitted upper body or the markers.
    for site in ("native_frame_0", "native_frame_1", "native_frame_2"):
        np.testing.assert_allclose(
            b["data"].site_xpos[b["model"].site(site).id],
            a["data"].site_xpos[a["model"].site(site).id],
            atol=1e-12,
        )
    upper = [n for n in names if "COMRod" in n or "Head" in n or "UpperArm" in n]
    for n in upper:
        np.testing.assert_allclose(b["com"][names[n]], a["com"][names[n]], atol=1e-12)
        assert b["mass"][names[n]] == pytest.approx(a["mass"][names[n]], rel=1e-14)
    assert float(np.sum(b["mass"])) == pytest.approx(
        float(np.sum(a["mass"])), rel=1e-14
    )
    # The legs moved with the hips (rigidly), the pelvis COM moved with the midpoint.
    assert not np.allclose(a["com"], b["com"])


def test_relocation_report_states_what_moved() -> None:
    spec = _load("anthro_driver")
    rev = module.relocate_pelvis(spec, CENTRE_R, CENTRE_L)
    rep = rev.report
    mid = 0.5 * (np.asarray(CENTRE_R) + np.asarray(CENTRE_L))
    np.testing.assert_allclose(rep["hjc_midpoint_hip_frame_m"], mid, atol=1e-15)
    assert rep["inter_hjc_m"] == pytest.approx(
        float(np.linalg.norm(np.asarray(CENTRE_R) - np.asarray(CENTRE_L)))
    )
    assert rep["l5s1_to_hjc_midpoint_m"] == pytest.approx(0.13818, abs=1e-4)
    assert 0.13 <= rep["l5s1_to_hjc_midpoint_m"] <= 0.15
    assert rep["orientation_policy"].startswith("marker_plane")


@pytest.mark.parametrize(
    "r, l, message",
    [
        ((np.nan, 0, 0), CENTRE_L, "finite"),
        (CENTRE_R, (0.0, 0.0), "3-vector"),
        (CENTRE_R, CENTRE_R, "distinct"),
    ],
)
def test_relocation_validates_centres(r, l, message) -> None:  # noqa: E741
    with pytest.raises(ValueError, match=message):
        module.relocate_pelvis(_load("anthro_driver"), r, l)


def test_relocation_validates_the_spec() -> None:
    spec = _load("anthro_driver")
    spec["joints"] = [j for j in spec["joints"] if j["name"] != "hip_l"]
    with pytest.raises(ValueError, match="hip_l"):
        module.relocate_pelvis(spec, CENTRE_R, CENTRE_L)
    spec = _load("anthro_driver")
    spec["schema_version"] = "full-body-v2"
    with pytest.raises(ValueError, match="full-body-v1"):
        module.relocate_pelvis(spec, CENTRE_R, CENTRE_L)


@pytest.mark.parametrize(
    "status, allowed",
    [("identified", True), ("prior_dominated", False), ("degenerate", False)],
)
def test_estimate_gate_fails_closed(status: str, allowed: bool) -> None:
    est = SimpleNamespace(
        centre_r=CENTRE_R,
        centre_l=CENTRE_L,
        status=status,
        status_reason="test",
        roll_deg=SimpleNamespace(estimate=-6.0, lo=-18.0, hi=6.0),
        height_difference_m=SimpleNamespace(estimate=-0.02, lo=-0.06, hi=0.02),
        inter_hjc_m=SimpleNamespace(estimate=0.187, lo=0.179, hi=0.198),
        prior_weight=0.25,
    )
    spec = _load("anthro_driver")
    if allowed:
        rev = module.relocate_pelvis_from_estimate(spec, est)
        assert rev.spec["spec_revision"]["estimate"]["status"] == status
    else:
        with pytest.raises(ValueError, match=status):
            module.relocate_pelvis_from_estimate(spec, est)
        rev = module.relocate_pelvis_from_estimate(
            spec, est, accepted_statuses=(status,)
        )
        assert rev.spec["spec_revision"]["estimate"]["status"] == status


def test_pelvic_plane_report_is_independent_of_the_hip_line() -> None:
    est = SimpleNamespace(
        centre_r=CENTRE_R,
        centre_l=CENTRE_L,
        roll_deg=SimpleNamespace(estimate=-6.2, lo=-18.2, hi=6.0),
        height_difference_m=SimpleNamespace(estimate=-0.0202, lo=-0.0602, hi=0.0204),
    )
    rep = module.pelvic_plane_report(est)
    # Marker-plane up is the reference; a level hip line is a hypothesis, not a result.
    assert rep["reference"].startswith("waist marker plane")
    assert rep["level_hip_line_excluded"] is False
    assert rep["roll_deg"] == pytest.approx(-6.2)
    tight = SimpleNamespace(
        centre_r=CENTRE_R,
        centre_l=CENTRE_L,
        roll_deg=SimpleNamespace(estimate=8.0, lo=6.0, hi=9.6),
        height_difference_m=SimpleNamespace(estimate=0.029, lo=0.022, hi=0.036),
    )
    assert module.pelvic_plane_report(tight)["level_hip_line_excluded"] is True


def test_write_revision_never_overwrites(tmp_path: Path) -> None:
    rev = _revision()
    out = tmp_path / "rev.json"
    paths = module.write_revision(rev, out)
    assert out.is_file() and paths.receipt.is_file()
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    import hashlib

    assert receipt["spec_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt["parent_spec_canonical_sha256"] == rev.parent_canonical_sha256
    assert receipt["spec_sha256"] != receipt["parent_spec_sha256"]
    with pytest.raises(FileExistsError):
        module.write_revision(rev, out)


@needs_mujoco
def test_invariance_check_detects_a_wrong_reexpression() -> None:
    """Guard against a vacuous invariance test: a bad inertia rotation must fail."""
    spec = _without_pelvis_shape(_load("anthro_driver"))
    transform = _rigid((12.0, -25.0, 140.0), (0.021, 0.034, -0.057))
    good = module.reexpress_body_frame(spec, PELVIS, transform)
    bad = copy.deepcopy(good)
    solid = bad["bodies"][module._body_index(bad, PELVIS)]["solids"][0]
    solid["inertia_com_kg_m2"] = np.asarray(
        _load("anthro_driver")["bodies"][module._body_index(spec, PELVIS)]["solids"][0][
            "inertia_com_kg_m2"
        ]
    ).tolist()  # forgot to rotate the tensor
    before = _snapshot(spec)
    _assert_invariant(before, _snapshot(good))
    with pytest.raises(AssertionError, match="world inertia"):
        _assert_invariant(before, _snapshot(bad))


# ----------------------------------------------- before/after report (MuJoCo)


def _trajectory(spec: dict, frames: int = 3):
    from src.engines.physics_engines.mujoco.python import pelvis_revision_report as rep

    names = list(spec["coordinate_order"])
    rng = np.random.default_rng(11)
    q = rng.normal(0.0, 0.2, (frames, len(names)))
    v = rng.normal(0.0, 1.0, (frames, len(names)))
    a = rng.normal(0.0, 3.0, (frames, len(names)))
    return rep, names, q, v, a


def _flatten(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _flatten(v, f"{prefix}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _flatten(v, f"{prefix}[{i}]")
    elif isinstance(node, float):
        yield prefix, node


def _legacy_hip_calibrated(spec: dict) -> dict:
    """What ``apply_hip_calibration`` leaves: hip joints moved, pelvis not."""
    out = copy.deepcopy(spec)
    for joint in out["joints"]:
        if joint["name"] in module.HIP_JOINTS:
            joint["parent_to_base"][0][3] += -0.01
            joint["parent_to_base"][2][3] += -0.0787
    return out


@needs_mujoco
def test_hip_reaction_wrench_and_lever_arms_are_invariant_under_reexpression() -> None:
    spec = _load("anthro_driver")
    moved = module.reexpress_body_frame(
        spec, PELVIS, _rigid((0, 0, 0), (0.0, 0.0, 0.07))
    )
    rep, names, q, v, a = _trajectory(spec)
    out = rep.compare(spec, moved, names, q, v, a, [0, 1, 2])
    pairs = dict(_flatten(out["before"]))
    other = dict(_flatten(out["after"]))
    assert pairs.keys() == other.keys() and len(pairs) > 50
    scale = max(abs(x) for x in pairs.values())
    worst = max(abs(pairs[k] - other[k]) for k in pairs) / scale
    assert worst <= REL_TOL, worst


@needs_mujoco
def test_relocation_changes_lever_arms_and_l5s1_torque_as_documented() -> None:
    spec = _legacy_hip_calibrated(_load("anthro_driver"))
    rev = module.relocate_pelvis(spec, CENTRE_R, CENTRE_L)
    rep, names, q, v, a = _trajectory(spec)
    zero = [names.index(n) for n in SPINE]
    q[:, zero] = 0.0
    out = rep.compare(spec, rev.spec, names, q, v, a, [0, 1, 2])
    b, c = out["before"][0], out["after"][0]
    # After: the design offset from the HJC midpoint; before: not.
    design = float(np.linalg.norm(module.PelvisDesign.from_spec(spec).com_m))
    assert c["lever_arms"]["pelvis_com_to_hjc_midpoint_m"] == pytest.approx(
        design, abs=1e-9
    )
    assert b["lever_arms"]["pelvis_com_to_hjc_midpoint_m"] != pytest.approx(
        c["lever_arms"]["pelvis_com_to_hjc_midpoint_m"], abs=1e-3
    )
    # The upper body moves rigidly about a different L5/S1 pivot, so its torque changes.
    assert b["l5s1_torque_nm"]["SpineInputX"] != pytest.approx(
        c["l5s1_torque_nm"]["SpineInputX"], rel=1e-6
    )
