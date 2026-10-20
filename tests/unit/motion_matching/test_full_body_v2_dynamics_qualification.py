"""Cross-engine inertia and gravity qualification of full-body-v2 (HRG-Q1, #13176).

The reference is computed from the specification alone, independently of any
engine: the forward kinematics compose the spec's own joint transforms and
primitives (as ``spec_body_poses`` does), and each primitive's world screw
axis gives an exact geometric Jacobian of every solid. From the spec's solid
masses, centres of mass and inertias the reference joint-space mass matrix is

    M = sum_s m_s Jv_s^T Jv_s + Jw_s^T (R_s I_s R_s^T) Jw_s

and the gravity term (the bias force at zero velocity) is

    g = -sum_s m_s Jv_s^T g_world.

Nothing is differentiated numerically, so the only error sources are the
exporters' 17-significant-digit text and double-precision composition. The
worst error measured over 120 seeded configurations on MuJoCo 3 is about
3e-14 kg m^2 on a mass matrix whose largest entry is 79 kg (relative 4e-16)
and 1e-13 N m on a gravity vector whose largest entry is 775 N (relative
1.3e-16). ``REL_TOL`` (1e-12 of the largest reference entry) keeps the same
precision budget as the HRG-D2 FK qualification and is never loosened.

Covered here: MuJoCo (MJCF export) and Pinocchio (native model). Drake,
OpenSim and MyoSuite inertia/gravity parity, limits, closure, contact and
inverse dynamics on a reference trajectory remain open on #13176.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance.skinned_body import _primitive_transform
from src.shared.python.motion_matching import full_body_v2_geometry as geo

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
N_CONFIGS = 120
SEED = 1317601
REL_TOL = 1e-12
_AXIS = {"x": 0, "y": 1, "z": 2}


@pytest.fixture(scope="module")
def doc() -> dict:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return geo.derive_landmark_v2_spec(v1, provenance="HRG-Q1 dynamics fixture")


@pytest.fixture(scope="module")
def configs(doc) -> list[dict[str, float]]:
    """Seeded configurations; trunk coordinates span their soft limits +-20 %."""
    rng = np.random.default_rng(SEED)
    soft = doc["soft_limits_deg"]
    out = []
    for _ in range(N_CONFIGS):
        q = {}
        for name in doc["coordinate_order"]:
            if name.startswith("Translation"):
                q[name] = float(rng.uniform(-0.5, 0.5))
            elif name in soft:
                lo, hi = np.radians(soft[name])
                pad = 0.2 * (hi - lo)
                q[name] = float(rng.uniform(lo - pad, hi + pad))
            else:
                q[name] = float(rng.uniform(-0.6, 0.6))
        out.append(q)
    return out


def _spec_screws(
    doc: dict, q: dict[str, float]
) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[int, str, Any, Any]]]]:
    """Body poses and, per body, the world screw of every ancestor primitive."""
    column = {name: i for i, name in enumerate(doc["coordinate_order"])}
    poses: dict[str, np.ndarray] = {"world": np.eye(4)}
    chains: dict[str, list[tuple[int, str, Any, Any]]] = {"world": []}
    pending = list(doc["joints"])
    while pending:
        progressed = False
        for joint in list(pending):
            if joint["parent"] not in poses:
                continue
            base = poses[joint["parent"]] @ np.asarray(joint["parent_to_base"], float)
            chain = list(chains[joint["parent"]])
            for primitive in joint["primitives"]:
                kind = primitive["primitive"]
                # A primitive moves along/about its own frame axis, which the
                # primitive itself leaves fixed: the axis before is the axis after.
                axis = base[:3, _AXIS[kind[1].lower()]].copy()
                chain.append(
                    (column[primitive["coordinate"]], kind[0], axis, base[:3, 3].copy())
                )
                base = base @ _primitive_transform(kind, q[primitive["coordinate"]])
            poses[joint["child"]] = base @ np.linalg.inv(
                np.asarray(joint["child_to_follower"], float)
            )
            chains[joint["child"]] = chain
            pending.remove(joint)
            progressed = True
        assert progressed, "joint tree is not rooted at world"
    return poses, chains


def _spec_mass_and_gravity(
    doc: dict, q: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    poses, chains = _spec_screws(doc, q)
    n = len(doc["coordinate_order"])
    mass = np.zeros((n, n))
    gravity = np.zeros(n)
    g_world = np.asarray(doc["gravity_m_s2"], dtype=float)
    for body in doc["bodies"]:
        if body["name"] == "world":
            continue
        for solid in body["solids"]:
            pose = poses[body["name"]] @ np.asarray(solid["placement"], dtype=float)
            rot = pose[:3, :3]
            com = rot @ np.asarray(solid["com_m"], dtype=float) + pose[:3, 3]
            inertia = rot @ np.asarray(solid["inertia_com_kg_m2"], dtype=float) @ rot.T
            jv = np.zeros((3, n))
            jw = np.zeros((3, n))
            for col, kind, axis, origin in chains[body["name"]]:
                if kind == "P":
                    jv[:, col] = axis
                else:
                    jw[:, col] = axis
                    jv[:, col] = np.cross(axis, com - origin)
            m = float(solid["mass_kg"])
            mass += m * jv.T @ jv + jw.T @ inertia @ jw
            gravity -= m * jv.T @ g_world
    return mass, gravity


def _check(name: str, doc: dict, configs, mass_and_gravity_of) -> None:
    worst_m = worst_g = scale_m = scale_g = 0.0
    for q in configs:
        ref_m, ref_g = _spec_mass_and_gravity(doc, q)
        got_m, got_g = mass_and_gravity_of(q)
        worst_m = max(worst_m, float(np.abs(np.asarray(got_m) - ref_m).max()))
        worst_g = max(worst_g, float(np.abs(np.asarray(got_g) - ref_g).max()))
        scale_m = max(scale_m, float(np.abs(ref_m).max()))
        scale_g = max(scale_g, float(np.abs(ref_g).max()))
    assert len(configs) >= 100
    assert scale_m > 0.0 and scale_g > 0.0
    assert worst_m <= REL_TOL * scale_m, f"{name} mass matrix error {worst_m:.3e}"
    assert worst_g <= REL_TOL * scale_g, f"{name} gravity error {worst_g:.3e}"


def test_reference_is_symmetric_positive_definite_and_weighs_the_body(
    doc, configs
) -> None:
    """Sanity of the spec reference itself: SPD mass matrix, gravity = m g."""
    total = sum(s["mass_kg"] for b in doc["bodies"] for s in b["solids"])
    g_world = np.asarray(doc["gravity_m_s2"], dtype=float)
    order = doc["coordinate_order"]
    translations = [order.index(f"TranslationInput{a}") for a in "XYZ"]
    for q in configs[:10]:
        mass, gravity = _spec_mass_and_gravity(doc, q)
        scale = float(np.abs(mass).max())
        np.testing.assert_allclose(mass, mass.T, rtol=0, atol=REL_TOL * scale)
        assert np.linalg.eigvalsh(mass).min() > 0.0
        # The root translations see the whole body: M_tt = m I, g_t = -m g.
        block = mass[np.ix_(translations, translations)]
        np.testing.assert_allclose(
            block, total * np.eye(3), rtol=0, atol=REL_TOL * total
        )
        np.testing.assert_allclose(
            gravity[translations],
            -total * g_world,
            rtol=0,
            atol=REL_TOL * total * float(np.linalg.norm(g_world)),
        )


@pytest.mark.requires_mujoco
def test_mujoco_mass_matrix_and_gravity_match_spec(doc, configs) -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _meta = export_full_body_mjcf(json.dumps(doc, sort_keys=True).encode("utf-8"))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    order = doc["coordinate_order"]
    qadr = [int(model.joint(n).qposadr[0]) for n in order]
    dof = [int(model.joint(n).dofadr[0]) for n in order]
    assert model.nv == len(order)
    np.testing.assert_array_equal(model.opt.gravity, doc["gravity_m_s2"])

    def mass_and_gravity(q):
        data.qpos[qadr] = [q[n] for n in order]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, full, data.qM)
        return full[np.ix_(dof, dof)], data.qfrc_bias[dof].copy()

    _check("MuJoCo", doc, configs, mass_and_gravity)


@pytest.mark.requires_pinocchio
def test_pinocchio_mass_matrix_and_gravity_match_spec(doc, configs) -> None:
    pin = pytest.importorskip("pinocchio")
    if not hasattr(pin, "Model"):
        pytest.skip("the installed 'pinocchio' is not the rigid-body dynamics library")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    model = FullBodyPinocchioModel(doc)
    order = doc["coordinate_order"]
    columns = []
    for name in order:
        unit = model.velocity({k: float(k == name) for k in order})
        columns.append(int(np.flatnonzero(unit)[0]))

    def mass_and_gravity(q):
        mass = model.mass_matrix(q)[np.ix_(columns, columns)]
        gravity = pin.computeGeneralizedGravity(
            model.model, model.data, model.configuration(q)
        )
        return mass, np.asarray(gravity, dtype=float)[columns]

    _check("Pinocchio", doc, configs, mass_and_gravity)
