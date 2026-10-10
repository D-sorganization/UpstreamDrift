"""MuJoCo and Drake full-body plants must agree on mass and centre of mass.

Both plants are built from the same spec document, so at identical
coordinates the total mass, the whole-body CoM and every spec body's CoM must
coincide (issue #12039).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pydrake")

from src.shared.python.motion_matching.pipeline.constants import (  # noqa: E402
    LEG_SEEDS,
    REPO_ROOT,
    square_forefoot_seeds,
)
from src.shared.python.motion_matching.pipeline.lane import (  # noqa: E402
    document_seed,
)
from src.shared.python.motion_matching.pipeline.plant import get_plant  # noqa: E402

pytestmark = pytest.mark.unit

SPEC_PATH = (
    REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
TOL_M = 1e-3


@pytest.fixture(scope="module")
def plants():
    spec = json.loads(SPEC_PATH.read_text())
    seeds = square_forefoot_seeds(LEG_SEEDS)
    mj = get_plant("mujoco", spec).create_ik(seeds)
    dk = get_plant("drake", spec).create_ik(seeds)
    return spec, mj, dk


def _poses(spec, mj):
    seed = document_seed(spec, mj)
    rng = np.random.default_rng(12039)
    out = [("seed", seed)]
    for i in range(5):
        noise = np.r_[np.zeros(6), rng.normal(0.0, 0.3, len(seed) - 6)]
        out.append((f"random{i}", seed + noise))
    return out


def _mj_com_and_bodies(mj, q):
    import mujoco

    mj._set(q)
    mujoco.mj_comPos(mj.model, mj.data)
    bodies = {
        mujoco.mj_id2name(mj.model, mujoco.mjtObj.mjOBJ_BODY, i): np.array(
            mj.data.xipos[i]
        )
        for i in range(mj.model.nbody)
        if mj.model.body_mass[i] > 0
    }
    return np.array(mj.data.subtree_com[0]), float(mj.model.body_mass.sum()), bodies


def _dk_com_and_bodies(spec, dk, q):
    dk._set(q)
    plant, ctx = dk._plant, dk._context
    links = dk.model.metadata["solid_links"]
    sums: dict[str, list] = {}
    for body in spec["bodies"]:
        for solid in body.get("solids", []):
            link = plant.GetBodyByName(links[solid["name"]], dk.model._instance)
            pose = plant.EvalBodyPoseInWorld(ctx, link)
            com_w = (
                pose.rotation().matrix() @ link.CalcCenterOfMassInBodyFrame(ctx)
                + pose.translation()
            )
            mass = float(link.default_mass())
            acc = sums.setdefault(body["name"], [0.0, np.zeros(3)])
            acc[0] += mass
            acc[1] += mass * com_w
    bodies = {n: m_c[1] / m_c[0] for n, m_c in sums.items() if m_c[0] > 0}
    com = np.asarray(plant.CalcCenterOfMassPositionInWorld(ctx), dtype=float)
    return com, float(plant.CalcTotalMass(ctx)), bodies


def test_total_mass_matches(plants):
    spec, mj, dk = plants
    _, mass_mj, _ = _mj_com_and_bodies(mj, document_seed(spec, mj))
    _, mass_dk, _ = _dk_com_and_bodies(spec, dk, document_seed(spec, mj))
    assert abs(mass_mj - mass_dk) < 1e-9


def test_whole_body_com_matches(plants):
    spec, mj, dk = plants
    for label, q in _poses(spec, mj):
        com_mj, _, _ = _mj_com_and_bodies(mj, q)
        com_dk, _, _ = _dk_com_and_bodies(spec, dk, q)
        err = float(np.linalg.norm(com_mj - com_dk))
        assert err < TOL_M, f"{label}: CoM differs by {err * 1000:.2f} mm"


def test_drake_marker_jacobian_columns_follow_spec_order(plants):
    """Jacobian column i must be d(markers)/d(q[i]) for the spec coordinate i."""
    spec, mj, dk = plants
    q = _poses(spec, mj)[1][1]
    jac = dk._marker_jacobian(dk.marker_positions(q))
    eps = 1e-6
    for i in (6, 12, 20, 27, 35):
        dq = np.zeros_like(q)
        dq[i] = eps
        fd = (dk.marker_positions(q + dq) - dk.marker_positions(q - dq)) / (2 * eps)
        assert np.abs(jac[:, :, i] - fd).max() < 1e-6, dk.coordinate_order[i]


def test_per_body_com_matches(plants):
    spec, mj, dk = plants
    for label, q in _poses(spec, mj):
        _, _, b_mj = _mj_com_and_bodies(mj, q)
        _, _, b_dk = _dk_com_and_bodies(spec, dk, q)
        joint_child = {j["child"] for j in spec["joints"]}
        bad = []
        for name, com_dk in b_dk.items():
            if name not in joint_child or name not in b_mj:
                continue
            err = float(np.linalg.norm(b_mj[name] - com_dk))
            if err >= TOL_M:
                bad.append(f"{name}: {err * 1000:.1f} mm")
        assert not bad, f"{label}: {bad}"


def test_drake_model_array_inputs_are_spec_ordered(plants):
    """Array coordinates mean ``coordinate_order``, exactly like a name mapping."""
    spec, mj, dk = plants
    model = dk.model
    q = _poses(spec, mj)[2][1]
    named = dict(zip(model.names, q, strict=True))
    attachments = {"M": ("Head", (0.0, 0.0, 0.1)), "N": ("RS", (0.05, 0.0, 0.0))}
    np.testing.assert_allclose(
        model.marker_positions(q, attachments),
        model.marker_positions(named, attachments),
        atol=1e-12,
    )
    rates = np.linspace(-0.5, 0.5, len(q))
    a_arr, b_arr = model.affine_dynamics(q, rates)
    a_map, b_map = model.affine_dynamics(
        named, dict(zip(model.names, rates, strict=True))
    )
    np.testing.assert_allclose(a_arr, a_map, atol=1e-9)
    np.testing.assert_allclose(b_arr, b_map, atol=1e-9)
