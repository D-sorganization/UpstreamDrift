"""R6 (#12686): the chain shaft in MuJoCo.

Acceptance criteria covered here:

2. First bending frequency of the grip-clamped chain plus head (MuJoCo mass
   matrix, emitted joint stiffness) equals the FEM reference, per bending
   axis, within 1 %, and changes < 1 % from N to 2N.
1. Static tip deflection through the MuJoCo tip Jacobian equals the FEM.
3. The compiled chain club (subtree mass, centre of mass, inertia) equals the
   compiled rigid club to 1e-9.
4. The rigid default is unchanged; the chain coordinates are appended.
5. Rigid limit: springs x 1e6 reproduce the rigid forward dynamics.
- DbC: ``VectorPlant`` refuses a chain spec for engines without chain support.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DET
import numpy as np
import pytest

from src.shared.python.motion_matching.club_models import CLUBS, DRIVER, ClubSpec
from src.shared.python.motion_matching.club_shaft_chain import (
    DEFAULT_LINKS,
    ShaftChainUnsupportedError,
    apply_shaft_chain,
    build_shaft_chain,
    clamped_head_frequencies,
    clamped_shaft_document,
    fem_tip_deflection,
    span_properties,
)

mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SEEDS = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
SPEC_PATH = SEEDS / "full_body_spec_hipcal_scaled.json"
RECORD_PATH = SEEDS / "dynamics_record.npz"
#: sha256 of json.dumps(coordinate_order) of the evidence spec MJCF on main
#: before #12686 (the rigid default must keep it).
RIGID_ORDER_SHA256 = "25ef774a2197dded61cc19317d9b6ee634f542ccd7772f34c0e025879b199783"
CLUB_NAMES = ("driver", "iron7")


def _export(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    return export_full_body_mjcf(json.dumps(spec).encode())


@pytest.fixture(scope="module")
def rigid_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chain_spec(rigid_spec: dict[str, Any]) -> dict[str, Any]:
    return apply_shaft_chain(rigid_spec, build_shaft_chain(DRIVER))


# --- criterion 4: rigid default unchanged, chain appended --------------------


def test_rigid_export_is_unchanged_by_the_spring_emitter(
    rigid_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.engines.physics_engines.mujoco.python import native_mjcf

    xml, metadata = _export(rigid_spec)
    order = json.dumps(metadata["coordinate_order"]).encode()
    assert hashlib.sha256(order).hexdigest() == RIGID_ORDER_SHA256
    monkeypatch.setattr(native_mjcf, "_set_joint_spring", lambda *_: None)
    assert _export(rigid_spec)[0] == xml  # byte-identical without the hook
    for joint in DET.fromstring(xml).iter("joint"):
        assert joint.get("stiffness") == "0" and joint.get("damping") == "0"
        assert joint.get("springref") is None


def test_rigid_shaft_model_is_the_default_and_identity() -> None:
    from src.shared.python.motion_matching.execution.spec_builder import (
        with_shaft_model,
    )

    document = {"club": {"name": "driver"}}
    assert with_shaft_model(document, "driver", None) is document
    assert with_shaft_model(document, "driver", "rigid") is document
    with pytest.raises(ValueError, match="shaft"):
        with_shaft_model(document, "driver", "wet-noodle")


def test_chain_coordinates_are_appended_with_springs(
    rigid_spec: dict[str, Any], chain_spec: dict[str, Any]
) -> None:
    rigid_xml, rigid_meta = _export(rigid_spec)
    xml, metadata = _export(chain_spec)
    chain = chain_spec["club"]["shaft"]["coordinates"]
    rigid_order = list(rigid_meta["coordinate_order"])
    assert list(metadata["coordinate_order"]) == rigid_order + chain
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nq == model.nv == len(rigid_order) + len(chain) == 52
    springs = {
        p["coordinate"]: p
        for j in chain_spec["joints"]
        for p in j["primitives"]
        if "stiffness_n_m_rad" in p
    }
    joints = {j.get("name"): j for j in DET.fromstring(xml).iter("joint")}
    for name in chain:
        assert (
            float(joints[name].get("stiffness")) == springs[name]["stiffness_n_m_rad"]
        )
        assert float(joints[name].get("damping")) == springs[name]["damping_n_m_s_rad"]
        assert float(joints[name].get("springref")) == 0.0
    for name in rigid_order:
        assert joints[name].get("springref") is None
    assert DET.fromstring(rigid_xml).find("worldbody") is not None


# --- criterion 3: compiled inertia -------------------------------------------


def _club_subtree(xml: str, club_body: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass and inertia (world) of the club subtree at qpos0."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    club = model.body(club_body).id
    members = [
        b for b in range(model.nbody) if any(a == club for a in _ancestors(model, b))
    ]
    mass = sum(model.body_mass[b] for b in members)
    com = sum(model.body_mass[b] * data.xipos[b] for b in members) / mass
    inertia = np.zeros((3, 3))
    for b in members:
        rot = data.ximat[b].reshape(3, 3)
        d = data.xipos[b] - com
        inertia += rot @ np.diag(model.body_inertia[b]) @ rot.T
        inertia += model.body_mass[b] * (d @ d * np.eye(3) - np.outer(d, d))
    return float(mass), com, inertia


def _ancestors(model: Any, body: int) -> list[int]:
    chain = [body]
    while body != 0:
        body = int(model.body_parentid[body])
        chain.append(body)
    return chain


def test_compiled_chain_club_matches_rigid_club(
    rigid_spec: dict[str, Any], chain_spec: dict[str, Any]
) -> None:
    club_body = chain_spec["club"]["shaft"]["club_body"]
    rigid = _club_subtree(_export(rigid_spec)[0], club_body)
    chain = _club_subtree(_export(chain_spec)[0], club_body)
    assert chain[0] == pytest.approx(rigid[0], abs=1e-9)
    np.testing.assert_allclose(chain[1], rigid[1], rtol=0, atol=1e-9)
    np.testing.assert_allclose(chain[2], rigid[2], rtol=0, atol=1e-9)


# --- criteria 1 and 2: grip-clamped shaft vs FEM ------------------------------


def _clamped(club: ClubSpec, n_links: int) -> tuple[Any, Any]:
    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import (
        export_clamped_shaft_mjcf,
    )

    xml = export_clamped_shaft_mjcf(
        clamped_shaft_document(build_shaft_chain(club, n_links))
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _axis_dofs(model: Any, axis: str) -> np.ndarray:
    return np.array(
        [
            model.jnt_dofadr[j]
            for j in range(model.njnt)
            if f"_bend_{axis}_" in model.joint(j).name
        ]
    )


def chain_first_frequency(club: ClubSpec, n_links: int, axis: str) -> float:
    """First frequency (Hz) of one bending family from MuJoCo's M and K."""
    from scipy.linalg import eigh

    from src.shared.python.engine_core.mujoco_compat import full_mass_matrix

    model, data = _clamped(club, n_links)
    dofs = _axis_dofs(model, axis)
    mass = full_mass_matrix(mujoco, model, data)[np.ix_(dofs, dofs)]
    stiffness = np.diag(model.jnt_stiffness[model.dof_jntid[dofs]])
    return float(np.sqrt(eigh(stiffness, mass, eigvals_only=True)[0]) / (2 * np.pi))


def chain_tip_deflection(club: ClubSpec, n_links: int) -> float:
    """Linear static tip deflection (m) under a unit club-``z`` tip load."""
    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import TIP_SITE

    model, data = _clamped(club, n_links)
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac, None, model.site(TIP_SITE).id)
    stiffness = model.jnt_stiffness[model.dof_jntid]
    q = (jac.T @ np.array([0.0, 0.0, 1.0])) / stiffness
    return float((jac @ q)[2])


def fem_first_frequency(club: ClubSpec, axis: str) -> float:
    """FEM reference with the head's rotary inertia about the bending axis."""
    gyration = club.head_gyration_m["xyz".index(axis)]
    inertia = club.head_mass_kg * gyration**2
    return float(clamped_head_frequencies(span_properties(club), club, inertia)[0])


@pytest.mark.parametrize("name", CLUB_NAMES)
@pytest.mark.parametrize("axis", ["x", "z"])
def test_first_frequency_matches_fem_and_converges(name: str, axis: str) -> None:
    """Criterion 2."""
    club = CLUBS[name]
    reference = fem_first_frequency(club, axis)
    values = {
        n: chain_first_frequency(club, n, axis)
        for n in (DEFAULT_LINKS, 2 * DEFAULT_LINKS)
    }
    assert values[DEFAULT_LINKS] == pytest.approx(reference, rel=0.01)
    assert abs(values[2 * DEFAULT_LINKS] / values[DEFAULT_LINKS] - 1.0) < 0.01


@pytest.mark.parametrize("name", CLUB_NAMES)
def test_mujoco_static_deflection_matches_fem(name: str) -> None:
    """Criterion 1 through the emitted MJCF (joint stiffness, tip Jacobian)."""
    club = CLUBS[name]
    reference = fem_tip_deflection(span_properties(club), 1.0)
    values = {
        n: chain_tip_deflection(club, n) for n in (DEFAULT_LINKS, 2 * DEFAULT_LINKS)
    }
    assert values[DEFAULT_LINKS] == pytest.approx(reference, rel=0.01)
    assert abs(values[2 * DEFAULT_LINKS] / values[DEFAULT_LINKS] - 1.0) < 0.01


def test_emitted_springs_act_as_passive_forces() -> None:
    model, data = _clamped(DRIVER, DEFAULT_LINKS)
    rng = np.random.default_rng(12686)
    data.qpos[:] = rng.normal(scale=1e-3, size=model.nq)
    data.qvel[:] = rng.normal(scale=1e-2, size=model.nv)
    mujoco.mj_forward(model, data)
    stiffness = model.jnt_stiffness[model.dof_jntid]
    expected = -stiffness * data.qpos - model.dof_damping * data.qvel
    np.testing.assert_allclose(data.qfrc_passive, expected, rtol=1e-12, atol=1e-12)


# --- engine wiring and DbC ------------------------------------------------------


@pytest.mark.parametrize("engine", ["drake", "pinocchio"])
def test_vector_plant_refuses_chain_spec_without_chain_support(
    chain_spec: dict[str, Any], engine: str
) -> None:
    from src.shared.python.motion_matching.same_input.plant import VectorPlant

    with pytest.raises(ShaftChainUnsupportedError, match="silent rigid replay"):
        VectorPlant(engine, json.dumps(chain_spec).encode())


def test_simulator_leaves_shaft_coordinates_unactuated(
    chain_spec: dict[str, Any],
) -> None:
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        FullBodySimulator,
    )

    adapter = NativeMujocoFullBodyModel(json.dumps(chain_spec).encode())
    simulator = FullBodySimulator(adapter)
    chain = [simulator.names.index(c) for c in adapter.passive_coordinates]
    assert len(chain) == 2 * DEFAULT_LINKS
    assert not set(chain) & set(simulator.actuated.tolist())
    assert not set(chain) & set(simulator.lower_limb.tolist())


# --- criterion 5: rigid limit ---------------------------------------------------

#: Canonical driver state (record row 450: downswing, ~15 rad/s joint rates).
LIMIT_ROW = 450
LIMIT_WINDOW_S = 1e-4
#: Explicit RK4 step inside the stability region of the stiffest damped mode:
#: x 1e6 raises the shaft eigenvalues 1e3-fold (|lambda| ~ 7e6 1/s).
LIMIT_STEP_S = {1e4: 2.5e-6, 1e6: 2.5e-7}
#: Recorded tolerance (#12686): the residual compliance of the x 1e6 springs
#: (deflection ~ load / k) and the stiff oscillation excited by starting at
#: zero deflection keep the shared coordinates within 1e-6 rad and the
#: clubhead within 0.1 um of the rigid model (measured 1.7e-7 rad, 12 nm).
LIMIT_COORDINATE_TOL_RAD = 1e-6
LIMIT_CLUBHEAD_TOL_M = 1e-7


def _limit_deviation(rigid_spec: dict[str, Any], scale: float) -> tuple[float, float]:
    from src.shared.python.motion_matching.same_input.integrator import (
        StepPolicy,
        integrate,
    )
    from src.shared.python.motion_matching.same_input.plant import VectorPlant

    record = np.load(RECORD_PATH)
    q, v, tau = (record[k][LIMIT_ROW] for k in ("q", "v", "tau"))
    chain_doc = apply_shaft_chain(
        rigid_spec, build_shaft_chain(DRIVER, stiffness_scale=scale)
    )
    rigid = VectorPlant("mujoco", json.dumps(rigid_spec).encode())
    chain = VectorPlant("mujoco", json.dumps(chain_doc).encode())
    pad = np.zeros(chain.nv - rigid.nv)
    h = LIMIT_STEP_S[scale]
    run = {
        "policy": StepPolicy(substeps=1, project=False),
        "steps": round(LIMIT_WINDOW_S / h),
        "dt_s": h,
    }
    r = integrate(rigid, q, v, lambda *_: tau, **run)
    c = integrate(
        chain, np.r_[q, pad], np.r_[v, pad], lambda *_: np.r_[tau, pad], **run
    )
    shared = float(np.abs(c.q[:, : rigid.nv] - r.q).max())
    head_r = rigid.kinematic_frames(r.q[-1])["Clubhead"][:3, 3]
    head_c = chain.kinematic_frames(c.q[-1])["Clubhead"][:3, 3]
    return shared, float(np.linalg.norm(head_c - head_r))


def test_stiff_chain_reproduces_rigid_forward_dynamics(
    rigid_spec: dict[str, Any],
) -> None:
    """Criterion 5: x 1e6 tracks the rigid model, and converges like 1/k."""
    shared, clubhead = _limit_deviation(rigid_spec, 1e6)
    assert shared < LIMIT_COORDINATE_TOL_RAD
    assert clubhead < LIMIT_CLUBHEAD_TOL_M
    softer, _ = _limit_deviation(rigid_spec, 1e4)
    assert softer > 20.0 * shared  # residual compliance shrinks with k
