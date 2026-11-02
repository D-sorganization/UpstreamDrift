"""Cross-engine inverse-dynamics qualification of full-body-v2 (HRG-Q1, #13176).

Every engine is driven along the same seeded, smooth reference trajectory
q(t) (one sinusoid per coordinate, with analytic q' and q''), and its native
inverse dynamics must return the reference generalized forces.

The reference is computed from the specification alone, by a recursive
Newton-Euler pass over the spec joints. Each body's world twist ``V`` and
spatial acceleration ``A`` (both at the world origin) accumulate the world
screw ``xi`` of every ancestor primitive (``_spec_screws``, shared with the
dynamics module):

    V += xi q',    A += xi q'' + (V x xi) q'      (V updated after A)

Each solid then contributes, through the power of its wrench on every
ancestor screw, Newton's and Euler's equations at its centre of mass ``c``:

    F = m (a_c - g),   M_c = I alpha + w x I w,
    tau_j += xi_j . [M_c + c x F; F]

The exact-ID calls are MuJoCo ``mj_inverse`` on the exported MJCF with
constraints disabled, Pinocchio ``rnea``, and the Drake and OpenSim
``mass_and_bias`` (``M q'' + h``); all are unconstrained and contact-free,
like the reference. The worst error measured over 24 samples (generalized
forces up to 7.3 kN) is 4.5e-12 (relative 6.2e-16) on MuJoCo, Pinocchio, Drake
and OpenSim, so ``REL_TOL`` (1e-12 of the largest reference force) keeps the
HRG-Q1 precision budget and is never loosened.

MyoSuite is reached by segment-orientation IK on the thorax, not by exporting
the spec (``EXPORT_STATUS["myosuite"]``), so it has no full-body dynamics
driven by the same q(t).
"""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import full_body_v2_geometry as geo
from tests.helpers.opensim_support import require_opensim
from tests.unit.motion_matching.test_full_body_v2_dynamics_qualification import (
    _spec_screws,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
N_SAMPLES = 24
SEED = 1317605
REL_TOL = 1e-12
State = tuple[np.ndarray, np.ndarray, np.ndarray]
InverseDynamics = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@pytest.fixture(scope="module")
def doc() -> dict:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return geo.derive_landmark_v2_spec(v1, provenance="HRG-Q1 inverse dynamics fixture")


@pytest.fixture(scope="module")
def trajectory(doc) -> list[State]:
    """Seeded q(t) = A sin(2 pi f t + phi) with analytic q' and q''."""
    n = len(doc["coordinate_order"])
    rng = np.random.default_rng(SEED)
    amplitude = rng.uniform(0.2, 0.6, n)
    omega = 2.0 * np.pi * rng.uniform(0.5, 2.0, n)
    phase = rng.uniform(0.0, 2.0 * np.pi, n)
    out = []
    for t in np.linspace(0.0, 1.0, N_SAMPLES):
        angle = omega * t + phase
        out.append(
            (
                amplitude * np.sin(angle),
                amplitude * omega * np.cos(angle),
                -amplitude * omega**2 * np.sin(angle),
            )
        )
    return out


def _cross_motion(twist: np.ndarray, screw: np.ndarray) -> np.ndarray:
    """Spatial cross product ``twist x screw`` of ``[w; v]`` motion vectors."""
    w, v = twist[:3], twist[3:]
    return np.r_[
        np.cross(w, screw[:3]), np.cross(w, screw[3:]) + np.cross(v, screw[:3])
    ]


def _spec_inverse_dynamics(doc: dict) -> InverseDynamics:
    """Spec-only recursive Newton-Euler generalized forces in spec order."""
    order = doc["coordinate_order"]
    gravity = np.asarray(doc["gravity_m_s2"], dtype=float)

    def inverse(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        poses, chains = _spec_screws(doc, dict(zip(order, q, strict=True)))
        tau = np.zeros(len(order))
        for body in doc["bodies"]:
            if body["name"] not in chains:
                continue
            twist, accel, screws = np.zeros(6), np.zeros(6), []
            for col, kind, axis, origin in chains[body["name"]]:
                xi = (
                    np.r_[np.zeros(3), axis]
                    if kind == "P"
                    else (np.r_[axis, np.cross(origin, axis)])
                )
                accel += xi * qdd[col] + _cross_motion(twist, xi) * qd[col]
                twist += xi * qd[col]
                screws.append((col, xi))
            w, alpha = twist[:3], accel[:3]
            for solid in body["solids"]:
                pose = poses[body["name"]] @ np.asarray(solid["placement"], float)
                rot = pose[:3, :3]
                c = rot @ np.asarray(solid["com_m"], float) + pose[:3, 3]
                inertia = rot @ np.asarray(solid["inertia_com_kg_m2"], float) @ rot.T
                v_c = twist[3:] + np.cross(w, c)
                a_c = accel[3:] + np.cross(alpha, c) + np.cross(w, v_c)
                force = float(solid["mass_kg"]) * (a_c - gravity)
                moment = inertia @ alpha + np.cross(w, inertia @ w) + np.cross(c, force)
                for col, xi in screws:
                    tau[col] += xi[:3] @ moment + xi[3:] @ force
        return tau

    return inverse


@pytest.fixture(scope="module")
def reference(doc, trajectory) -> list[np.ndarray]:
    inverse = _spec_inverse_dynamics(doc)
    return [inverse(*state) for state in trajectory]


def _real_engine(engine: str) -> None:
    """Skip when the engine is missing or a CI ``MagicMock`` stub."""
    if engine == "opensim":
        module, attr = require_opensim(), "Model"
    else:
        name, attr = {
            "mujoco": ("mujoco", "MjModel"),
            "pinocchio": ("pinocchio", "Model"),
            "drake": ("pydrake.all", "MultibodyPlant"),
        }[engine]
        module = pytest.importorskip(name)
    if not (
        isinstance(module, types.ModuleType)
        and isinstance(getattr(module, attr, None), type)
    ):
        pytest.skip(f"{engine} is a stub or not the physics library")


def _mujoco_inverse(doc: dict, spec_bytes: bytes) -> InverseDynamics:
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    model = mujoco.MjModel.from_xml_string(export_full_body_mjcf(spec_bytes)[0])
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    data = mujoco.MjData(model)
    joints = [model.joint(name) for name in doc["coordinate_order"]]
    qpos = np.array([int(j.qposadr[0]) for j in joints])
    dof = np.array([int(j.dofadr[0]) for j in joints])

    def inverse(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        data.qpos[qpos], data.qvel[dof], data.qacc[dof] = q, qd, qdd
        mujoco.mj_inverse(model, data)
        return data.qfrc_inverse[dof].copy()

    return inverse


def _pinocchio_inverse(doc: dict, spec_bytes: bytes) -> InverseDynamics:
    import pinocchio

    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    native = FullBodyPinocchioModel(doc)
    order = doc["coordinate_order"]
    columns = [
        int(np.flatnonzero(native.velocity({k: float(k == name) for k in order}))[0])
        for name in order
    ]

    def inverse(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        state = [np.zeros(native.model.nv) for _ in range(3)]
        for vector, values in zip(state, (q, qd, qdd), strict=True):
            vector[columns] = values
        tau = pinocchio.rnea(native.model, native.data, *state)
        return np.asarray(tau, dtype=float)[columns]

    return inverse


def _mass_and_bias_inverse(
    engine: str, doc: dict, spec_bytes: bytes
) -> InverseDynamics:
    if engine == "drake":
        from src.engines.physics_engines.drake.python.full_body_model import (
            FullBodyDrakeModel,
        )

        adapter = FullBodyDrakeModel(doc)
    else:
        from src.engines.physics_engines.opensim.python.full_body_parity import (
            build_parity_adapter,
        )

        adapter = build_parity_adapter(spec_bytes)

    def inverse(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        mass, bias = adapter.mass_and_bias(q, qd)
        return mass @ qdd + bias

    return inverse


def _engine_inverse(engine: str, doc: dict) -> InverseDynamics:
    spec_bytes = json.dumps(doc, sort_keys=True).encode("utf-8")
    if engine == "mujoco":
        return _mujoco_inverse(doc, spec_bytes)
    if engine == "pinocchio":
        return _pinocchio_inverse(doc, spec_bytes)
    return _mass_and_bias_inverse(engine, doc, spec_bytes)


def test_reference_trajectory_loads_every_coordinate(reference) -> None:
    """Finite, non-trivial forces on every coordinate: the parity is not vacuous."""
    forces = np.stack(reference)
    assert forces.shape[0] == N_SAMPLES
    assert np.all(np.isfinite(forces))
    assert np.all(np.abs(forces).max(axis=0) > 0.0)
    assert np.abs(forces).max() > 100.0


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("mujoco", marks=pytest.mark.requires_mujoco),
        pytest.param("pinocchio", marks=pytest.mark.requires_pinocchio),
        pytest.param("drake", marks=pytest.mark.requires_drake),
        pytest.param("opensim", marks=pytest.mark.requires_opensim),
    ],
)
def test_inverse_dynamics_matches_reference(engine, doc, trajectory, reference) -> None:
    """Each engine's inverse dynamics equals the spec's along q(t)."""
    _real_engine(engine)
    inverse = _engine_inverse(engine, doc)
    tol = REL_TOL * max(np.abs(tau).max() for tau in reference)
    for k, (state, expected) in enumerate(zip(trajectory, reference, strict=True)):
        error = np.abs(inverse(*state) - expected)
        worst = int(error.argmax())
        assert error[worst] <= tol, (
            f"{engine} sample {k}: {doc['coordinate_order'][worst]} "
            f"differs by {error[worst]:.3g} (tol {tol:.3g})"
        )


def test_myosuite_has_no_full_body_dynamics_on_the_same_q() -> None:
    """MyoSuite is IK-mapped on the thorax, so its inverse-dynamics cell is exempt."""
    status = geo.EXPORT_STATUS["myosuite"]
    assert status["status"] == "mapped_thorax_only"
    assert "segment-orientation IK" in status["reason"]
