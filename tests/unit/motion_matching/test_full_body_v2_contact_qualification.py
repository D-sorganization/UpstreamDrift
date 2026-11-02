"""Cross-engine foot-contact qualification of full-body-v2 (HRG-Q1, #13176).

The reference is computed from the specification alone. Each foot sphere's
centre is its body pose (``_spec_screws``, shared with the dynamics module)
applied to ``position_m``, and its velocity is exact, ``J v``, from the world
screw of every ancestor primitive. The spec law is restated here from the
spec's ``contact`` block, independently of ``contact_kernel``: with the ground
normal ``n`` opposite gravity at height 0,

    delta = max(0, r - n.c),   v_n = n.v,   F_n = max(0, k delta (1 - d v_n)) n
    mu(s) = mu_d + (mu_s - mu_d) exp(-(s/v_tr)^2) + mu_v s
    F_t = -(v_t / s) mu(s) |F_n| tanh(s / v_tr)        (slip s = |v_t| > 0)

Configurations are seeded and the pelvis is lowered until the lowest sphere
penetrates 1-10 mm, with seeded rates so that dissipation and friction act.
Each engine's ``VectorPlant.contact_samples`` must reproduce the penetration
(``PEN_TOL``) and the normal and friction forces (``REL_TOL`` of the largest
reference normal force) of every sphere, the separated ones included. The
worst error measured over 16 configurations (19 penetrating spheres, some
lifted off by damping; normal forces up to 803 N) is 5.4e-16 m and 4.6e-11 N (relative 5.7e-14) on MuJoCo,
Pinocchio, Drake and OpenSim; the tolerances keep the HRG-Q1 precision
budget and are never loosened.

MyoSuite is reached by segment-orientation IK on the thorax, not by exporting
the spec (``EXPORT_STATUS["myosuite"]``), so it carries no foot spheres.
Inverse dynamics on a reference trajectory remains open on #13176.
"""

from __future__ import annotations

import json
import types
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
N_CONFIGS = 16
SEED = 1317604
PEN_TOL = 1e-12
REL_TOL = 1e-12
Sample = tuple[float, np.ndarray, np.ndarray]


@pytest.fixture(scope="module")
def doc() -> dict:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return geo.derive_landmark_v2_spec(v1, provenance="HRG-Q1 contact fixture")


def _normal(doc: dict) -> np.ndarray:
    gravity = np.asarray(doc["gravity_m_s2"], dtype=float)
    return -gravity / np.linalg.norm(gravity)


def _sphere_states(
    doc: dict, q: np.ndarray, v: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """Spec-only centre, exact velocity and radius of every foot sphere."""
    order = doc["coordinate_order"]
    poses, chains = _spec_screws(doc, dict(zip(order, q, strict=True)))
    states = {}
    for sphere in doc["contact"]["spheres"]:
        centre = (poses[sphere["body"]] @ np.r_[sphere["position_m"], 1.0])[:3]
        jacobian = np.zeros((3, len(order)))
        for col, kind, axis, origin in chains[sphere["body"]]:
            jacobian[:, col] = axis if kind == "P" else np.cross(axis, centre - origin)
        states[sphere["name"]] = (centre, jacobian @ v, float(sphere["radius_m"]))
    return states


def _spec_law(
    doc: dict, centre: np.ndarray, velocity: np.ndarray, radius: float
) -> Sample:
    """Penetration, normal force and friction force of the spec contact law."""
    p, normal = doc["contact"]["parameters"], _normal(doc)
    delta = max(0.0, radius - float(normal @ centre))
    v_n = float(normal @ velocity)
    f_n = max(0.0, p["stiffness_n_m"] * delta * (1.0 - p["dissipation_s_m"] * v_n))
    slip = velocity - v_n * normal
    speed = float(np.linalg.norm(slip))
    friction = np.zeros(3)
    if speed > 0.0 and f_n > 0.0:
        ratio = speed / p["transition_velocity_m_s"]
        mu = (
            p["dynamic_friction"]
            + (p["static_friction"] - p["dynamic_friction"]) * np.exp(-(ratio**2))
            + p["viscous_friction"] * speed
        )
        friction = -slip / speed * mu * f_n * np.tanh(ratio)
    return delta, f_n * normal, friction


@pytest.fixture(scope="module")
def cases(doc) -> list[tuple[np.ndarray, np.ndarray]]:
    """Seeded (q, v); the pelvis is lowered so a foot sphere penetrates."""
    order, normal = doc["coordinate_order"], _normal(doc)
    lift = order.index("TranslationInputZ")
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_CONFIGS):
        q = rng.uniform(-0.3, 0.3, len(order))
        q[lift] = 0.0
        states = _sphere_states(doc, q, np.zeros(len(order)))
        clearance = min(float(normal @ c) - r for c, _, r in states.values())
        q[lift] = -clearance - rng.uniform(0.001, 0.01)
        out.append((q, rng.normal(0.0, 0.3, len(order))))
    return out


@pytest.fixture(scope="module")
def reference(doc, cases) -> list[dict[str, Sample]]:
    return [
        {
            name: _spec_law(doc, *state)
            for name, state in _sphere_states(doc, q, v).items()
        }
        for q, v in cases
    ]


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


def test_reference_exercises_contact_dissipation_and_friction(reference) -> None:
    """Spheres press, separate and lift off; slip friction acts on loaded ones."""
    samples = [s for case in reference for s in case.values()]
    pressing = [s for s in samples if s[0] > 0.0]
    loaded = [s for s in pressing if np.linalg.norm(s[1]) > 0.0]
    assert len(samples) == N_CONFIGS * 6
    assert 0 < len(pressing) < len(samples)
    # Damping liftoff (1 - d v_n <= 0) clips some penetrating spheres to zero.
    assert 0 < len(loaded) < len(pressing)
    assert all(np.linalg.norm(f_t) > 0.0 for _, _, f_t in loaded)
    assert max(np.linalg.norm(f_n) for _, f_n, _ in loaded) > 100.0


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("mujoco", marks=pytest.mark.requires_mujoco),
        pytest.param("pinocchio", marks=pytest.mark.requires_pinocchio),
        pytest.param("drake", marks=pytest.mark.requires_drake),
        pytest.param("opensim", marks=pytest.mark.requires_opensim),
    ],
)
def test_contact_normal_force_matches_spec_law(engine, doc, cases, reference) -> None:
    """Each engine's foot contact reproduces the spec law sphere by sphere."""
    _real_engine(engine)
    from src.shared.python.motion_matching.same_input import VectorPlant

    plant = VectorPlant(engine, json.dumps(doc, sort_keys=True).encode("utf-8"))
    force_tol = REL_TOL * max(
        np.linalg.norm(f_n) for case in reference for _, f_n, _ in case.values()
    )
    for (q, v), expected in zip(cases, reference, strict=True):
        samples = plant.contact_samples(q, v)
        assert set(samples) == set(expected)
        for name, (delta, f_n, f_t) in expected.items():
            got = samples[name]
            assert abs(max(got.penetration_m, 0.0) - delta) <= PEN_TOL, name
            normal_error = np.abs(np.asarray(got.normal_force_n) - f_n).max()
            friction_error = np.abs(np.asarray(got.friction_force_n) - f_t).max()
            assert normal_error <= force_tol, f"{engine} {name}: F_n {normal_error:.3g}"
            assert friction_error <= force_tol, (
                f"{engine} {name}: F_t {friction_error:.3g}"
            )


def test_myosuite_has_no_foot_spheres_to_qualify() -> None:
    """MyoSuite is IK-mapped on the thorax, so its contact cell is exempt."""
    status = geo.EXPORT_STATUS["myosuite"]
    assert status["status"] == "mapped_thorax_only"
    assert "segment-orientation IK" in status["reason"]
