"""NSX-02 (d): canonical closure Jacobians agree across engines (#12551).

Each engine's native weld Jacobian is read from its own code path, converted with
its declared ``NativeClosureConvention`` and compared on closure-consistent
states of the anthropometric driver spec.  An absolute anchor, independent of
every engine Jacobian, is the finite difference of MuJoCo site poses, which fixes
the canonical sign (``V_a - V_b``) and frame.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.loop_nullspace.closure_convention import (
    NativeClosureConvention,
    to_canonical_closure,
)
from src.shared.python.motion_matching.same_input import (
    VectorPlant,
    project_to_closure,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_pinocchio,
    pytest.mark.requires_mujoco,
    pytest.mark.requires_drake,
    pytest.mark.requires_opensim,
]

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
ENGINES = ("pinocchio", "drake", "mujoco", "opensim")
# Absolute tolerance on the unscaled entries (linear rows in m, angular rows
# unitless).  C-SCALE-FORCE-MOMENT (F_c = 100 N, M_c = 5 N m) would only divide
# the entries and so loosen this; the unscaled comparison is the stricter one.
TOL = 1e-6


@pytest.fixture(scope="module")
def plants() -> dict[str, VectorPlant]:
    for module in ("pinocchio", "mujoco", "pydrake", "opensim"):
        pytest.importorskip(module)
    spec = SPEC.read_bytes()
    return {engine: VectorPlant(engine, spec) for engine in ENGINES}


_CONVENTION_MODULES = {
    "pinocchio": "src.engines.physics_engines.pinocchio.python.native_model",
    "drake": "src.engines.physics_engines.drake.python.full_body_model",
    "mujoco": "src.engines.physics_engines.mujoco.python.native_model",
    "opensim": "src.engines.physics_engines.opensim.python.full_body_parity",
}


@pytest.fixture(scope="module")
def conventions(plants) -> dict[str, NativeClosureConvention]:
    """Each engine's declared constant, imported once (engine modules are isolated
    between tests by the suite's sys.modules fixture)."""
    return {
        engine: importlib.import_module(module).CLOSURE_CONVENTION
        for engine, module in _CONVENTION_MODULES.items()
    }


@pytest.fixture(scope="module", params=[0, 1, 2])
def state(request, plants) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234 + request.param)
    nv = plants["pinocchio"].nv
    projected = project_to_closure(
        plants["pinocchio"],
        0.05 * rng.standard_normal(nv),
        0.2 * rng.standard_normal(nv),
    )
    return projected.q, projected.v


def _native(plant: VectorPlant, q: np.ndarray) -> dict[str, Any]:
    """Native ``J`` (spec column order), its convention inputs, via engine code."""
    adapter, named = plant._adapter, plant._named(q)
    if plant.engine == "pinocchio":
        jac = adapter.closure_force_jacobian(named).jacobian
        rot, origin = adapter.closure_frame_pose(named)
        return {"J": jac, "R": rot, "p": origin}
    if plant.engine == "drake":
        adapter.plant.SetPositions(
            adapter.context, adapter._vector(named, adapter._q_indices)
        )
        frame_a, _, jac = adapter._closure_jacobian()
        pose = adapter.plant.CalcRelativeTransform(
            adapter.context, adapter.plant.world_frame(), frame_a
        )
        return {
            "J": jac[:, adapter._v_indices],
            "R": pose.rotation().matrix(),
            "p": pose.translation(),
        }
    if plant.engine == "mujoco":
        zero = plant._named(np.zeros(plant.nv))
        adapter.accelerations(named, zero, zero)
        jac, _ = adapter.evaluate_weld_closure()
        columns = [adapter._indices[name] for name in plant.coordinate_order]
        return {
            "J": jac[:, columns],
            "R": np.eye(3),
            "p": np.array(adapter.data.site_xpos[adapter._closure[0]]),
        }
    adapter._set_state(plant._ordered(named), np.zeros(plant.nv))
    _, jac, _ = adapter._closure_blocks()
    body, placement = adapter._closure[0]
    pose = adapter._pose(body, adapter._state, placement)
    return {"J": jac, "R": np.eye(3), "p": pose[:3, 3]}


def _grip_frame(plants, q) -> tuple[np.ndarray, np.ndarray]:
    """Arbitrary canonical grip frame: Pinocchio hand frame, rotated and shifted."""
    reference = _native(plants["pinocchio"], q)
    tilt = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    return reference["R"] @ tilt, reference["p"] + np.array([0.05, 0.02, -0.03])


def _canonical(plants, conventions, q) -> dict[str, np.ndarray]:
    r_g, p_g = _grip_frame(plants, q)
    out = {}
    for engine in ENGINES:
        native = _native(plants[engine], q)
        out[engine], _ = to_canonical_closure(
            native["J"],
            np.zeros(6),
            native=conventions[engine],
            native_frame_rotation=native["R"],
            grip_frame_rotation=r_g,
            native_reference_point=native["p"],
            grip_reference_point=p_g,
        )
    return out


def test_declared_conventions_match_the_golden_table(conventions) -> None:
    golden = json.loads(
        (
            SPEC.parents[3] / "tests/fixtures/loop_nullspace/goldens/"
            "closure_convention.json"
        ).read_text()
    )
    for engine, expected in golden["natives"].items():
        conv, cfg = conventions[engine], expected["native_convention"]
        assert (conv.row_order, conv.frame, conv.point, conv.relative_sign) == (
            cfg["row_order"],
            cfg["frame"],
            cfg["point"],
            cfg["relative_sign"],
        ), engine


def test_canonical_jacobian_agrees_across_engines(plants, conventions, state) -> None:
    q, _ = state
    canonical = _canonical(plants, conventions, q)
    reference = canonical["pinocchio"]
    for engine in ENGINES[1:]:
        diff = np.abs(canonical[engine] - reference)
        assert diff.max() < TOL, f"{engine} vs pinocchio: max |dJ_c| = {diff.max():.3e}"
    assert np.abs(reference).max() > 1e-3  # not vacuous


def test_canonical_sign_and_frame_match_finite_difference_of_poses(
    plants, conventions, state
) -> None:
    """Independent anchor: d/dt of (site_a - site_b, rotvec(R_a R_b^T)) is J_c v."""
    q, _ = state
    v = np.random.default_rng(99).standard_normal(q.size)  # off the tangent space
    plant = plants["mujoco"]
    adapter = plant._adapter
    mj = adapter._mj
    a, b = adapter._closure

    def relative_pose(qq: np.ndarray) -> tuple[np.ndarray, Rotation]:
        adapter.data.qpos[:] = adapter._vector(plant._named(qq))
        mj.mj_kinematics(adapter.model, adapter.data)
        ra = Rotation.from_matrix(adapter.data.site_xmat[a].reshape(3, 3))
        rb = Rotation.from_matrix(adapter.data.site_xmat[b].reshape(3, 3))
        return adapter.data.site_xpos[a] - adapter.data.site_xpos[b], ra * rb.inv()

    eps = 1e-6
    plus, minus = relative_pose(q + eps * v), relative_pose(q - eps * v)
    linear = (plus[0] - minus[0]) / (2 * eps)
    angular = (plus[1] * minus[1].inv()).as_rotvec() / (2 * eps)
    r_g, p_g = _grip_frame(plants, q)
    # Transport the relative linear velocity from the site to the grip point.
    site = _native(plants["mujoco"], q)["p"]
    linear = linear + np.cross(angular, p_g - site)
    expected = np.concatenate((r_g.T @ linear, r_g.T @ angular))
    canonical = _canonical(plants, conventions, q)
    for engine in ENGINES:
        np.testing.assert_allclose(
            canonical[engine] @ v, expected, atol=2e-6, err_msg=engine
        )
