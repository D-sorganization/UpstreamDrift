"""Real-pydrake regression tests for the Drake engine (#12980, #12981, #12982).

The mock-based Drake unit tests accepted calls that pydrake does not have
(``plant.MakeMultibodyForces``, ``SetDefaultPositions(context)``) and so hid
three P1 defects.  These tests drive a REAL ``MultibodyPlant`` on the double
pendulum fixture and compare against real MuJoCo and Pinocchio engines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import is_engine_available

pytestmark = [pytest.mark.unit]

# Import pinocchio at collection time: the autouse ``_protect_engine_modules``
# fixture discards engine modules after every test, and Pinocchio's C
# extension cannot be re-imported once discarded.
try:
    import pinocchio  # noqa: F401
except ImportError:  # pragma: no cover - optional engine
    pass

_URDF = (
    Path(__file__).resolve().parents[2] / "fixtures" / "models" / "double_pendulum.urdf"
)
_Q = np.array([0.5, 0.3])
_V = np.array([0.2, -0.4])
_QACC = np.array([0.7, -0.2])
_ATOL = 1e-8


def _load_drake():
    from src.engines.physics_engines.drake.python.drake_physics_engine import (
        DrakePhysicsEngine,
    )

    eng = DrakePhysicsEngine()
    eng.load_from_path(str(_URDF))
    return eng


def _load_pinocchio():
    if not is_engine_available("pinocchio"):
        pytest.skip("pinocchio not installed")
    from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
        PinocchioPhysicsEngine,
    )

    eng = PinocchioPhysicsEngine()
    eng.load_from_path(str(_URDF))
    eng.set_state(_Q, _V)
    eng.forward()
    return eng


def _load_mujoco():
    if not is_engine_available("mujoco"):
        pytest.skip("mujoco not installed")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    eng = MuJoCoPhysicsEngine()
    eng.load_from_path(str(_URDF))
    eng.set_state(_Q, _V)
    eng.forward()
    return eng


def _mujoco_reference(mj):
    """MuJoCo ID/drift/ZTCF/ZVCF built from MuJoCo's own M, bias and g.

    ``mj_forward``/``mj_inverse`` also add MuJoCo's constraint-solver forces
    (joint-limit rows are active on this fixture), which are not part of the
    rigid-body equation ``M a + C v + g = tau`` that the other engines
    expose, so the comparison uses the MuJoCo mass matrix and bias directly.
    """
    mass = mj.compute_mass_matrix()
    bias = mj.compute_bias_forces()
    grav = mj.compute_gravity_forces()
    return {
        "id": mass @ _QACC + bias,
        "drift": -np.linalg.solve(mass, bias),
        "zvcf": -np.linalg.solve(mass, grav),
    }


def _drake_at_state():
    eng = _load_drake()
    eng.set_state(_Q, _V)
    eng.forward()
    return eng


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_forward_runs_on_real_plant() -> None:
    """#12980: forward() must not raise on a real plant."""
    _drake_at_state()


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_bias_and_id_match_pinocchio() -> None:
    """#12980: bias and inverse dynamics agree with Pinocchio."""
    drake = _drake_at_state()
    pin = _load_pinocchio()
    bias = drake.compute_bias_forces()
    assert bias.shape == (2,)
    np.testing.assert_allclose(bias, pin.compute_bias_forces(), atol=_ATOL)
    np.testing.assert_allclose(bias, [8.22767, 3.52511], atol=1e-4)
    np.testing.assert_allclose(
        drake.compute_inverse_dynamics(_QACC),
        pin.compute_inverse_dynamics(_QACC),
        atol=_ATOL,
    )


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_bias_and_id_match_mujoco() -> None:
    """#12980: bias and inverse dynamics agree with MuJoCo."""
    drake = _drake_at_state()
    mj = _load_mujoco()
    np.testing.assert_allclose(
        drake.compute_bias_forces(), mj.compute_bias_forces(), atol=_ATOL
    )
    np.testing.assert_allclose(
        drake.compute_inverse_dynamics(_QACC), _mujoco_reference(mj)["id"], atol=_ATOL
    )


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_drift_and_ztcf_match_pinocchio() -> None:
    """#12980: drift acceleration and ZTCF agree with Pinocchio."""
    drake = _drake_at_state()
    pin = _load_pinocchio()
    np.testing.assert_allclose(
        drake.compute_drift_acceleration(),
        pin.compute_drift_acceleration(),
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        drake.compute_ztcf(_Q, _V), pin.compute_ztcf(_Q, _V), atol=_ATOL
    )


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_drift_and_ztcf_match_mujoco() -> None:
    """#12980: drift acceleration and ZTCF agree with MuJoCo."""
    drake = _drake_at_state()
    mj = _load_mujoco()
    ref = _mujoco_reference(mj)["drift"]
    np.testing.assert_allclose(drake.compute_drift_acceleration(), ref, atol=_ATOL)
    np.testing.assert_allclose(drake.compute_ztcf(_Q, _V), ref, atol=_ATOL)


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_reset_returns_to_initial_state() -> None:
    """#12981: reset() restores t=0 and the post-load state."""
    drake = _load_drake()
    q0, v0 = drake.get_state()
    drake.set_state(_Q, _V)
    drake.step(0.01)
    assert drake.get_time() > 0.0
    drake.reset()
    assert drake.get_time() == 0.0
    q, v = drake.get_state()
    np.testing.assert_allclose(q, q0, atol=1e-12)
    np.testing.assert_allclose(v, v0, atol=1e-12)
    np.testing.assert_allclose(q, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v, [0.0, 0.0], atol=1e-12)


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_reset_uninitialised_raises() -> None:
    """#12981: reset()/step() on an uninitialised engine raise, not log."""
    from src.engines.physics_engines.drake.python.drake_physics_engine import (
        DrakePhysicsEngine,
    )
    from src.shared.python.core.contracts import PreconditionError

    eng = DrakePhysicsEngine()
    with pytest.raises(PreconditionError):
        eng.reset()
    with pytest.raises(PreconditionError):
        eng.step(0.01)


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_gravity_sign_matches_pinocchio_and_mujoco() -> None:
    """#12982: g(q) has the same sign across engines; bias == g at v=0."""
    drake = _drake_at_state()
    pin = _load_pinocchio()
    mj = _load_mujoco()
    g = drake.compute_gravity_forces()
    np.testing.assert_allclose(g, pin.compute_gravity_forces(), atol=_ATOL)
    np.testing.assert_allclose(g, mj.compute_gravity_forces(), atol=_ATOL)
    np.testing.assert_allclose(g, [8.22767, 3.52215], atol=1e-4)

    drake.set_state(_Q, np.zeros(2))
    drake.forward()
    np.testing.assert_allclose(drake.compute_bias_forces(), g, atol=_ATOL)


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_zvcf_matches_pinocchio_and_mujoco() -> None:
    """#12982: ZVCF agrees across engines."""
    drake = _drake_at_state()
    pin = _load_pinocchio()
    mj = _load_mujoco()
    z = drake.compute_zvcf(_Q)
    np.testing.assert_allclose(z, pin.compute_zvcf(_Q), atol=_ATOL)
    np.testing.assert_allclose(z, _mujoco_reference(mj)["zvcf"], atol=_ATOL)


_SPEC_TEST_FILES = (
    "tests/unit/test_drake_physics_engine.py",
    "tests/unit/test_drake_wrapper.py",
    "tests/integration/test_phase1_drake_integration.py",
    "tests/heavy_integration/test_phase1_drake_integration.py",
)


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_mock_plant_specs_only_list_real_multibody_plant_attributes() -> None:
    """Mock-plant specs must mirror the real API (#12980, #12981)."""
    import ast

    from pydrake.multibody.plant import MultibodyPlant

    repo_root = Path(__file__).resolve().parents[3]
    real = set(dir(MultibodyPlant))
    checked = 0
    for rel in _SPEC_TEST_FILES:
        tree = ast.parse((repo_root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "_PLANT_SPEC_ATTRS"
                    for t in node.targets
                )
                and isinstance(node.value, ast.List)
            ):
                names = {ast.literal_eval(e) for e in node.value.elts}
                assert names - real == set(), f"{rel}: not on MultibodyPlant"
                checked += 1
    assert checked == len(_SPEC_TEST_FILES)


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake missing")
def test_multibody_forces_constructor_is_the_real_api() -> None:
    """#12980: ``MakeMultibodyForces`` must not exist in src/."""
    from pydrake.multibody.plant import MultibodyPlant

    assert not hasattr(MultibodyPlant, "MakeMultibodyForces")
    repo_root = Path(__file__).resolve().parents[3]
    for path in (repo_root / "src").rglob("*.py"):
        assert "MakeMultibodyForces" not in path.read_text(
            encoding="utf-8", errors="ignore"
        ), path
