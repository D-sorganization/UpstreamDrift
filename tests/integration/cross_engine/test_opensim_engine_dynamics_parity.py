"""OpenSim engine dynamics on OpenSim 4.x match MuJoCo and Pinocchio (#12983).

On OpenSim 4.6 ``OpenSimPhysicsEngine`` built ``opensim.Vector(n)`` (no such
constructor) and called ``Manager.setSessionTime`` (no such method): inverse
dynamics, bias, gravity and drift came back as empty arrays and ``reset``
raised, while the capabilities still claimed ``inverse_dynamics=FULL``.

These tests run the real engines on the cross-engine double pendulum
(``tests/fixtures/models/double_pendulum.urdf``). The OpenSim model is the same
pendulum built through the OpenSim API; MuJoCo and Pinocchio load the URDF.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from tests.helpers.opensim_support import require_opensim

osim = require_opensim()
if not hasattr(osim, "Model"):
    pytest.skip("real opensim is unavailable", allow_module_level=True)
mujoco = pytest.importorskip("mujoco")
pinocchio = pytest.importorskip("pinocchio")

from src.engines.physics_engines.opensim.python import (  # noqa: E402
    opensim_physics_engine as engine_module,
)
from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
    OpenSimPhysicsEngine,
)
from src.shared.python.engine_core.capabilities import CapabilityLevel  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.headless_safe,
    pytest.mark.requires_opensim,
]

URDF = Path(__file__).resolve().parents[2] / "fixtures/models/double_pendulum.urdf"
GRAVITY = 9.81
JOINTS = ("upper_joint", "lower_joint")
#: Same-model parity: both sides are exact recursive algorithms in float64.
ATOL = 1e-10

STATES = [
    (np.array([0.0, 0.0]), np.array([0.0, 0.0])),
    (np.array([0.3, -0.7]), np.array([1.2, -0.4])),
    (np.array([1.1, 2.0]), np.array([-2.5, 3.0])),
]


@pytest.fixture(autouse=True)
def _real_opensim_in_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "opensim", osim)


def _link(name: str, mass: float, com_z: float, inertia: float) -> object:
    return osim.Body(
        name,
        mass,
        osim.Vec3(0.0, 0.0, com_z),
        osim.Inertia(inertia, inertia, inertia),
    )


def _double_pendulum_osim(path: Path) -> Path:
    """Write the URDF double pendulum as an OpenSim model (Z up, axes along +y).

    A PinJoint turns about the z axis of its frames, so both frames are rotated
    by -pi/2 about x, which carries that axis onto +y.
    """
    model = osim.Model()
    model.setName("double_pendulum")
    model.setGravity(osim.Vec3(0.0, 0.0, -GRAVITY))
    to_y = osim.Vec3(-math.pi / 2, 0.0, 0.0)
    upper = _link("upper_link", 1.0, -0.5, 0.001)
    lower = _link("lower_link", 1.0, -0.5, 0.001)
    tip = _link("end_effector", 0.001, 0.0, 1e-9)
    for body in (upper, lower, tip):
        model.addBody(body)
    model.addJoint(
        osim.PinJoint(
            "upper_joint",
            model.getGround(),
            osim.Vec3(0.0, 0.0, 2.0),
            to_y,
            upper,
            osim.Vec3(0.0, 0.0, 0.0),
            to_y,
        )
    )
    model.addJoint(
        osim.PinJoint(
            "lower_joint",
            upper,
            osim.Vec3(0.0, 0.0, -0.5),
            to_y,
            lower,
            osim.Vec3(0.0, 0.0, 0.0),
            to_y,
        )
    )
    model.addJoint(
        osim.WeldJoint(
            "lower_to_end",
            lower,
            osim.Vec3(0.0, 0.0, -0.5),
            osim.Vec3(0.0, 0.0, 0.0),
            tip,
            osim.Vec3(0.0, 0.0, 0.0),
            osim.Vec3(0.0, 0.0, 0.0),
        )
    )
    for joint in JOINTS:
        coordinate = model.updJointSet().get(joint).upd_coordinates(0)
        coordinate.setName(joint)
        coordinate.setRangeMin(-10.0)
        coordinate.setRangeMax(10.0)
    model.finalizeConnections()
    model.printToXML(str(path))
    return path


@pytest.fixture
def engine(tmp_path: Path) -> OpenSimPhysicsEngine:
    loaded = OpenSimPhysicsEngine()
    loaded.load_from_path(str(_double_pendulum_osim(tmp_path / "dp.osim")))
    return loaded


class _Reference:
    """MuJoCo and Pinocchio dynamics of the URDF at one state."""

    def __init__(self) -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(str(URDF))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.pin_model = pinocchio.buildModelFromUrdf(str(URDF))
        self.pin_data = self.pin_model.createData()

    def mujoco(self, q: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
        model, data = self.mj_model, self.mj_data
        data.qpos[:] = q
        data.qvel[:] = v
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        # The URDF's collision shapes touch at the joints, so MuJoCo adds contact
        # forces; the unconstrained acceleration is the comparable drift.
        return {
            "M": mass,
            "bias": data.qfrc_bias.copy(),
            "drift": data.qacc_smooth.copy(),
        }

    def pinocchio(self, q: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
        model, data = self.pin_model, self.pin_data
        mass = pinocchio.crba(model, data, q)
        mass = np.triu(mass) + np.triu(mass, 1).T
        return {
            "M": mass,
            "bias": pinocchio.rnea(model, data, q, v, np.zeros(model.nv)),
            "gravity": pinocchio.computeGeneralizedGravity(model, data, q),
            "drift": pinocchio.aba(model, data, q, v, np.zeros(model.nv)),
        }

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray):
        return pinocchio.rnea(self.pin_model, self.pin_data, q, v, a)


@pytest.fixture(scope="module")
def reference() -> _Reference:
    return _Reference()


@pytest.mark.parametrize(("q", "v"), STATES)
def test_bias_gravity_drift_match_mujoco_and_pinocchio(
    engine: OpenSimPhysicsEngine, reference: _Reference, q, v
) -> None:
    engine.set_state(q, v)
    mj, pin = reference.mujoco(q, v), reference.pinocchio(q, v)
    np.testing.assert_allclose(mj["bias"], pin["bias"], atol=ATOL, rtol=0)

    np.testing.assert_allclose(engine.compute_mass_matrix(), pin["M"], atol=ATOL)
    np.testing.assert_allclose(engine.compute_bias_forces(), mj["bias"], atol=ATOL)
    np.testing.assert_allclose(engine.compute_bias_forces(), pin["bias"], atol=ATOL)
    np.testing.assert_allclose(
        engine.compute_gravity_forces(), pin["gravity"], atol=ATOL
    )
    np.testing.assert_allclose(
        engine.compute_drift_acceleration(), mj["drift"], atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(
        engine.compute_drift_acceleration(), pin["drift"], atol=1e-9, rtol=0
    )
    # Gravity leaves the state's velocities in place.
    np.testing.assert_array_equal(engine.get_state()[1], v)


@pytest.mark.parametrize(("q", "v"), STATES)
@pytest.mark.parametrize("a", [np.array([0.0, 0.0]), np.array([1.5, -2.0])])
def test_inverse_dynamics_matches_pinocchio(
    engine: OpenSimPhysicsEngine, reference: _Reference, q, v, a
) -> None:
    engine.set_state(q, v)
    tau = engine.compute_inverse_dynamics(a)

    assert tau.shape == (2,)
    np.testing.assert_allclose(
        tau, reference.inverse_dynamics(q, v, a), atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(
        tau, engine.compute_mass_matrix() @ a + engine.compute_bias_forces(), atol=ATOL
    )


@pytest.mark.parametrize(("q", "v"), STATES)
def test_counterfactuals_match_pinocchio(
    engine: OpenSimPhysicsEngine, reference: _Reference, q, v
) -> None:
    """ZTCF and ZVCF are the zero-control forward dynamics, not stale udot."""
    engine.set_state(np.array([0.2, 0.1]), np.array([0.0, 0.0]))
    pin_ztcf = reference.pinocchio(q, v)["drift"]
    pin_zvcf = reference.pinocchio(q, np.zeros(2))["drift"]

    np.testing.assert_allclose(engine.compute_ztcf(q, v), pin_ztcf, atol=1e-9, rtol=0)
    np.testing.assert_allclose(engine.compute_zvcf(q), pin_zvcf, atol=1e-9, rtol=0)
    # The caller's state is restored.
    np.testing.assert_array_equal(engine.get_state()[0], [0.2, 0.1])


def test_inverse_dynamics_rejects_wrong_length(engine: OpenSimPhysicsEngine) -> None:
    with pytest.raises(ValueError, match="2 entries"):
        engine.compute_inverse_dynamics(np.zeros(3))


def test_reset_after_step_restores_initial_state(engine: OpenSimPhysicsEngine) -> None:
    q0, v0 = engine.get_state()
    engine.set_state(np.array([0.4, -0.2]), np.array([0.5, 0.1]))
    engine.step(0.01)
    assert engine.get_time() == pytest.approx(0.01)

    engine.reset()

    assert engine.get_time() == 0.0
    q, v = engine.get_state()
    np.testing.assert_array_equal(q, q0)
    np.testing.assert_array_equal(v, v0)
    # The engine still integrates after a reset.
    engine.set_state(np.array([0.4, -0.2]), np.array([0.5, 0.1]))
    engine.step(0.01)
    assert engine.get_time() == pytest.approx(0.01)


def test_joint_names_follow_the_coordinates(engine: OpenSimPhysicsEngine) -> None:
    assert engine.get_joint_names() == list(JOINTS)


def test_capabilities_match_what_these_tests_prove(
    engine: OpenSimPhysicsEngine,
) -> None:
    capabilities = engine.get_capabilities()

    assert capabilities.mass_matrix is CapabilityLevel.FULL
    assert capabilities.inverse_dynamics is CapabilityLevel.FULL
    # Drift uses the model's forces at the state's muscle activations, which
    # zero controls do not remove, so it stays partial for muscle models.
    assert capabilities.drift_acceleration is CapabilityLevel.PARTIAL


GOLFER = (
    Path(__file__).resolve().parents[3]
    / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
)


def test_golfer_joint_names_name_each_state_coordinate() -> None:
    """Entry k of get_joint_names names q[k], although declaration order differs."""
    golfer = OpenSimPhysicsEngine()
    golfer.load_from_path(str(GOLFER))
    q, v = golfer.get_state()
    names = golfer.get_joint_names()
    assert len(names) == len(q) == len(v) == 39
    assert len(set(names)) == len(names)

    probe = 0.001 * np.arange(1, len(q) + 1)
    golfer.set_state(probe, np.zeros_like(v))
    coordinates = golfer._model.getCoordinateSet()  # noqa: SLF001 - checks the engine
    for k, name in enumerate(names):
        value = coordinates.get(name).getValue(golfer._state)  # noqa: SLF001
        assert value == pytest.approx(probe[k], abs=1e-15)

    bias = golfer.compute_bias_forces()
    assert bias.shape == (39,)
    assert np.all(np.isfinite(bias))
