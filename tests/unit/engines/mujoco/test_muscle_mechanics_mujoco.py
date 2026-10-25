"""NSX-94 (#12723): MuJoCo-muscle (MyoSuite) frame, rigid-tendon stiffness, sweep.

A two-muscle antagonist elbow written as MJCF (spatial tendons, ``muscle``
actuators) stands in for MyoSuite, which shares exactly this actuator model; a
second set runs the frame and the co-contraction sweep on the real MyoSuite
elbow (``myoelbow_1dof6muscles``) from the pinned ``myo_sim`` checkout.

Oracles (all independent of the provider's own difference):

* moment arm ``R = -dL/dq`` -- central difference of ``actuator_length``
  (step ``1e-6``: truncation ``~1e-13``, round-off ``~1e-10``; tolerance ``1e-7``);
* rigid-tendon stiffness ``k_m = dT/dL`` -- ratio of central differences of the
  tension to the *actuator length* (not the coordinate; tension is positive in tension, so
  ``k_m > 0`` on the ascending limb), step ``1e-6`` rad,
  tolerance ``1e-5`` relative;
* the secant law -- MuJoCo muscle tension is linear in the activation at fixed
  length and velocity, so ``linearity_residual`` must vanish to round-off
  (``1e-9`` of the torque, ~1e6 eps).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.myosuite.python.muscle_mechanics import (  # noqa: E402
    mujoco_muscle_frame,
)
from src.engines.physics_engines.myosuite.python.golfer_scene import (  # noqa: E402
    MYO_SIM_RELATIVE,
    REPO_ROOT_DEFAULT,
)
from src.shared.python.loop_nullspace.evidence import AllocationDependence  # noqa: E402
from src.shared.python.loop_nullspace.muscle_capability import (  # noqa: E402
    MuscleCapabilityUnavailable,
    default_muscle_registry,
)
from src.shared.python.loop_nullspace.muscle_cocontraction import (  # noqa: E402
    SweepConfig,
    SweepStatus,
    cocontraction_sweep,
)
from src.shared.python.loop_nullspace.muscle_fiber import total_activation  # noqa: E402
from src.shared.python.loop_nullspace.muscle_frame import TendonKind  # noqa: E402
from src.shared.python.loop_nullspace.muscle_noise import (  # noqa: E402
    muscle_noise_spec,
)
from src.shared.python.loop_nullspace.muscle_stiffness import (  # noqa: E402
    StiffnessModel,
    muscle_stiffness,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

XML = """
<mujoco>
  <option gravity="0 0 0" timestep="0.001"/>
  <worldbody>
    <body name="upper">
      <geom type="capsule" fromto="0 0 0 0 -0.3 0" size="0.02" mass="1"/>
      <site name="o_flex" pos="0.03 -0.05 0"/>
      <site name="o_ext" pos="-0.03 -0.05 0"/>
      <body name="fore" pos="0 -0.3 0">
        <joint name="elbow" type="hinge" axis="0 0 1" range="-0.1 2.5"/>
        <geom type="capsule" fromto="0 0 0 0 -0.25 0" size="0.02" mass="1"/>
        <site name="i_flex" pos="0.03 -0.05 0"/>
        <site name="i_ext" pos="-0.03 -0.05 0"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_flex"><site site="o_flex"/><site site="i_flex"/></spatial>
    <spatial name="t_ext"><site site="o_ext"/><site site="i_ext"/></spatial>
  </tendon>
  <actuator>
    <muscle name="flexor" tendon="t_flex" force="1000" lengthrange="0.1 0.4"/>
    <muscle name="extensor" tendon="t_ext" force="1200" lengthrange="0.1 0.4"/>
  </actuator>
</mujoco>
"""

Q_STEP = 1e-6


@pytest.fixture(scope="module")
def elbow():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.0
    mujoco.mj_forward(model, data)
    return model, data


def _length_and_tension(model, qpos, level):
    work = mujoco.MjData(model)
    work.qpos[0] = qpos
    work.act[:] = level
    mujoco.mj_forward(model, work)
    return np.array(work.actuator_length), -np.array(work.actuator_force)


def test_muscle_gain_constant_matches_the_mujoco_enum() -> None:
    from src.engines.physics_engines.myosuite.python import muscle_mechanics as mm

    assert int(mujoco.mjtGain.mjGAIN_MUSCLE) == mm._MUSCLE_GAIN
    assert int(mujoco.mjtDyn.mjDYN_NONE) == mm._NO_DYNAMICS


def test_moment_arms_equal_minus_actuator_length_gradient(elbow) -> None:
    model, data = elbow
    frame = mujoco_muscle_frame(model, data)
    assert frame.tendon is TendonKind.RIGID and frame.tendon_stiffness_grid is None
    assert frame.names == ("flexor", "extensor") and frame.coordinates == ("elbow",)
    len_p, _ = _length_and_tension(model, Q_STEP, 0.0)
    len_m, _ = _length_and_tension(model, -Q_STEP, 0.0)
    np.testing.assert_allclose(
        frame.moment_arms[:, 0], -(len_p - len_m) / (2 * Q_STEP), atol=1e-7
    )
    assert frame.moment_arms[0, 0] * frame.moment_arms[1, 0] < 0.0
    np.testing.assert_allclose(frame.max_isometric_force, [1000.0, 1200.0])


def test_tension_is_linear_in_activation(elbow) -> None:
    model, data = elbow
    frame = mujoco_muscle_frame(model, data, grid=(0.0, 0.25, 0.5, 1.0))
    assert frame.linearity_residual() < 1e-9


def test_fibre_stiffness_equals_minus_tension_slope_against_length(elbow) -> None:
    model, data = elbow
    frame = mujoco_muscle_frame(model, data)
    assert frame.fiber_stiffness_grid is not None
    for g, level in enumerate(frame.activation_grid):
        len_p, t_p = _length_and_tension(model, Q_STEP, level)
        len_m, t_m = _length_and_tension(model, -Q_STEP, level)
        oracle = (t_p - t_m) / (len_p - len_m)
        np.testing.assert_allclose(
            frame.fiber_stiffness_grid[g], oracle, rtol=1e-5, atol=1e-6
        )


def test_rigid_tendon_series_stiffness_is_the_fibre_stiffness(elbow) -> None:
    model, data = elbow
    frame = mujoco_muscle_frame(model, data)
    stiff = muscle_stiffness(frame, np.array([0.5, 0.5]), StiffnessModel.ENGINE_CURVE)
    assert np.all(np.isinf(stiff.tendon))
    np.testing.assert_allclose(stiff.muscle, stiff.fiber)


def test_geometry_term_matches_engine_torque_slope(elbow) -> None:
    model, data = elbow
    frame = mujoco_muscle_frame(model, data, with_geometry=True)
    act = np.array([0.5, 0.5])  # grid node
    joint = muscle_stiffness(frame, act, StiffnessModel.ENGINE_CURVE).joint[0, 0]

    def torque(qpos: float) -> float:
        work = mujoco.MjData(model)
        work.qpos[0] = qpos
        work.act[:] = act
        mujoco.mj_forward(model, work)
        return float(work.qfrc_actuator[0])

    k_engine = -(torque(Q_STEP) - torque(-Q_STEP)) / (2 * Q_STEP)
    assert joint == pytest.approx(k_engine, rel=1e-4)


def test_callers_data_is_not_modified(elbow) -> None:
    model, data = elbow
    qpos, act, ctrl = data.qpos.copy(), data.act.copy(), data.ctrl.copy()
    mujoco_muscle_frame(model, data, with_geometry=True)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.act, act)
    np.testing.assert_array_equal(data.ctrl, ctrl)


def test_selection_and_precondition_errors(elbow) -> None:
    model, data = elbow
    assert mujoco_muscle_frame(model, data, muscles=["extensor"]).names == ("extensor",)
    with pytest.raises(ValueError, match="actuators not in the model"):
        mujoco_muscle_frame(model, data, muscles=["nope"])
    with pytest.raises(ValueError, match="joints not in the model"):
        mujoco_muscle_frame(model, data, coordinates=["nope"])
    bare = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body>'
        "</worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="no muscle actuators"):
        mujoco_muscle_frame(bare, mujoco.MjData(bare))


def test_cocontraction_tradeoff_with_rigid_tendon_stiffness(elbow) -> None:
    model, data = elbow
    frame = mujoco_muscle_frame(model, data)
    cfg = SweepConfig(
        noise=muscle_noise_spec(frame, kappa=0.05),
        dt=1e-3,
        stiffness_model=StiffnessModel.ENGINE_CURVE,
        n_levels=5,
    )
    sweep = cocontraction_sweep(frame, np.array([10.0]), total_activation(2), cfg)
    assert sweep.status is SweepStatus.COMPUTED
    assert np.max(sweep.torque_residual) < 1e-12
    assert np.max(sweep.engine_torque_gap) < 1e-9 * 10.0
    assert (
        sweep.noise_trace.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    )
    assert np.all(np.isfinite(sweep.stiffness_trace.values))


def test_registry_declares_a_rigid_tendon_for_this_provider() -> None:
    reg = default_muscle_registry()
    assert reg.load("myosuite") is mujoco_muscle_frame
    with pytest.raises(MuscleCapabilityUnavailable, match="rigid tendon"):
        reg.require_series_tendon("myosuite")


# The real MyoSuite elbow comes from the pinned ``myo_sim`` gitlink (the same
# path the golfer scene resolves), not from an installed ``myo_sim`` package --
# no such top-level package exists, so an ``importorskip`` never ran this test.
MYO_ELBOW = Path("elbow") / "myoelbow_1dof6muscles.xml"
MYO_ELBOW_FLEXION = 1.0  # rad; mid-range of r_elbow_flex
# Joint torques (N m) inside the six-muscle capacity at that flexion, and one
# far outside it (the largest reachable flexion torque there is ~1e2 N m).
MYO_TORQUES = (2.0, 5.0)
MYO_UNREACHABLE_TORQUE = 200.0


@pytest.fixture(scope="module")
def myo_elbow_frame():
    path = REPO_ROOT_DEFAULT / MYO_SIM_RELATIVE / MYO_ELBOW
    if not path.is_file():
        pytest.skip(
            f"pinned myo_sim not checked out at {MYO_SIM_RELATIVE} "
            "(run scripts/setup_myosuite_models.sh)"
        )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[0] = MYO_ELBOW_FLEXION
    mujoco.mj_forward(model, data)
    return mujoco_muscle_frame(model, data)


def _myo_sweep(frame, torque: float):
    cfg = SweepConfig(
        noise=muscle_noise_spec(frame, kappa=0.05),
        dt=1e-3,
        stiffness_model=StiffnessModel.ENGINE_CURVE,
        n_levels=7,
    )
    return cocontraction_sweep(
        frame, np.array([torque]), total_activation(frame.n_muscles), cfg
    )


def test_real_myosuite_elbow_builds_a_rigid_tendon_frame(myo_elbow_frame) -> None:
    frame = myo_elbow_frame
    assert frame.n_muscles == 6 and frame.tendon is TendonKind.RIGID
    assert frame.coordinates == ("r_elbow_flex",)
    assert frame.linearity_residual() < 1e-9
    arms = frame.moment_arms[:, 0]
    assert np.any(arms > 0.0) and np.any(arms < 0.0)  # antagonists exist


@pytest.mark.parametrize("torque", MYO_TORQUES)
def test_real_myosuite_elbow_cocontraction_tradeoff(myo_elbow_frame, torque) -> None:
    sweep = _myo_sweep(myo_elbow_frame, torque)
    assert sweep.status is SweepStatus.COMPUTED
    assert np.max(sweep.torque_residual) < 1e-12
    assert np.max(sweep.engine_torque_gap) < 1e-9 * torque
    assert (
        sweep.noise_trace.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    )
    stiffness = sweep.stiffness_trace.values.ravel()
    noise = sweep.noise_trace.values.ravel()
    assert np.all(np.isfinite(stiffness)) and np.all(np.isfinite(noise))
    # The trade-off: co-contraction buys joint stiffness and costs motor noise.
    assert np.all(np.diff(stiffness) > 0.0)
    assert np.all(np.diff(noise) > 0.0)


def test_real_myosuite_elbow_unreachable_torque_is_not_qualified(
    myo_elbow_frame,
) -> None:
    sweep = _myo_sweep(myo_elbow_frame, MYO_UNREACHABLE_TORQUE)
    assert sweep.status is SweepStatus.NOT_QUALIFIED
    assert sweep.reasons
