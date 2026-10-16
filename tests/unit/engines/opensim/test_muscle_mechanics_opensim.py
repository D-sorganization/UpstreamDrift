"""NSX-94 (#12723): OpenSim muscle frame, series stiffness and the sweep.

Oracles are the engine itself, queried differently from the code under test:

* moment arm ``R = -dL_MT/dq`` -- central difference of ``getLength`` (step
  ``1e-6``: truncation ``h^2 L'''/6 ~ 1e-13``, round-off ``eps L / h ~ 1e-10``;
  tolerance ``1e-7``);
* series stiffness ``dF_T/dL_MT = k_m k_T / (k_m + k_T)`` -- central difference
  of the *re-equilibrated* tendon force against a coordinate change (step
  ``1e-4`` rad: the equilibrium solve is accurate to ~1e-10 relative, so the
  difference is good to ~1e-6; tolerance ``1e-4`` relative);
* joint stiffness ``K = -d tau / d q`` at fixed activations -- central
  difference of the engine joint torque (same step and tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.helpers.opensim_support import require_opensim

osim = require_opensim()

from src.engines.physics_engines.opensim.python.muscle_mechanics import (  # noqa: E402
    opensim_muscle_frame,
)
from src.shared.python.loop_nullspace.evidence import AllocationDependence  # noqa: E402
from src.shared.python.loop_nullspace.muscle_capability import (  # noqa: E402
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
    StiffnessNotQualified,
    muscle_stiffness,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_opensim]

Q_STEP = 1e-4


def _build(rigid: bool = False, tendon_slack: float = 0.18):
    osim.Logger.setLevelString("error")
    model = osim.Model()
    model.setName("elbow-nsx94")
    ground = model.getGround()
    body = osim.Body(
        "forearm", 1.0, osim.Vec3(0, -0.15, 0), osim.Inertia(0.02, 0.02, 0.005)
    )
    model.addBody(body)
    joint = osim.PinJoint(
        "elbow", ground, osim.Vec3(0), osim.Vec3(0), body, osim.Vec3(0), osim.Vec3(0)
    )
    joint.upd_coordinates(0).setName("flex")
    model.addJoint(joint)
    for name, f_max, x in (("flexor", 1000.0, 0.03), ("extensor", 1200.0, -0.03)):
        mus = osim.Millard2012EquilibriumMuscle(name, f_max, 0.12, tendon_slack, 0.0)
        mus.set_ignore_tendon_compliance(rigid)
        mus.addNewPathPoint(name + "_o", ground, osim.Vec3(x, 0.25, 0.0))
        mus.addNewPathPoint(name + "_i", body, osim.Vec3(x, -0.04, 0.0))
        model.addForce(mus)
    state = model.initSystem()
    model.getCoordinateSet().get(0).setValue(state, 0.0)
    model.realizePosition(state)
    return model, state


@pytest.fixture(scope="module")
def elbow():
    return _build()


def _tension_at(model, state, q, activation):
    work = osim.State(state)
    model.getCoordinateSet().get(0).setValue(work, q)
    model.realizePosition(work)
    for k in range(model.getMuscles().getSize()):
        model.getMuscles().get(k).setActivation(work, float(activation[k]))
    model.equilibrateMuscles(work)
    model.realizeDynamics(work)
    coord = model.getCoordinateSet().get(0)
    mus = [model.getMuscles().get(k) for k in range(model.getMuscles().getSize())]
    tens = np.array([m.getTendonForce(work) for m in mus])
    arms = np.array([m.computeMomentArm(work, coord) for m in mus])
    lens = np.array([m.getLength(work) for m in mus])
    return tens, arms, lens


def test_frame_moment_arms_equal_minus_length_gradient(elbow) -> None:
    model, state = elbow
    frame = opensim_muscle_frame(model, state)
    assert frame.engine == "opensim" and frame.tendon is TendonKind.COMPLIANT
    assert frame.names == ("flexor", "extensor") and frame.coordinates == ("flex",)
    h = 1e-6
    _, _, len_p = _tension_at(model, state, h, np.zeros(2))
    _, _, len_m = _tension_at(model, state, -h, np.zeros(2))
    np.testing.assert_allclose(
        frame.moment_arms[:, 0], -(len_p - len_m) / (2 * h), atol=1e-7
    )
    assert frame.moment_arms[0, 0] > 0.0 > frame.moment_arms[1, 0]
    np.testing.assert_allclose(frame.max_isometric_force, [1000.0, 1200.0])
    assert np.all(frame.force_grid[-1] > frame.force_grid[0])


def test_series_stiffness_equals_engine_tendon_force_slope(elbow) -> None:
    model, state = elbow
    frame = opensim_muscle_frame(model, state)
    for level in (0.25, 1.0):  # grid nodes: between nodes the table is interpolated
        act = np.full(2, level)
        k_series = muscle_stiffness(frame, act, StiffnessModel.ENGINE_CURVE).muscle
        f_p, _, l_p = _tension_at(model, state, Q_STEP, act)
        f_m, _, l_m = _tension_at(model, state, -Q_STEP, act)
        np.testing.assert_allclose(k_series, (f_p - f_m) / (l_p - l_m), rtol=1e-4)
        # The series element is strictly softer than either spring.
        stiff = muscle_stiffness(frame, act, StiffnessModel.ENGINE_CURVE)
        assert np.all(stiff.muscle < stiff.fiber) and np.all(
            stiff.muscle < stiff.tendon
        )


def test_joint_stiffness_matches_engine_torque_slope(elbow) -> None:
    model, state = elbow
    frame = opensim_muscle_frame(model, state, with_geometry=True)
    assert frame.moment_arm_jacobian is not None
    act = np.array([0.5, 0.25])  # grid nodes
    stiff = muscle_stiffness(frame, act, StiffnessModel.ENGINE_CURVE)
    assert stiff.geometric_term_included

    def torque(q: float) -> float:
        tens, arms, _ = _tension_at(model, state, q, act)
        return float(arms @ tens)

    k_engine = -(torque(Q_STEP) - torque(-Q_STEP)) / (2 * Q_STEP)
    assert stiff.joint[0, 0] == pytest.approx(k_engine, rel=1e-4)


def test_frame_does_not_modify_the_callers_state(elbow) -> None:
    model, state = elbow
    before = [
        model.getStateVariableValue(state, f"/forceset/{name}/activation")
        for name in ("flexor", "extensor")
    ]
    q_before = model.getCoordinateSet().get(0).getValue(state)
    opensim_muscle_frame(model, state, with_geometry=True)
    after = [
        model.getStateVariableValue(state, f"/forceset/{name}/activation")
        for name in ("flexor", "extensor")
    ]
    assert after == before
    assert model.getCoordinateSet().get(0).getValue(state) == q_before


def test_selection_and_precondition_errors(elbow) -> None:
    model, state = elbow
    one = opensim_muscle_frame(model, state, muscles=["extensor"])
    assert one.names == ("extensor",)
    with pytest.raises(ValueError, match="muscles not in the model"):
        opensim_muscle_frame(model, state, muscles=["nope"])
    with pytest.raises(ValueError, match="coordinates not in the model"):
        opensim_muscle_frame(model, state, coordinates=["nope"])


def test_descending_limb_muscle_fails_closed_on_stiffness() -> None:
    # Tendon slack 0.15 m puts the fibre past optimal length (path 0.29 m): the
    # active force-length slope is negative there, so dF/dl is not a stiffness.
    model, state = _build(tendon_slack=0.15)
    frame = opensim_muscle_frame(model, state)
    with pytest.raises(StiffnessNotQualified, match="descending limb"):
        muscle_stiffness(frame, np.array([0.6, 0.6]), StiffnessModel.ENGINE_CURVE)
    cfg = SweepConfig(
        noise=muscle_noise_spec(frame, kappa=0.05),
        dt=1e-3,
        stiffness_model=StiffnessModel.ENGINE_CURVE,
        n_levels=3,
    )
    sweep = cocontraction_sweep(frame, np.array([10.0]), total_activation(2), cfg)
    assert (
        sweep.stiffness_trace.allocation_dependence is AllocationDependence.UNAVAILABLE
    )
    assert "descending limb" in (sweep.stiffness_trace.reason or "")
    assert (
        sweep.noise_trace.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    )


def test_rigid_tendon_model_is_flagged_rigid_without_tendon_stiffness() -> None:
    model, state = _build(rigid=True)
    frame = opensim_muscle_frame(model, state)
    assert frame.tendon is TendonKind.RIGID
    assert frame.tendon_stiffness_grid is None


def test_cocontraction_tradeoff_in_opensim(elbow) -> None:
    model, state = elbow
    frame = opensim_muscle_frame(model, state)
    cfg = SweepConfig(
        noise=muscle_noise_spec(frame, kappa=0.05),
        dt=1e-3,
        stiffness_model=StiffnessModel.ENGINE_CURVE,
        n_levels=7,
        linearity_limit=0.10,
    )
    sweep = cocontraction_sweep(frame, np.array([15.0]), total_activation(2), cfg)
    assert sweep.status is SweepStatus.COMPUTED
    assert np.max(sweep.torque_residual) < 1e-12
    # The engine's own (nonlinear) torque differs from the secant law the fibre
    # is built on by no more than the declared linearity limit of the torque.
    assert np.max(sweep.engine_torque_gap) <= 0.10 * 15.0
    assert (
        sweep.noise_trace.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    )
    assert sweep.stiffness_trace.hypotheses == ("H-MUSCLE-SHORT-RANGE-STIFFNESS",)
    assert sweep.noise_trace.values[-1] > sweep.noise_trace.values[0]
    assert sweep.stiffness_trace.values[-1] > sweep.stiffness_trace.values[0]


def test_unreachable_torque_is_not_qualified(elbow) -> None:
    model, state = elbow
    frame = opensim_muscle_frame(model, state)
    cfg = SweepConfig(
        noise=muscle_noise_spec(frame, kappa=0.05), dt=1e-3, stiffness_model=None
    )
    sweep = cocontraction_sweep(frame, np.array([200.0]), total_activation(2), cfg)
    assert sweep.status is SweepStatus.NOT_QUALIFIED
    assert sweep.noise_trace.allocation_dependence is AllocationDependence.UNAVAILABLE


def test_registry_loads_the_opensim_provider() -> None:
    assert default_muscle_registry().load("opensim") is opensim_muscle_frame
