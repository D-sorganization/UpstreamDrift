"""NSX-93 (#12722): allocated, impedance and feedback effort on separate lines."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.feedback_comparison import ReplayPolicy
from src.shared.python.loop_nullspace.feedback_layer import (
    DelayedFeedback,
    FeedbackLayer,
    GainSchedule,
)
from src.shared.python.loop_nullspace.impedance_spec import (
    ImpedanceWindow,
    JointImpedance,
)
from src.shared.python.loop_nullspace.layer_accounting import (
    account_layer,
    effort_cost,
)
from src.shared.python.loop_nullspace.replay import ReplaySettings, replay
from tests.unit.loop_nullspace._layer_fixtures import (
    PointMass,
    point_mass_setup,
    zero_torque_allocation,
)

pytestmark = pytest.mark.unit

T_END, KP, KD = 1.0, 200.0, 20.0


def _layer(limit=None) -> FeedbackLayer:
    imp = JointImpedance(
        (ImpedanceWindow(0.0, T_END, np.array([0.5 * KP]), np.array([0.5 * KD])),)
    )
    sched = GainSchedule(
        np.array([0.0, T_END]), np.array([[[0.5 * KP]]]), np.array([[[0.5 * KD]]])
    )
    return FeedbackLayer(1, imp, DelayedFeedback(sched, delay_s=0.01), limit)


def _run(layer):
    setup = point_mass_setup(T_END)
    report = replay(
        zero_torque_allocation(T_END, 101, 1),
        PointMass(),
        policy=ReplayPolicy.STATE_FEEDBACK,
        setup=setup,
        feedback=layer,
        settings=ReplaySettings(substeps=2, divergence_tolerance=1e6),
    )
    return report, setup


def test_effort_cost_closed_form() -> None:
    t = np.linspace(0.0, 2.0, 201)
    tau = np.column_stack([3.0 * np.ones_like(t)])
    omega = np.column_stack([np.where(t < 1.0, 2.0, -2.0)])
    cost = effort_cost(t, tau, omega)
    assert cost.l2_effort == pytest.approx(9.0 * 2.0)
    assert cost.positive_work_j == pytest.approx(6.0, abs=0.04)  # step in the trapezoid
    assert cost.negative_work_j == pytest.approx(6.0, abs=0.04)
    assert cost.net_work_j == pytest.approx(0.0, abs=0.08)
    with pytest.raises(ValueError, match="conform"):
        effort_cost(t, tau, omega[:-1])


def test_the_three_lines_are_separate_and_add_up() -> None:
    layer = _layer()
    report, setup = _run(layer)
    acc = account_layer(report, layer, setup)
    assert not acc.saturated
    assert acc.impedance is not None and acc.feedback is not None
    np.testing.assert_allclose(
        acc.impedance.torque + acc.feedback.torque,
        report.feedback_torque,
        rtol=0,
        atol=1e-12,
    )
    # no allocated torque at all: the layer's effort is not hidden in that line
    assert acc.allocated.l2_effort == 0.0 and acc.allocated.abs_work_j == 0.0
    assert acc.total.cost.l2_effort > 0.0
    # both parts spend something, and they are different series
    assert acc.impedance.cost.l2_effort > 0.0 and acc.feedback.cost.l2_effort > 0.0
    assert not np.allclose(acc.impedance.torque, acc.feedback.torque)
    # the replay's own budget is the total line
    assert report.feedback_budget is not None
    assert acc.total.budget.energy_ratio == report.feedback_budget.energy_ratio
    assert acc.hypotheses == ("H-DECLARED-JOINT-IMPEDANCE", "H-SENSORIMOTOR-DELAY")


def test_the_impedance_dissipates_energy_and_stores_elastic_energy() -> None:
    layer = _layer()
    report, setup = _run(layer)
    acc = account_layer(report, layer, setup)
    assert acc.impedance_energy is not None
    assert acc.impedance_energy.dissipated_j > 0.0
    # initial displacement 0.1 m held by K = 0.5 kp: peak 1/2 K x0^2
    assert acc.impedance_energy.peak_elastic_j == pytest.approx(
        0.5 * (0.5 * KP) * 0.1**2, rel=1e-9
    )
    # the damper removes energy: its work on the mass is net negative
    assert acc.impedance is not None
    assert acc.impedance.cost.negative_work_j > 0.0


def test_a_declared_saturation_is_reported() -> None:
    layer = _layer(limit=np.array([2.0]))
    report, setup = _run(layer)
    acc = account_layer(report, layer, setup)
    assert acc.saturated
    assert np.max(np.abs(report.feedback_torque)) == pytest.approx(2.0)


def test_account_layer_rejects_a_foreign_report() -> None:
    layer = _layer()
    report, setup = _run(layer)
    with pytest.raises(ValueError, match="this layer"):
        account_layer(report, _layer(), setup)
    with pytest.raises(TypeError, match="FeedbackLayer"):
        account_layer(report, object(), setup)  # type: ignore[arg-type]
