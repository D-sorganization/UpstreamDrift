"""NSX-94 (#12723): the co-contraction sweep, typed with the NSX-81 evidence axes.

Tolerance for torque invariance across the sweep: see ``test_muscle_fiber`` --
``(m + 2) eps sum|G_k| = 1.2e-13`` N m for the elbow, so ``1e-12`` (the issue's
figure) holds with a factor eight of headroom.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import CapabilityLevel
from src.shared.python.loop_nullspace.evidence import AllocationDependence
from src.shared.python.loop_nullspace.muscle_capability import (
    MuscleCapabilityUnavailable,
    default_muscle_registry,
)
from src.shared.python.loop_nullspace.muscle_cocontraction import (
    SweepConfig,
    SweepStatus,
    cocontraction_sweep,
)
from src.shared.python.loop_nullspace.muscle_fiber import (
    FiberInfeasibleError,
    muscle_subset_index,
    total_activation,
)
from src.shared.python.loop_nullspace.muscle_noise import muscle_noise_spec
from src.shared.python.loop_nullspace.muscle_stiffness import StiffnessModel
from tests.unit.loop_nullspace._muscle_fixtures import (
    PAIR_ARM,
    PAIR_FORCE,
    PAIR_KAPPA,
    PAIR_RHO,
    elbow_frame,
    pair_frame,
)

pytestmark = pytest.mark.unit

DT = 1e-3


def _config(frame, *, stiffness: bool = True, kappa: float = PAIR_KAPPA, **kw):
    return SweepConfig(
        noise=muscle_noise_spec(frame, kappa=kappa, sigma0=0.0),
        dt=DT,
        stiffness_model=StiffnessModel.FORCE_PROPORTIONAL if stiffness else None,
        force_gain=PAIR_RHO if stiffness else None,
        **kw,
    )


def test_net_torque_is_invariant_over_the_sweep() -> None:
    frame = elbow_frame()
    sweep = cocontraction_sweep(
        frame, np.array([20.0]), total_activation(3), _config(frame, n_levels=9)
    )
    assert sweep.status is SweepStatus.COMPUTED
    assert sweep.levels.shape == (9,)
    assert np.max(sweep.torque_residual) < 1e-12
    np.testing.assert_allclose(sweep.net_torque.values, [20.0], atol=1e-12)
    assert sweep.net_torque.allocation_dependence is AllocationDependence.INVARIANT
    assert sweep.fiber_nullity == 2


def test_cocontraction_raises_muscle_noise_at_fixed_net_torque() -> None:
    frame = elbow_frame()
    sweep = cocontraction_sweep(
        frame, np.array([20.0]), total_activation(3), _config(frame, n_levels=9)
    )
    trace = sweep.noise_trace.values
    assert trace[-1] > trace[0]
    # The weights F_max differ across muscles, so the noise minimum need not sit
    # at the lowest total activation; beyond it the noise only rises.
    start = int(np.argmin(trace))
    assert start <= 2
    assert np.all(np.diff(trace[start:]) > 0.0)
    rec = sweep.noise_trace
    assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert rec.hypotheses == ("H-NOISE-SIGNAL-DEPENDENT",)


def test_pair_noise_matches_the_closed_form_across_levels() -> None:
    frame = pair_frame()
    tau = 8.0
    d = tau / (PAIR_ARM * PAIR_FORCE)
    sweep = cocontraction_sweep(
        frame,
        np.array([tau]),
        total_activation(2),
        _config(frame, n_levels=7),
    )
    c = sweep.levels
    expected = PAIR_ARM**2 * (PAIR_KAPPA * PAIR_FORCE) ** 2 * (c**2 + d**2) / 2 / DT
    np.testing.assert_allclose(sweep.noise_trace.values, expected, rtol=1e-10)
    k = PAIR_ARM**2 * PAIR_RHO * PAIR_FORCE * c
    np.testing.assert_allclose(sweep.stiffness_trace.values, k, rtol=1e-10)
    assert sweep.stiffness_trace.hypotheses == ("H-MUSCLE-SHORT-RANGE-STIFFNESS",)
    # Raising co-contraction buys stiffness and costs noise: the trade-off.
    assert np.all(np.diff(sweep.stiffness_trace.values) > 0.0)
    assert np.all(np.diff(sweep.noise_trace.values) > 0.0)


def test_stiffness_is_unavailable_with_a_reason_when_not_declared() -> None:
    frame = pair_frame()
    sweep = cocontraction_sweep(
        frame,
        np.array([8.0]),
        total_activation(2),
        _config(frame, stiffness=False, n_levels=3),
    )
    rec = sweep.stiffness_trace
    assert rec.allocation_dependence is AllocationDependence.UNAVAILABLE
    assert "no stiffness model" in (rec.reason or "")
    assert sweep.noise_trace.values.shape == (3,)


def test_engine_curve_without_engine_stiffness_fails_closed_per_quantity() -> None:
    frame = pair_frame()
    cfg = SweepConfig(
        noise=muscle_noise_spec(frame, kappa=PAIR_KAPPA),
        dt=DT,
        stiffness_model=StiffnessModel.ENGINE_CURVE,
        n_levels=3,
    )
    sweep = cocontraction_sweep(frame, np.array([8.0]), total_activation(2), cfg)
    assert (
        sweep.stiffness_trace.allocation_dependence is AllocationDependence.UNAVAILABLE
    )
    assert "fibre stiffness" in (sweep.stiffness_trace.reason or "")
    assert sweep.status is SweepStatus.COMPUTED  # noise is still honest


def test_reserve_carrying_frame_is_not_qualified_and_has_no_values() -> None:
    frame = elbow_frame()
    sweep = cocontraction_sweep(
        frame, np.array([80.0]), total_activation(3), _config(frame)
    )
    assert sweep.status is SweepStatus.NOT_QUALIFIED
    assert sweep.levels.size == 0
    assert any("reserve" in r for r in sweep.reasons)
    for rec in (sweep.noise_trace, sweep.stiffness_trace, sweep.net_torque):
        assert rec.allocation_dependence is AllocationDependence.UNAVAILABLE
        assert "NOT_QUALIFIED" in (rec.reason or "")
        assert np.all(np.isnan(rec.values))


def test_levels_are_declared_inputs_and_out_of_range_fails_closed() -> None:
    frame = pair_frame()
    cfg = _config(frame, levels=(0.1,))
    with pytest.raises(FiberInfeasibleError, match="level"):
        cocontraction_sweep(frame, np.array([8.0]), total_activation(2), cfg)
    with pytest.raises(TypeError):
        cocontraction_sweep(frame, np.array([8.0]))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="levels or n_levels"):
        SweepConfig(
            noise=muscle_noise_spec(frame, kappa=0.1),
            dt=DT,
            stiffness_model=None,
            levels=(0.5,),
            n_levels=5,
        )


def test_antagonist_subset_index_sweeps_only_the_declared_muscles() -> None:
    frame = elbow_frame()
    index = muscle_subset_index(frame, ["triceps"])
    sweep = cocontraction_sweep(
        frame, np.array([20.0]), index, _config(frame, n_levels=5)
    )
    triceps = sweep.activations[:, 2]
    np.testing.assert_allclose(triceps, sweep.levels, atol=1e-12)
    assert np.all(np.diff(triceps) > 0.0)


# ------------------------------------------------------------------ registry


def test_registry_levels_and_fail_closed_reasons() -> None:
    reg = default_muscle_registry()
    assert reg.require("opensim").level is CapabilityLevel.FULL
    assert reg.require("myosuite").level is CapabilityLevel.PARTIAL
    reg.require_series_tendon("opensim")
    with pytest.raises(MuscleCapabilityUnavailable, match="rigid tendon"):
        reg.require_series_tendon("myosuite")
    for torque_engine in ("pinocchio", "drake", "simscape"):
        with pytest.raises(MuscleCapabilityUnavailable, match="NSX-93"):
            reg.require(torque_engine)
    with pytest.raises(MuscleCapabilityUnavailable, match="not declared"):
        reg.require("nonesuch")
    with pytest.raises(MuscleCapabilityUnavailable, match="spec full-body"):
        reg.load("mujoco")


def test_registry_rejects_inconsistent_declarations() -> None:
    reg = default_muscle_registry()
    with pytest.raises(ValueError, match="already declared"):
        reg.declare("opensim", CapabilityLevel.NONE, reason="x")
    with pytest.raises(ValueError, match="reason"):
        reg.declare("new", CapabilityLevel.NONE)
    with pytest.raises(ValueError, match="tendon"):
        reg.declare("new", CapabilityLevel.PARTIAL)
