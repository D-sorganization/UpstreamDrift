"""Tests for canonical Tools rows executed by the analytical pendulum."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.canonical_trial_executor import (
    execute_serial_variation,
)
from src.shared.python.perturbation.double_pendulum_trial_adapter import (
    DoublePendulumTrialAdapter,
    DoublePendulumTrialConfig,
    DoublePendulumTrialResult,
)
from src.shared.python.simulation_backends import GolfModelParams, SimState

pytestmark = pytest.mark.unit

_PLAN_SHA = "a" * 64
_SCENARIO_SHA = "d" * 64
_TOOLS_REVISION = "17474249b9267d0e73a779c1d72f231e7b8de39c"
_ENGINE_REVISION = "c" * 40
_SHOULDER_DAMPING = "swing_sim.swing.damping_shoulder"
_WRIST_TORQUE = "swing_sim.swing.wrist_commanded_torque_offset_nm"


@dataclass(frozen=True)
class _Spec:
    variable_key: str
    time_window_s: tuple[float, float] | None = None
    point_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Plan:
    noise: tuple[_Spec, ...]
    n_runs: int = 1
    seed: int = 19
    mode: str = "swing"


class _Gateway:
    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples

    def sample_inputs(self, _plan: object) -> np.ndarray:
        return self._samples


def _config(
    *, target_position_m: tuple[float, float, float]
) -> DoublePendulumTrialConfig:
    return DoublePendulumTrialConfig(
        model_params=GolfModelParams.default(),
        initial_state=SimState(
            q=np.radians(np.array([-45.0, -90.0])),
            v=np.array([2.0, 1.0]),
        ),
        duration_s=0.04,
        dt_s=0.01,
        base_torques_nm=(4.0, 0.5),
        target_position_m=target_position_m,
        contact_radius_m=0.01,
        frame_id="pendulum-plane:x-forward-y-out-z-up",
        alignment_id="downswing-start/v1",
    )


def _adapter(
    plan: _Plan, config: DoublePendulumTrialConfig
) -> DoublePendulumTrialAdapter:
    return DoublePendulumTrialAdapter(
        plan=plan,
        config=config,
        plan_sha256=_PLAN_SHA,
        scenario_sha256=_SCENARIO_SHA,
        tools_revision=_TOOLS_REVISION,
        engine_revision=_ENGINE_REVISION,
    )


def test_adapter_is_exposed_by_public_perturbation_package() -> None:
    assert perturbation.DoublePendulumTrialAdapter is DoublePendulumTrialAdapter
    assert perturbation.DoublePendulumTrialConfig is DoublePendulumTrialConfig


def test_real_ode_trial_retains_controls_markers_and_hit_evidence() -> None:
    plan = _Plan(
        noise=(
            _Spec(_SHOULDER_DAMPING),
            _Spec(_WRIST_TORQUE, (0.01, 0.03), ("joint.wrist",)),
        )
    )
    # The initial clubhead location is a declared target, making the event
    # deterministic without inventing a ball-flight result.
    target = (-1.237436867076458, 0.0, 0.17677669529663698)
    adapter = _adapter(plan, _config(target_position_m=target))

    result = adapter.run(np.array([0.4, -2.0]))

    assert isinstance(result, DoublePendulumTrialResult)
    assert result.trace.backend == "ode"
    assert result.trace.q.shape == (5, 2)
    assert result.trace.markers is not None
    assert result.trace.markers.shape == (5, 2, 3)
    assert result.trace.u is not None
    np.testing.assert_allclose(result.trace.u[:, 1], [0.5, -1.5, -1.5, 0.5, 0.0])
    assert result.contact_observed is True

    evidence = adapter.collect_success(0, plan.seed, np.array([0.4, -2.0]), result)
    assert evidence.outcome == "hit"
    assert evidence.scenario_sha256 == _SCENARIO_SHA
    assert len(evidence.execution_config_sha256) == 64
    assert evidence.impact is not None
    assert evidence.impact.time_s == 0.0
    assert evidence.trace is not None
    assert evidence.trace.coordinate_ids == (
        "joint.shoulder.angle",
        "joint.wrist.relative_angle",
    )
    assert evidence.trace.marker_ids == ("joint.wrist", "clubhead.center")
    assert evidence.closest_approach is not None
    assert evidence.closest_approach.contact_observed is True


def test_damping_sample_changes_the_actual_ode_trajectory() -> None:
    plan = _Plan(noise=(_Spec(_SHOULDER_DAMPING),), n_runs=2)
    config = _config(target_position_m=(20.0, 0.0, 20.0))
    adapter = _adapter(plan, config)

    low = adapter.run(np.array([0.0]))
    high = adapter.run(np.array([2.0]))

    assert not np.allclose(low.trace.v, high.trace.v)


def test_evidence_collection_rejects_a_trace_without_geometry() -> None:
    plan = _Plan(noise=(_Spec(_SHOULDER_DAMPING),))
    adapter = _adapter(plan, _config(target_position_m=(20.0, 0.0, 20.0)))
    result = adapter.run(np.array([0.4]))
    incomplete = replace(result, trace=replace(result.trace, markers=None))

    with pytest.raises(ValueError, match="must retain markers"):
        adapter.collect_success(0, plan.seed, np.array([0.4]), incomplete)


def test_serial_execution_retains_legitimate_miss_and_domain_failure() -> None:
    plan = _Plan(noise=(_Spec(_SHOULDER_DAMPING),), n_runs=2)
    adapter = _adapter(plan, _config(target_position_m=(20.0, 0.0, 20.0)))

    records = execute_serial_variation(
        plan,
        _Gateway(np.array([[0.4], [-0.1]])),
        adapter.run,
        adapter,
    )

    assert [record.outcome for record in records] == [
        "no_impact",
        "numerical_failure",
    ]
    assert records[0].closest_approach is not None
    assert records[0].impact is None
    assert records[1].trace is None
    assert records[1].failure_reason is not None
    assert "damping_shoulder" in records[1].failure_reason


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (_Plan(noise=(_Spec("swing_sim.swing.yaw_deg"),)), "unsupported"),
        (_Plan(noise=(_Spec(_WRIST_TORQUE),)), "time window"),
        (
            _Plan(noise=(_Spec(_WRIST_TORQUE, (0.0, 0.02), ("joint.shoulder",)),)),
            "point",
        ),
        (
            _Plan(noise=(_Spec(_WRIST_TORQUE, (0.0, 0.05), ("joint.wrist",)),)),
            "duration",
        ),
    ],
)
def test_adapter_rejects_unsupported_or_invalid_plan_loci(
    plan: _Plan, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _adapter(plan, _config(target_position_m=(20.0, 0.0, 20.0)))


def test_config_rejects_nonintegral_horizon_and_invalid_contact_radius() -> None:
    base = _config(target_position_m=(20.0, 0.0, 20.0))

    with pytest.raises(ValueError, match="integer number of steps"):
        DoublePendulumTrialConfig(
            model_params=base.model_params,
            initial_state=base.initial_state,
            duration_s=0.035,
            dt_s=0.01,
            base_torques_nm=base.base_torques_nm,
            target_position_m=base.target_position_m,
            contact_radius_m=base.contact_radius_m,
            frame_id=base.frame_id,
            alignment_id=base.alignment_id,
        )

    with pytest.raises(ValueError, match="contact_radius_m"):
        DoublePendulumTrialConfig(
            model_params=base.model_params,
            initial_state=base.initial_state,
            duration_s=base.duration_s,
            dt_s=base.dt_s,
            base_torques_nm=base.base_torques_nm,
            target_position_m=base.target_position_m,
            contact_radius_m=0.0,
            frame_id=base.frame_id,
            alignment_id=base.alignment_id,
        )
