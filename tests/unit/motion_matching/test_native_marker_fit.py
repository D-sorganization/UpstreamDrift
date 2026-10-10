"""F03 native marker fitting through the existing BoxFDDP and T01 replay."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pytest

from tests.unit.motion_matching.test_native_tangent_derivative import (
    _floating_two_hinge_xml,
    _initial,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _site_fixture() -> str:
    return (
        _floating_two_hinge_xml()
        .replace(
            '<body name="upper" pos="0.2 0 0">',
            '<body name="upper" pos="0.2 0 0">'
            '<site name="upper_marker" pos="0.24 0 0" size="0.005"/>',
        )
        .replace(
            '<body name="lower" pos="0.3 0 0">',
            '<body name="lower" pos="0.3 0 0">'
            '<site name="lower_marker" pos="0.22 0 0" size="0.005"/>',
        )
    )


def _positions(mj: object, model: object, qpos: np.ndarray) -> np.ndarray:
    data = mj.MjData(model)
    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    return data.site_xpos.copy()


def test_native_marker_cost_has_correct_tangent_gradient_and_exact_clock() -> None:
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("crocoddyl")
    from src.engines.physics_engines.mujoco.python.native_marker_fit import (
        NativeMarkerObservationCost,
        NativeMarkerTargets,
    )

    model = mj.MjModel.from_xml_string(_site_fixture())
    initial = _initial(model)
    state = np.r_[
        initial[1 : 1 + model.nq], initial[1 + model.nq : 1 + model.nq + model.nv]
    ]
    observed = _positions(mj, model, state[: model.nq])
    target = observed.copy()
    target[0, 0] += 0.01
    samples = np.stack((observed, target))
    mask = np.array([[True, True], [True, False]])
    targets = NativeMarkerTargets(
        time_seconds=np.array([0.0, model.opt.timestep]),
        site_names=("upper_marker", "lower_marker"),
        position_m=samples,
        valid_mask=mask,
        site_weights=np.array([100.0, 200.0]),
        frame_id="world",
    )
    cost = NativeMarkerObservationCost(model, targets)
    value = cost.cost(state, 1)
    gradient, hessian = cost.tangent(state, 1)
    assert value > 0 and gradient.shape == (16,) and hessian.shape == (16, 16)
    from src.engines.physics_engines.mujoco.python.native_manifold_box_fddp import (
        NativeManifoldState,
    )

    chart = NativeManifoldState(model)
    numerical_site_jacobian = np.zeros((3, 16))
    for column in range(16):
        bump = np.eye(16)[column] * 1e-6
        plus_state = chart.integrate(state, bump)
        minus_state = chart.integrate(state, -bump)
        numerical_site_jacobian[:, column] = (
            _positions(mj, model, plus_state[: model.nq])[0]
            - _positions(mj, model, minus_state[: model.nq])[0]
        ) / 2e-6
        finite = (cost.cost(plus_state, 1) - cost.cost(minus_state, 1)) / 2e-6
        assert np.isclose(gradient[column], finite, atol=1e-5)
    np.testing.assert_allclose(
        hessian,
        2
        * targets.site_weights[0]
        * numerical_site_jacobian.T
        @ numerical_site_jacobian,
        atol=1e-5,
        rtol=1e-5,
    )
    assert np.linalg.eigvalsh(hessian).min() >= -1e-9
    from src.engines.physics_engines.mujoco.python.native_manifold_box_fddp import (
        NativeManifoldAction,
    )

    action = NativeManifoldAction(
        chart,
        state,
        np.full(16, 1e-4),
        np.full(2, 1e-4),
        cost,
        1,
    )
    command = np.array([0.2, -0.1])
    data = action.createData()
    action.calc(data, state, command)
    action.calcDiff(data, state, command)
    plus = action.createData()
    minus = action.createData()
    for column in (3, 7):  # floating-root quaternion rotation, then hinge
        direction = np.eye(16)[column] * 1e-6
        action.calc(plus, chart.integrate(state, direction), command)
        action.calc(minus, chart.integrate(state, -direction), command)
        assert np.isclose(data.Lx[column], (plus.cost - minus.cost) / 2e-6, atol=1e-5)
    step = np.array([1e-6, 0.0])
    action.calc(plus, state, command + step)
    action.calc(minus, state, command - step)
    assert np.isclose(data.Lu[0], (plus.cost - minus.cost) / 2e-6, atol=1e-5)
    wrong_clock = NativeMarkerTargets(
        time_seconds=np.array([0.0, model.opt.timestep * 1.1]),
        site_names=targets.site_names,
        position_m=samples,
        valid_mask=mask,
        site_weights=targets.site_weights,
        frame_id="world",
    )
    with pytest.raises(ValueError, match="native grid"):
        NativeMarkerObservationCost(model, wrong_clock)
    with pytest.raises(ValueError, match="visibility"):
        NativeMarkerTargets(
            targets.time_seconds,
            targets.site_names,
            targets.position_m,
            np.zeros_like(mask),
            targets.site_weights,
            "world",
        )
    one_joint = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="hinge" type="hinge"/>'
        '<geom type="sphere" size="0.1"/><site name="upper_marker"/></body>'
        '</worldbody><actuator><motor joint="hinge"/></actuator></mujoco>'
    )
    with pytest.raises(ValueError, match="nq=9/nv=8/nu=2"):
        NativeMarkerObservationCost(one_joint, targets)


def test_native_marker_fit_improves_reachable_observations_and_replays(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("crocoddyl")
    from src.engines.physics_engines.mujoco.python.native_manifold_box_fddp import (
        NativeManifoldBoxFDDP,
        run_native_manifold_box_fddp_tracking,
    )
    from src.engines.physics_engines.mujoco.python.native_marker_fit import (
        NativeMarkerObservationCost,
        NativeMarkerTargets,
    )
    from src.engines.physics_engines.mujoco.python.native_torque_replay import (
        build_native_torque_bundle,
        replay_native_torque_bundle,
    )

    path = tmp_path / "marker-fixture.xml"
    path.write_text(_site_fixture(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    initial = _initial(model)
    steps, horizon = 4, 3
    times = np.arange(steps + horizon + 1) * model.opt.timestep
    teacher_input = np.repeat([[0.75, -0.5]], len(times), axis=0)
    teacher = replay_native_torque_bundle(
        build_native_torque_bundle(
            path, initial, times, teacher_input, experiment_id="marker-teacher"
        ),
        path,
    )
    observations = np.stack([_positions(mj, model, q) for q in teacher.qpos])
    mask = np.ones(observations.shape[:2], dtype=bool)
    mask[2, 0] = False
    fit_wall_started = time.perf_counter()
    fit_cpu_started = time.process_time()
    targets = NativeMarkerTargets(
        time_seconds=times,
        site_names=("upper_marker", "lower_marker"),
        position_m=observations,
        valid_mask=mask,
        site_weights=np.array([1e4, 1e4]),
        frame_id="world",
    )
    cost = NativeMarkerObservationCost(model, targets)
    physical = np.r_[teacher.qpos[0], teacher.qvel[0]]
    references = np.repeat(physical[None], len(times), axis=0)
    controller = NativeManifoldBoxFDDP(
        model,
        references,
        horizon_steps=horizon,
        max_iterations=12,
        max_wall_s=5.0,
        observation_cost=cost,
        state_weights=np.full(16, 1e-8),
        input_weights=np.full(2, 1e-5),
    )
    fitted = run_native_manifold_box_fddp_tracking(
        path, initial, controller, steps=steps, experiment_id="marker-fitted"
    )
    zero_torque = np.zeros((steps + 1, model.nu))
    baseline = replay_native_torque_bundle(
        build_native_torque_bundle(
            path,
            initial,
            times[: steps + 1],
            zero_torque,
            experiment_id="marker-zero-baseline",
        ),
        path,
    )
    selected = mask[: steps + 1]
    fitted_xyz = np.stack([_positions(mj, model, q) for q in fitted.native_qpos])
    baseline_xyz = np.stack([_positions(mj, model, q) for q in baseline.qpos])
    fit_rmse = np.sqrt(
        np.mean((fitted_xyz[selected] - observations[: steps + 1][selected]) ** 2)
    )
    zero_rmse = np.sqrt(
        np.mean((baseline_xyz[selected] - observations[: steps + 1][selected]) ** 2)
    )
    assert fit_rmse < zero_rmse
    statuses = tuple(receipt.status for receipt in fitted.commands)
    accepted = statuses.count("optimized")
    assert accepted > 0
    np.testing.assert_allclose(
        fitted.native_integration_states,
        fitted.replay.integration_states,
        atol=1e-12,
        rtol=0,
    )
    # Charge baseline comparison, score validation and independent replay too.
    fit_total_wall_s = time.perf_counter() - fit_wall_started
    fit_total_cpu_s = time.process_time() - fit_cpu_started
    if os.environ.get("F03_NATIVE_RECEIPT") == "1":
        bundle = fitted.bundle
        record_property(
            "f03_marker_fit_evidence",
            json.dumps(
                {
                    "observation_sha256": targets.identity_sha256,
                    "source_model_sha256": bundle.model.source_model_sha256,
                    "loaded_native_model_sha256": bundle.model.loaded_native_model_sha256,
                    "initial_state_sha256": bundle.integrity.initial_state_sha256,
                    "policy_sha256": bundle.policy_sha256,
                    "time_grid_sha256": bundle.time_grid_sha256,
                    "applied_input_sha256": bundle.applied_input_sha256,
                    "masked_site_samples": int((~mask).sum()),
                    "observed_site_samples": int(mask.sum()),
                    "accepted_commands": accepted,
                    "steps": steps,
                    "full_trajectory_accepted": accepted == steps,
                    "statuses": statuses,
                    "fit_marker_rmse_m": float(fit_rmse),
                    "zero_marker_rmse_m": float(zero_rmse),
                    "max_full_state_replay_error": float(
                        np.max(
                            np.abs(
                                fitted.native_integration_states
                                - fitted.replay.integration_states
                            )
                        )
                    ),
                    "total_wall_s": fit_total_wall_s,
                    "total_cpu_s": fit_total_cpu_s,
                },
                sort_keys=True,
            ),
        )
