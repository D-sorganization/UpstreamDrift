"""F02 closed-loop feedback on a native floating-root MuJoCo manifold."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_tangent_derivative import (
    linearize_native_tangent_step,
)
from src.shared.python.estimation.mosaic.local_policy import (
    LinearizedDynamics,
    tvlqr_gains,
)
from src.shared.python.motion_matching.distributed_feedback import NominalTrajectory
from tests.unit.motion_matching.test_native_tangent_derivative import (
    _floating_two_hinge_xml,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_STEPS = 12


def _providers() -> tuple[object, object]:
    mj = pytest.importorskip("mujoco")
    return mj, importlib.import_module(
        "src.engines.physics_engines.mujoco.python.native_distributed_feedback"
    )


def _nominal(mj: object, model: object) -> np.ndarray:
    data = mj.MjData(model)
    data.qpos[:3] = (0.1, -0.2, 0.8)
    data.qpos[3:7] = (np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8))
    state = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
    return state


def _schedule(mj: object, model: object, nominal: np.ndarray) -> NominalTrajectory:
    derivative = linearize_native_tangent_step(model, nominal, np.zeros(2))
    q = np.diag([1.0] * 6 + [30.0, 30.0] + [0.1] * 8)
    gains = tvlqr_gains(
        LinearizedDynamics(
            np.repeat(derivative.A[None], _STEPS, axis=0),
            np.repeat(derivative.B[None], _STEPS, axis=0),
        ),
        q,
        np.eye(2) * 0.01,
    )
    return NominalTrajectory(
        times=np.arange(_STEPS + 1) * model.opt.timestep,
        q=np.repeat(nominal[1 : 1 + model.nq][None], _STEPS, axis=0),
        v=np.zeros((_STEPS, model.nv)),
        feedforward=np.zeros((_STEPS, model.nu)),
        gains=gains,
        phase_names=("swing",) * _STEPS,
        channel_ids=derivative.ordered_input_channel_ids,
    )


def test_native_tangent_identifies_quaternion_sign_and_spd_mass() -> None:
    mj, module = _providers()
    model = mj.MjModel.from_xml_string(_floating_two_hinge_xml())
    nominal = _nominal(mj, model)
    tangent = module.NativeMuJoCoTangentModel(model, nominal)
    q = nominal[1 : 1 + model.nq].copy()
    sign_equivalent = q.copy()
    sign_equivalent[3:7] *= -1
    assert np.linalg.norm(q - sign_equivalent) > 1
    np.testing.assert_allclose(tangent.difference(q, sign_equivalent), 0, atol=1e-12)
    np.testing.assert_allclose(tangent.difference(sign_equivalent, q), 0, atol=1e-12)
    inverse_mass = tangent.mass_inverse(q)
    assert inverse_mass.shape == (8, 8)
    assert np.linalg.eigvalsh(inverse_mass).min() > 0
    with pytest.raises(ValueError, match="quaternion"):
        tangent.difference(q, np.r_[q[:3], q[3:7] * 1.1, q[7:]])


def test_native_f02_feedback_replays_full_state_and_beats_frozen_nominal(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    mj, module = _providers()
    path = tmp_path / "floating.xml"
    path.write_text(_floating_two_hinge_xml(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    nominal = _nominal(mj, model)
    schedule = _schedule(mj, model, nominal)
    perturbed = nominal.copy()
    perturbed[8] = 0.4
    results = {}
    for enabled in (True, False):
        results[enabled] = module.run_native_distributed_feedback_tracking(
            path,
            perturbed,
            schedule,
            max_torque_rate_nm_s=np.array([400.0, 400.0]),
            enabled=enabled,
            experiment_id=f"f02-native-{enabled}",
        )
    controlled = results[True]
    nominal_only = results[False]
    assert controlled.tracking.bundle.model == nominal_only.tracking.bundle.model
    for result in (controlled, nominal_only):
        np.testing.assert_allclose(
            result.tracking.native_integration_states,
            result.tracking.replay.integration_states,
            atol=1e-12,
            rtol=0,
        )
        assert len(result.control_steps) == _STEPS
        assert len(result.tracking.commands) == _STEPS
        assert (
            result.tracking.bundle.input_history.input_kind.value == "actuator_torque"
        )
        assert all(
            step.input_boundary == "actuator_torque" for step in result.control_steps
        )
    assert all(
        np.array_equal(step.applied, [0, 0]) for step in nominal_only.control_steps
    )
    assert np.abs(controlled.tracking.native_qpos[-1, 7]) < np.abs(
        nominal_only.tracking.native_qpos[-1, 7]
    )
    assert np.any(np.abs(controlled.tracking.replay.applied_actuator_torques) > 0)
    assert all(
        step.information_pattern == "exact_simulated_state"
        for step in controlled.control_steps
    )
    if os.environ.get("F02_NATIVE_RECEIPT") == "1":
        bundle = controlled.tracking.bundle
        record_property(
            "f02_native_evidence",
            json.dumps(
                {
                    "source_model_sha256": bundle.model.source_model_sha256,
                    "loaded_native_model_sha256": bundle.model.loaded_native_model_sha256,
                    "initial_state_sha256": bundle.integrity.initial_state_sha256,
                    "policy_sha256": bundle.policy_sha256,
                    "time_grid_sha256": bundle.time_grid_sha256,
                    "state_schema_sha256": bundle.state_schema_sha256,
                    "input_channel_schema_sha256": bundle.input_channel_schema_sha256,
                    "controlled_applied_input_sha256": bundle.applied_input_sha256,
                    "nominal_applied_input_sha256": nominal_only.tracking.bundle.applied_input_sha256,
                    "controlled_final_hip_error_rad": float(
                        abs(controlled.tracking.native_qpos[-1, 7])
                    ),
                    "nominal_final_hip_error_rad": float(
                        abs(nominal_only.tracking.native_qpos[-1, 7])
                    ),
                    "max_full_state_replay_error": float(
                        np.max(
                            np.abs(
                                controlled.tracking.native_integration_states
                                - controlled.tracking.replay.integration_states
                            )
                        )
                    ),
                },
                sort_keys=True,
            ),
        )


def test_native_f02_rejects_wrong_channels_and_contact(tmp_path: Path) -> None:
    mj, module = _providers()
    path = tmp_path / "floating.xml"
    path.write_text(_floating_two_hinge_xml(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    nominal = _nominal(mj, model)
    schedule = _schedule(mj, model, nominal)
    wrong = NominalTrajectory(
        schedule.times,
        schedule.q,
        schedule.v,
        schedule.feedforward,
        schedule.gains,
        schedule.phase_names,
        schedule.channel_ids[::-1],
    )
    with pytest.raises(ValueError, match="channel"):
        module.run_native_distributed_feedback_tracking(
            path,
            nominal,
            wrong,
            max_torque_rate_nm_s=np.array([400.0, 400.0]),
            experiment_id="wrong-channels",
        )
    contact = tmp_path / "contact.xml"
    contact.write_text(
        _floating_two_hinge_xml().replace(
            "<worldbody>",
            '<worldbody><geom name="ground" type="plane" size="1 1 0.1"/>',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contact-free"):
        module.run_native_distributed_feedback_tracking(
            contact,
            nominal,
            schedule,
            max_torque_rate_nm_s=np.array([400.0, 400.0]),
            experiment_id="contact-rejected",
        )


def test_native_f02_stores_post_limit_torques_not_requested_effort(
    tmp_path: Path,
) -> None:
    mj, module = _providers()
    path = tmp_path / "floating.xml"
    path.write_text(_floating_two_hinge_xml(), encoding="utf-8")
    model = mj.MjModel.from_xml_path(str(path))
    initial = _nominal(mj, model)
    base = _schedule(mj, model, initial)
    schedule = NominalTrajectory(
        base.times,
        base.q,
        base.v,
        np.repeat(np.array([[4.0, -4.0]]), _STEPS, axis=0),
        base.gains,
        base.phase_names,
        base.channel_ids,
    )
    result = module.run_native_distributed_feedback_tracking(
        path,
        initial,
        schedule,
        max_torque_rate_nm_s=np.array([400.0, 400.0]),
        enabled=False,
        experiment_id="post-limit-only",
    )
    assert np.array_equal(result.control_steps[0].total_requested, [4.0, -4.0])
    assert np.array_equal(result.control_steps[0].applied, [2.0, -1.5])
    assert np.array_equal(
        result.tracking.replay.applied_actuator_torques[0], [2.0, -1.5]
    )
    np.testing.assert_allclose(
        result.tracking.native_integration_states,
        result.tracking.replay.integration_states,
        atol=1e-12,
        rtol=0,
    )
