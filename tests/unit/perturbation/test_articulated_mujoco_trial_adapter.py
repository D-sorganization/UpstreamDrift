"""Tests for canonical variation rows on the articulated MuJoCo model."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python import perturbation
from src.shared.python.perturbation.articulated_mujoco_trial_adapter import (
    ArticulatedMujocoTrialAdapter,
    ArticulatedMujocoTrialConfig,
    MujocoVariationBinding,
    NamedJointTorque,
)

pytestmark = pytest.mark.unit

_PLAN_SHA = "a" * 64
_SCENARIO_SHA = "d" * 64
_TOOLS_REVISION = "f9730033fd279ba8b4abe03bab2aadd950400b47"
_ENGINE_REVISION = "c" * 40
_SHOULDER_TORQUE = "swing_sim.swing.shoulder_commanded_torque_offset_nm"
_SHOULDER_DAMPING = "swing_sim.swing.damping_shoulder"


@dataclass(frozen=True)
class _Spec:
    variable_key: str
    time_window_s: tuple[float, float] | None = None
    point_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Plan:
    noise: tuple[_Spec, ...]
    n_runs: int = 1
    seed: int = 31
    mode: str = "swing"


def _upper_body_xml() -> str:
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "engines"
        / "physics_engines"
        / "mujoco"
        / "_golf_swing_upper_body_xml.py"
    )
    spec = importlib.util.spec_from_file_location("_upper_body_xml_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UPPER_BODY_GOLF_SWING_XML


def _upper_body_xml_with_ball_at_clubhead() -> str:
    xml = _upper_body_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    clubhead_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "clubhead")
    position = " ".join(f"{value:.17g}" for value in data.xpos[clubhead_id])
    return xml.replace(
        '<body name="ball" pos="0 0.1 0.02">',
        f'<body name="ball" pos="{position}">',
    )


_COORDINATE_JOINTS = (
    "spine_rotation",
    "left_shoulder_swing",
    "left_shoulder_lift",
    "left_elbow",
    "left_wrist",
    "right_shoulder_swing",
    "right_shoulder_lift",
    "right_elbow",
    "right_wrist",
    "club_wrist",
)


def _config(
    binding: MujocoVariationBinding,
) -> ArticulatedMujocoTrialConfig:
    return ArticulatedMujocoTrialConfig(
        model_xml=_upper_body_xml(),
        model_id="bilateral-upper-body-welded-club/v1",
        duration_s=0.01,
        dt_s=0.002,
        coordinate_joint_names=_COORDINATE_JOINTS,
        marker_body_names=("left_hand", "right_hand", "clubhead", "ball"),
        source_body_name="clubhead",
        target_body_name="ball",
        base_joint_torques=(NamedJointTorque("spine_rotation", 1.0),),
        bindings=(binding,),
        frame_id="mujoco-world:x-forward-y-left-z-up",
        alignment_id="address-state/v1",
    )


def _adapter(
    plan: _Plan,
    config: ArticulatedMujocoTrialConfig,
) -> ArticulatedMujocoTrialAdapter:
    return ArticulatedMujocoTrialAdapter(
        plan=plan,
        config=config,
        plan_sha256=_PLAN_SHA,
        scenario_sha256=_SCENARIO_SHA,
        tools_revision=_TOOLS_REVISION,
        engine_revision=_ENGINE_REVISION,
    )


def test_articulated_adapter_is_exposed_by_public_package() -> None:
    assert perturbation.ArticulatedMujocoTrialAdapter is ArticulatedMujocoTrialAdapter
    assert perturbation.MujocoVariationBinding is MujocoVariationBinding


def test_actual_bilateral_upper_body_model_runs_and_retains_geometry() -> None:
    plan = _Plan(noise=(_Spec(_SHOULDER_TORQUE, (0.0, 0.006), ("joint.shoulder",)),))
    binding = MujocoVariationBinding(
        variable_key=_SHOULDER_TORQUE,
        unit="N·m",
        kind="joint_torque_offset",
        target_joint_names=("left_shoulder_swing", "right_shoulder_swing"),
        allocation_weights=(0.5, 0.5),
        plan_point_ids=("joint.shoulder",),
    )
    adapter = _adapter(plan, _config(binding))

    result = adapter.run(np.array([8.0]))

    assert result.trace.backend == "mujoco-articulated"
    assert result.trace.q.shape == (6, 10)
    assert result.trace.v.shape == (6, 10)
    assert result.trace.markers is not None
    assert result.trace.markers.shape == (6, 4, 3)
    assert result.trace.u is not None
    np.testing.assert_allclose(result.trace.u[:3, [1, 5]], 4.0)
    np.testing.assert_allclose(result.trace.u[3:, [1, 5]], 0.0)

    evidence = adapter.collect_success(0, plan.seed, np.array([8.0]), result)
    assert evidence.model_id == "bilateral-upper-body-welded-club/v1"
    assert evidence.outcome == "no_impact"
    assert evidence.trace is not None
    assert evidence.trace.coordinate_ids == tuple(
        f"joint.{name}" for name in _COORDINATE_JOINTS
    )
    assert evidence.trace.marker_ids == (
        "body.left_hand",
        "body.right_hand",
        "body.clubhead",
        "body.ball",
    )
    assert evidence.scenario_sha256 == _SCENARIO_SHA
    assert len(evidence.execution_config_sha256) == 64


def test_push_pull_allocation_is_explicit_and_changes_the_real_trajectory() -> None:
    plan = _Plan(noise=(_Spec(_SHOULDER_TORQUE, (0.0, 0.01), ("joint.shoulder",)),))
    binding = MujocoVariationBinding(
        variable_key=_SHOULDER_TORQUE,
        unit="N·m",
        kind="joint_torque_offset",
        target_joint_names=("left_shoulder_swing", "right_shoulder_swing"),
        allocation_weights=(1.0, -1.0),
        plan_point_ids=("joint.shoulder",),
    )
    adapter = _adapter(plan, _config(binding))

    positive = adapter.run(np.array([6.0]))
    reversed_drive = adapter.run(np.array([-6.0]))

    assert positive.trace.u is not None
    np.testing.assert_allclose(positive.trace.u[0, [1, 5]], [6.0, -6.0])
    assert not np.allclose(positive.trace.q, reversed_drive.trace.q)


def test_hit_requires_actual_source_target_geom_contact() -> None:
    plan = _Plan(noise=(_Spec(_SHOULDER_DAMPING),))
    binding = MujocoVariationBinding(
        variable_key=_SHOULDER_DAMPING,
        unit="N·m·s",
        kind="joint_damping",
        target_joint_names=("left_shoulder_swing", "right_shoulder_swing"),
        allocation_weights=(1.0, 1.0),
        plan_point_ids=(),
    )
    config = _config(binding)
    adapter = _adapter(
        plan, replace(config, model_xml=_upper_body_xml_with_ball_at_clubhead())
    )

    result = adapter.run(np.array([1.5]))
    evidence = adapter.collect_success(0, plan.seed, np.array([1.5]), result)

    assert result.first_contact_index == 0
    assert evidence.outcome == "hit"
    assert evidence.impact is not None
    assert evidence.impact.time_s == 0.0
    assert evidence.closest_approach is not None
    assert evidence.closest_approach.contact_observed is True


def test_global_damping_binding_changes_both_declared_shoulder_dofs() -> None:
    plan = _Plan(noise=(_Spec(_SHOULDER_DAMPING),))
    binding = MujocoVariationBinding(
        variable_key=_SHOULDER_DAMPING,
        unit="N·m·s",
        kind="joint_damping",
        target_joint_names=("left_shoulder_swing", "right_shoulder_swing"),
        allocation_weights=(1.0, 1.0),
        plan_point_ids=(),
    )
    adapter = _adapter(plan, _config(binding))

    low = adapter.run(np.array([0.1]))
    high = adapter.run(np.array([4.0]))

    assert not np.allclose(low.trace.v, high.trace.v)


@pytest.mark.parametrize(
    ("spec", "binding", "message"),
    [
        (
            _Spec(_SHOULDER_TORQUE),
            MujocoVariationBinding(
                _SHOULDER_TORQUE,
                "N·m",
                "joint_torque_offset",
                ("left_shoulder_swing",),
                (1.0,),
                ("joint.shoulder",),
            ),
            "time window",
        ),
        (
            _Spec(_SHOULDER_TORQUE, (0.0, 0.006), ("joint.wrist",)),
            MujocoVariationBinding(
                _SHOULDER_TORQUE,
                "N·m",
                "joint_torque_offset",
                ("left_shoulder_swing",),
                (1.0,),
                ("joint.shoulder",),
            ),
            "point",
        ),
        (
            _Spec(_SHOULDER_TORQUE, (0.0, 0.006), ("joint.shoulder",)),
            MujocoVariationBinding(
                _SHOULDER_TORQUE,
                "N·m",
                "joint_torque_offset",
                ("not_a_joint",),
                (1.0,),
                ("joint.shoulder",),
            ),
            "not found",
        ),
    ],
)
def test_adapter_rejects_missing_locus_semantics_or_model_topology(
    spec: _Spec,
    binding: MujocoVariationBinding,
    message: str,
) -> None:
    plan = _Plan(noise=(spec,))

    with pytest.raises(ValueError, match=message):
        _adapter(plan, _config(binding))
