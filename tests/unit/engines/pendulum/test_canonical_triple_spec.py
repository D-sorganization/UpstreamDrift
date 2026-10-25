from __future__ import annotations

import numpy as np
import pytest
from double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumState,
)

from src.engines.physics_engines.mujoco.pendulum_models_xml import TRIPLE_PENDULUM_XML
from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.pendulum_simulator.physics_triple import TriplePendulumParams
from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    POINT_MASS_TRIPLE_PENDULUM_SPEC,
    TRIPLE_PENDULUM_SPEC,
    UNDAMPED_POINT_MASS_TRIPLE_PENDULUM_SPEC,
    TriplePendulumSegmentSpec,
    TriplePendulumSpec,
)

pytestmark = pytest.mark.unit


def test_canonical_spec_drives_models_and_mujoco_xml() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.models_pendulum import (
        TRIPLE_PENDULUM_XML as COMPAT_TRIPLE_PENDULUM_XML,
    )

    assert TRIPLE_PENDULUM_SPEC.gravity_m_s2 == float(GRAVITY_M_S2)
    assert COMPAT_TRIPLE_PENDULUM_XML is TRIPLE_PENDULUM_XML
    assert f'gravity="0 0 -{TRIPLE_PENDULUM_SPEC.gravity_m_s2}"' in TRIPLE_PENDULUM_XML
    assert TRIPLE_PENDULUM_XML.count('axis="0 1 0"') == 3
    for segment in TRIPLE_PENDULUM_SPEC.segments:
        assert f'mass="{segment.mass_kg}"' in TRIPLE_PENDULUM_XML


@pytest.mark.parametrize(
    ("field", "value"),
    [("length_m", 0.0), ("mass_kg", 0.0), ("inertia_about_com", 0.0)],
)
def test_segment_spec_rejects_nonpositive_physics(field: str, value: float) -> None:
    fields = {
        "length_m": 1.0,
        "mass_kg": 1.0,
        "center_of_mass_ratio": 0.5,
        "inertia_about_com": 0.1,
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        TriplePendulumSegmentSpec(**fields)


def test_spec_rejects_negative_damping_and_unknown_gravity_plane() -> None:
    with pytest.raises(ValueError, match="damping"):
        TriplePendulumSpec(TRIPLE_PENDULUM_SPEC.segments, (0.1, -0.1, 0.1))
    with pytest.raises(ValueError, match="gravity plane"):
        TriplePendulumSpec(
            TRIPLE_PENDULUM_SPEC.segments,
            TRIPLE_PENDULUM_SPEC.damping_nm_s_rad,
            gravity_plane="unknown",
        )


def test_xml_gravity_generates_joint_bias_torque() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(TRIPLE_PENDULUM_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = (0.3, -0.2, 0.15)
    mujoco.mj_forward(model, data)
    assert data.qfrc_bias.shape == (3,)
    assert max(abs(data.qfrc_bias)) > 1e-6


def test_engine_defaults_are_built_from_named_specs() -> None:
    distributed = TriplePendulumDynamics().parameters
    assert tuple(s.length_m for s in distributed.segments) == tuple(
        s.length_m for s in TRIPLE_PENDULUM_SPEC.segments
    )
    assert distributed.damping == TRIPLE_PENDULUM_SPEC.damping_nm_s_rad

    point_mass = TriplePendulumParams.from_point_mass_spec()
    assert (
        point_mass.b1,
        point_mass.b2,
        point_mass.b3,
    ) == POINT_MASS_TRIPLE_PENDULUM_SPEC.damping_nm_s_rad
    assert UNDAMPED_POINT_MASS_TRIPLE_PENDULUM_SPEC.damping_nm_s_rad == (0.0, 0.0, 0.0)
    undamped = TriplePendulumParams.from_point_mass_spec(
        UNDAMPED_POINT_MASS_TRIPLE_PENDULUM_SPEC
    )
    assert (undamped.b1, undamped.b2, undamped.b3) == (0.0, 0.0, 0.0)


def test_distributed_model_mass_and_gravity_match_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    dynamics = TriplePendulumDynamics()
    model = mujoco.MjModel.from_xml_string(TRIPLE_PENDULUM_XML)
    data = mujoco.MjData(model)
    full_mass = np.empty((3, 3))
    for angles, velocities in (
        ((0.3, -0.2, 0.15), (0.2, -0.1, 0.05)),
        ((-0.4, 0.25, -0.1), (-0.15, 0.12, -0.08)),
        ((0.1, 0.5, -0.35), (0.05, 0.18, -0.2)),
    ):
        data.qpos[:] = angles
        data.qvel[:] = velocities
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, full_mass, data.qM)
        state = TriplePendulumState(*angles, *velocities)
        np.testing.assert_allclose(full_mass, dynamics.mass_matrix(state), rtol=1e-9)
        np.testing.assert_allclose(
            data.qfrc_bias - data.qfrc_passive,
            dynamics.bias_vector(state),
            rtol=1e-9,
            atol=1e-12,
        )


def test_mujoco_and_in_house_rk4_replay_same_torque_profile() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(TRIPLE_PENDULUM_XML)
    data = mujoco.MjData(model)
    dynamics = TriplePendulumDynamics()
    state = TriplePendulumState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    torques = ((0.2, -0.1, 0.05),) * 10 + ((-0.1, 0.15, 0.08),) * 10
    for index, control in enumerate(torques):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        state = dynamics.step(
            index * model.opt.timestep, state, model.opt.timestep, control
        )

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head")
    assert geom_id >= 0
    expected_tip = data.geom_xpos[geom_id]
    angles = (
        state.theta1,
        state.theta1 + state.theta2,
        state.theta1 + state.theta2 + state.theta3,
    )
    lengths = tuple(segment.length_m for segment in dynamics.parameters.segments)
    tip_x = -sum(seg_l * np.sin(a) for seg_l, a in zip(lengths, angles, strict=True))
    tip_z = 1.2 - sum(
        seg_l * np.cos(a) for seg_l, a in zip(lengths, angles, strict=True)
    )
    actual_tip = np.array([tip_x, 0.0, tip_z])
    assert np.linalg.norm(expected_tip - actual_tip) <= 1e-6
