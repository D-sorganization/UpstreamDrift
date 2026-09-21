"""Native MuJoCo reference checks for allocation equations (#10439)."""

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    MujocoForceAdapter,
)

pytestmark = [pytest.mark.live_simulation, pytest.mark.requires_mujoco]


@pytest.fixture
def native_pair():
    mj = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    spec = (
        Path(__file__).resolve().parents[3]
        / "docs/development/full_body_models/full_body_spec_v1.json"
    ).read_bytes()
    adapter = MujocoForceAdapter(spec)
    xml, metadata = export_full_body_mjcf(spec)
    model = mj.MjModel.from_xml_string(xml)
    return adapter, mj, model, mj.MjData(model), metadata


def _reference(native_pair, q, v):
    _, mj, model, data, _ = native_pair
    data.qpos[:] = q
    data.qvel[:] = v
    mj.mj_fwdPosition(model, data)
    mj.mj_fwdVelocity(model, data)
    mass = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    return mass, data.qfrc_bias.copy()


def test_inverse_dynamics_is_raw_mass_acceleration_plus_bias(native_pair):
    adapter, _, model, _, _ = native_pair
    q = np.linspace(-0.02, 0.03, model.nq)
    v = np.linspace(-0.1, 0.1, model.nv)
    a = np.linspace(-0.2, 0.2, model.nv)
    mass, bias = _reference(native_pair, q, v)
    actual = adapter.compute_inverse_dynamics(q, v, a)
    np.testing.assert_allclose(actual, mass @ a + bias, atol=1e-9, rtol=1e-10)


def test_parity_refreshes_native_state_without_prior_inverse_call(native_pair):
    adapter, _, model, _, _ = native_pair
    q = np.linspace(-0.02, 0.03, model.nq)
    v = np.linspace(-0.1, 0.1, model.nv)
    a = np.linspace(-0.2, 0.2, model.nv)
    mass, bias = _reference(native_pair, q, v)
    assert adapter.verify_acceleration_parity(q, v, mass @ a + bias, a) < 1e-7


def test_fresh_contact_jacobian_matches_native_position_differences(native_pair):
    adapter, mj, model, data, metadata = native_pair
    q = np.linspace(-0.02, 0.03, model.nq)
    direction = np.random.default_rng(10439).normal(size=model.nv)
    ids = [model.site(name).id for name in metadata["contact_sites"].values()]
    jac = adapter.compute_contact_jacobian(q)
    positions = []
    step = 1e-6
    for sign in [-1, 1]:
        data.qpos[:] = q + sign * step * direction
        mj.mj_kinematics(model, data)
        positions.append(data.site_xpos[ids].copy().reshape(-1))
    finite_difference = (positions[1] - positions[0]) / (2 * step)
    np.testing.assert_allclose(jac @ direction, finite_difference, atol=1e-7, rtol=1e-6)


def test_wrong_coordinate_count_rejected_before_native_mutation(native_pair):
    adapter, _, model, _, _ = native_pair
    with pytest.raises(ValueError, match="configuration.*shape"):
        adapter.compute_inverse_dynamics(
            np.zeros(44), np.zeros(model.nv), np.zeros(model.nv)
        )


def test_nonfinite_native_input_rejected(native_pair):
    adapter, _, model, _, _ = native_pair
    q = np.zeros(model.nq)
    q[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.compute_contact_jacobian(q)
