"""Tests for MuJoCo/MJX inertial regressor provider (MOSAIC-03)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

# Import the module to be implemented
from src.shared.python.estimation.mosaic.mujoco_regressor import (
    MujocoRegressorModel,
    is_mujoco_available,
    matrix_to_quat,
    model_to_spatial_parameters,
    mujoco_to_spatial_parameters,
    quat_to_matrix,
    spatial_parameters_to_pseudo_inertia,
    spatial_to_mujoco_parameters,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def _build_double_pendulum_xml(
    l1: float = 0.9,
    l2: float = 0.6,
    m1: float = 1.3,
    m2: float = 0.7,
    lc1: float = 0.45,
    lc2: float = 0.25,
    i1c: float = 0.08,
    i2c: float = 0.02,
) -> str:
    """Build MJCF XML for a 2-link planar pendulum matching PlanarChain conventions."""
    return f"""
<mujoco model="planar_double_pendulum">
  <option gravity="0 -9.81 0"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" pos="0 0 0" axis="0 0 1"/>
      <geom name="geom1" type="sphere" size="0.01" mass="{m1}" pos="{lc1} 0 0"/>
      <body name="link2" pos="{l1} 0 0">
        <joint name="joint2" type="hinge" pos="0 0 0" axis="0 0 1"/>
        <geom name="geom2" type="sphere" size="0.01" mass="{m2}" pos="{lc2} 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_quaternion_matrix_roundtrip() -> None:
    rng = np.random.default_rng(42)
    for _ in range(10):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rng.uniform(-np.pi, np.pi))
        w = np.cos(angle / 2.0)
        xyz = np.sin(angle / 2.0) * axis
        q_orig = np.array([w, xyz[0], xyz[1], xyz[2]])
        q_orig /= np.linalg.norm(q_orig)

        R = quat_to_matrix(q_orig)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)

        q_recovered = matrix_to_quat(R)
        # Check quaternion hemisphere sign equivalence: q == q_recovered or q == -q_recovered
        sign = 1.0 if np.dot(q_orig, q_recovered) > 0 else -1.0
        assert np.allclose(q_orig, sign * q_recovered, atol=1e-12)


def test_parameter_mapping_roundtrip() -> None:
    rng = np.random.default_rng(123)
    for _ in range(10):
        mass = float(rng.uniform(0.5, 5.0))
        ipos = rng.uniform(-0.3, 0.3, size=3)
        # Principal moments satisfying triangle inequality
        a, b, c = sorted(rng.uniform(0.01, 0.1, size=3))
        principal_inertia = np.array([b + c, a + c, a + b])

        # Random orientation
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rng.uniform(-np.pi, np.pi))
        iquat = np.array(
            [
                np.cos(angle / 2.0),
                *(np.sin(angle / 2.0) * axis),
            ]
        )
        iquat /= np.linalg.norm(iquat)

        # MuJoCo -> Spatial
        pi = mujoco_to_spatial_parameters(mass, ipos, principal_inertia, iquat)
        assert pi.shape == (10,)
        assert pi[0] == mass
        assert np.allclose(pi[1:4], mass * ipos, atol=1e-12)

        # Spatial -> MuJoCo
        m_rec, ipos_rec, inertia_rec, iquat_rec = spatial_to_mujoco_parameters(pi)
        assert np.isclose(m_rec, mass, atol=1e-12)
        assert np.allclose(ipos_rec, ipos, atol=1e-12)

        # Re-map to spatial
        pi_rec = mujoco_to_spatial_parameters(m_rec, ipos_rec, inertia_rec, iquat_rec)
        assert np.allclose(pi, pi_rec, atol=1e-12)


def test_spatial_to_pseudo_inertia_roundtrip() -> None:
    mass = 2.0
    ipos = np.array([0.1, -0.05, 0.2])
    principal_inertia = np.array([0.05, 0.06, 0.08])
    iquat = np.array([1.0, 0.0, 0.0, 0.0])

    pi = mujoco_to_spatial_parameters(mass, ipos, principal_inertia, iquat)
    pseudo = spatial_parameters_to_pseudo_inertia(pi)
    assert pseudo.shape == (4, 4)
    assert np.allclose(pseudo, pseudo.T, atol=1e-12)

    # All eigenvalues should be positive for consistent bodies
    eigvals = np.linalg.eigvalsh(pseudo)
    assert np.all(eigvals > 0)


def test_regression_torque_reproduction() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo is not available")

    xml = _build_double_pendulum_xml()
    model = MujocoRegressorModel.from_xml_string(xml)

    rng = np.random.default_rng(2024)
    for _ in range(5):
        q = rng.uniform(-2.0, 2.0, size=2)
        v = rng.normal(scale=1.5, size=2)
        a = rng.normal(scale=2.0, size=2)

        Y = model.compute_regressor(q, v, a)
        assert Y.shape == (2, 20)  # 2 DOFs, 2 moving bodies * 10 parameters

        pi = model.spatial_parameters
        assert pi.shape == (20,)

        tau_reconstructed = Y @ pi
        tau_direct = model.inverse_dynamics(q, v, a)

        # Regressor multiplied by parameters must match mj_inverse to numerical accuracy
        assert np.allclose(tau_reconstructed, tau_direct, atol=1e-7, rtol=1e-7)


def test_planar_oracle_conformance() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo is not available")

    l1, l2 = 0.9, 0.6
    m1, m2 = 1.3, 0.7
    lc1, lc2 = 0.45, 0.25
    i1c, i2c = 0.08, 0.02

    chain = PlanarChain(
        link_lengths=np.array([l1, l2]), actuated=np.array([True, True])
    )
    pi_planar = np.array(
        [
            m1,
            m1 * lc1,
            0.0,
            i1c + m1 * lc1**2,
            m2,
            m2 * lc2,
            0.0,
            i2c + m2 * lc2**2,
        ]
    )

    xml = _build_double_pendulum_xml(
        l1=l1, l2=l2, m1=m1, m2=m2, lc1=lc1, lc2=lc2, i1c=i1c, i2c=i2c
    )
    model = MujocoRegressorModel.from_xml_string(xml, planar=True)

    # Override exact inertia values on bodies 1 and 2
    model.set_body_parameters(
        1,
        mass=m1,
        ipos=np.array([lc1, 0.0, 0.0]),
        inertia=np.array([0.01, 0.01, i1c]),
        iquat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    model.set_body_parameters(
        2,
        mass=m2,
        ipos=np.array([lc2, 0.0, 0.0]),
        inertia=np.array([0.01, 0.01, i2c]),
        iquat=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    rng = np.random.default_rng(999)
    for _ in range(5):
        q = rng.uniform(-2.0, 2.0, size=2)
        v = rng.normal(scale=1.5, size=2)
        a = rng.normal(scale=2.0, size=2)

        # PlanarChain regressor
        Y_planar = chain.regressor(q[None, :], v[None, :], a[None, :])[0]
        tau_planar = Y_planar @ pi_planar

        # MuJoCo inverse dynamics
        tau_mujoco = model.inverse_dynamics(q, v, a)
        assert np.allclose(tau_planar, tau_mujoco, atol=1e-12)

        # MuJoCo planar regressor
        Y_mujoco_planar = model.compute_regressor(q, v, a)
        assert Y_mujoco_planar.shape == (2, 8)
        assert np.allclose(Y_planar, Y_mujoco_planar, atol=1e-7)


def test_batch_compute_regressor() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo is not available")

    xml = _build_double_pendulum_xml()
    model = MujocoRegressorModel.from_xml_string(xml)

    N = 10
    rng = np.random.default_rng(42)
    q_batch = rng.uniform(-1.5, 1.5, size=(N, 2))
    v_batch = rng.normal(size=(N, 2))
    a_batch = rng.normal(size=(N, 2))

    # Batched regressor
    Y_batch = model.batch_compute_regressor(q_batch, v_batch, a_batch)
    assert Y_batch.shape == (N, 2, 20)

    # Node-by-node equivalence
    for i in range(N):
        Y_single = model.compute_regressor(q_batch[i], v_batch[i], a_batch[i])
        assert np.allclose(Y_batch[i], Y_single, atol=1e-12)

    # regressor() method should accept batched input
    Y_from_regressor = model.regressor(q_batch, v_batch, a_batch)
    assert np.allclose(Y_batch, Y_from_regressor, atol=1e-12)


def test_regressor_model_protocol_properties() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo is not available")

    xml = _build_double_pendulum_xml()
    model = MujocoRegressorModel.from_xml_string(xml)

    assert model.n_dof == 2
    assert model.n_parameters == 20
    assert model.input_matrix.shape == (2, 2)
    assert np.allclose(model.input_matrix, np.eye(2))

    # Planar mode
    model_planar = MujocoRegressorModel.from_xml_string(xml, planar=True)
    assert model_planar.n_dof == 2
    assert model_planar.n_parameters == 8


def test_preconditions_enforced() -> None:
    if not is_mujoco_available():
        pytest.skip("MuJoCo is not available")

    xml = _build_double_pendulum_xml()
    model = MujocoRegressorModel.from_xml_string(xml)

    # Wrong shape
    with pytest.raises(ContractViolationError):
        model.compute_regressor(np.zeros(3), np.zeros(2), np.zeros(2))

    # Non-finite values
    with pytest.raises(ContractViolationError):
        model.compute_regressor(np.array([np.nan, 0.0]), np.zeros(2), np.zeros(2))

    # Wrong parameters in mapping
    with pytest.raises(ContractViolationError):
        mujoco_to_spatial_parameters(
            -1.0, np.zeros(3), np.ones(3), np.array([1, 0, 0, 0])
        )
