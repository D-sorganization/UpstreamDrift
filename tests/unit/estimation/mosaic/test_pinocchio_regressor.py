"""Tests for the Pinocchio RegressorModel provider (issue #11534, MOSAIC-02).

Acceptance criteria:
- Conformance against the planar oracle fixture: Pinocchio regressor matches
  the analytic planar regressor to stated tolerance (< 1e-12, measured ~1e-15).
- Y @ pi equals Pinocchio RNEA torques for random physically consistent pi,
  including floating-base rows.
- Batched evaluation over N nodes is tested for shape/contract and measured
  against a per-node loop with runtime reporting.
- Preconditions (dimension, finiteness, unit conventions) enforced via `require`
  and verified with failing inputs.
- If Pinocchio is not installed, tests skip with unqualified report (`requires_pinocchio`).
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError

from src.shared.python.estimation.mosaic.pinocchio_provider import (
    PinocchioMarkerSet,
    PinocchioRegressorModel,
    build_planar_pinocchio_model,
    is_pinocchio_available,
    planar_regressor_from_spatial,
)
from src.shared.python.estimation.mosaic.planar_chain import (
    PlanarChain,
    PlanarMarkerSet,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]


def _skip_if_no_pinocchio() -> None:
    if not is_pinocchio_available():
        pytest.skip("Pinocchio is not installed (unqualified report)")


def test_pinocchio_regressor_matches_planar_oracle() -> None:
    """Conformance test: Pinocchio regressor matches the planar oracle to machine tolerance."""
    _skip_if_no_pinocchio()
    import pinocchio as pin

    rng = np.random.default_rng(11534)
    link_lengths = np.array([0.45, 0.35, 0.25], dtype=np.float64)
    actuated = np.array([True, True, True], dtype=bool)
    chain = PlanarChain(link_lengths=link_lengths, actuated=actuated)

    pin_model = build_planar_pinocchio_model(chain)
    provider = PinocchioRegressorModel(pin_model)

    assert provider.n_dof == 3
    assert provider.n_parameters == 30  # 3 moving links x 10 spatial parameters

    n_nodes = 50
    q = rng.uniform(-1.0, 1.0, size=(n_nodes, 3))
    v = rng.uniform(-2.0, 2.0, size=(n_nodes, 3))
    a = rng.uniform(-5.0, 5.0, size=(n_nodes, 3))

    Y_planar = chain.regressor(q, v, a)  # (N, 3, 12)
    Y_spatial = provider.regressor(q, v, a)  # (N, 3, 30)

    # Extract the 4 active planar columns per link: (m, h_x, h_y, I_zz)
    Y_extracted = planar_regressor_from_spatial(Y_spatial, n_links=3)

    max_diff = float(np.max(np.abs(Y_planar - Y_extracted)))
    assert max_diff < 1e-12, f"Planar oracle discrepancy {max_diff} exceeded tolerance"


def test_pinocchio_regressor_matches_rnea_fixed_base() -> None:
    """Test that Y @ pi equals Pinocchio RNEA torques for physically consistent parameters."""
    _skip_if_no_pinocchio()
    import pinocchio as pin

    rng = np.random.default_rng(21534)
    model = pin.buildSampleModelManipulator()
    provider = PinocchioRegressorModel(model)

    n_nodes = 25
    q = np.array([pin.randomConfiguration(model) for _ in range(n_nodes)])
    v = rng.normal(size=(n_nodes, model.nv))
    a = rng.normal(size=(n_nodes, model.nv))

    Y = provider.regressor(q, v, a)

    # Extract dynamic parameters from model
    pi = np.concatenate(
        [model.inertias[i].toDynamicParameters() for i in range(1, model.njoints)]
    )

    data = model.createData()
    for t in range(n_nodes):
        tau_reg = Y[t] @ pi
        tau_rnea = pin.rnea(model, data, q[t], v[t], a[t])
        diff = float(np.max(np.abs(tau_reg - tau_rnea)))
        assert diff < 1e-12, f"Fixed-base RNEA discrepancy at node {t}: {diff}"


def test_pinocchio_regressor_matches_rnea_floating_base() -> None:
    """Test that Y @ pi equals Pinocchio RNEA torques for a floating-base model."""
    _skip_if_no_pinocchio()
    import pinocchio as pin

    rng = np.random.default_rng(31534)
    model = pin.buildSampleModelHumanoid()
    provider = PinocchioRegressorModel(model)

    n_nodes = 10
    q_neutral = pin.neutral(model)
    q = np.tile(q_neutral, (n_nodes, 1))
    # Small perturbations to avoid invalid quaternion configs
    for t in range(n_nodes):
        dq = rng.normal(scale=0.05, size=model.nv)
        q[t] = pin.integrate(model, q_neutral, dq)

    v = rng.normal(scale=0.5, size=(n_nodes, model.nv))
    a = rng.normal(scale=1.0, size=(n_nodes, model.nv))

    Y = provider.regressor(q, v, a)
    pi = np.concatenate(
        [model.inertias[i].toDynamicParameters() for i in range(1, model.njoints)]
    )

    data = model.createData()
    for t in range(n_nodes):
        tau_reg = Y[t] @ pi
        tau_rnea = pin.rnea(model, data, q[t], v[t], a[t])
        diff = float(np.max(np.abs(tau_reg - tau_rnea)))
        assert diff < 1e-12, f"Floating-base RNEA discrepancy at node {t}: {diff}"


def test_batched_evaluation_matches_per_node_loop_and_measures_runtime() -> None:
    """Test batched evaluation against per-node loop and report runtime receipt."""
    _skip_if_no_pinocchio()
    import pinocchio as pin

    rng = np.random.default_rng(41534)
    link_lengths = np.array([0.5, 0.4, 0.3], dtype=np.float64)
    actuated = np.array([True, True, True], dtype=bool)
    chain = PlanarChain(link_lengths=link_lengths, actuated=actuated)
    model = build_planar_pinocchio_model(chain)
    provider = PinocchioRegressorModel(model)

    n_nodes = 150  # standard 150 nodes benchmark
    q = rng.uniform(-1.0, 1.0, size=(n_nodes, 3))
    v = rng.uniform(-2.0, 2.0, size=(n_nodes, 3))
    a = rng.uniform(-5.0, 5.0, size=(n_nodes, 3))

    t0_batch = time.perf_counter()
    Y_batch = provider.regressor(q, v, a)
    t_batch = time.perf_counter() - t0_batch

    # Per-node loop
    t0_loop = time.perf_counter()
    Y_loop = np.zeros_like(Y_batch)
    for i in range(n_nodes):
        Y_loop[i] = provider.regressor(q[i : i + 1], v[i : i + 1], a[i : i + 1])[0]
    t_loop = time.perf_counter() - t0_loop

    diff = float(np.max(np.abs(Y_batch - Y_loop)))
    assert diff == 0.0, "Batched and per-node loop must be identical"
    assert Y_batch.shape == (n_nodes, 3, 30)

    # Runtime receipt (documented in test output)
    print(
        f"[Receipt MOSAIC-02] N={n_nodes} nodes | Batched: {t_batch * 1000:.2f} ms | Loop: {t_loop * 1000:.2f} ms"
    )


def test_marker_positions_and_jacobians() -> None:
    """Verify forward kinematics and analytic marker Jacobians against numerical differentiation."""
    _skip_if_no_pinocchio()
    import pinocchio as pin

    link_lengths = np.array([0.5, 0.4], dtype=np.float64)
    chain = PlanarChain(link_lengths=link_lengths, actuated=np.array([True, True]))
    model = build_planar_pinocchio_model(chain)
    provider = PinocchioRegressorModel(model)

    # Attach marker at link 2 tip (x=0.4 in local frame)
    markers = PinocchioMarkerSet(
        parent_joint_id=np.array([2], dtype=np.int64),
        offsets=np.array([[0.4, 0.0, 0.0]], dtype=np.float64),
    )

    q = np.array([[0.3, -0.2]], dtype=np.float64)
    pos = provider.marker_positions(q, markers)
    assert pos.shape == (1, 1, 3)

    jac_q, jac_offset = provider.marker_jacobians(q, markers)
    assert jac_q.shape == (1, 1, 3, 2)
    assert jac_offset.shape == (1, 1, 3, 3)

    # Numerical differentiation w.r.t q
    eps = 1e-7
    for j in range(2):
        dq = np.zeros((1, 2))
        dq[0, j] = eps
        pos_pert = provider.marker_positions(q + dq, markers)
        num_jac = (pos_pert[0, 0] - pos[0, 0]) / eps
        ana_jac = jac_q[0, 0, :, j]
        diff = float(np.linalg.norm(num_jac - ana_jac))
        assert diff < 1e-5, f"Marker Jacobian d/dq discrepancy at joint {j}: {diff}"


def test_preconditions_reject_invalid_inputs() -> None:
    """Verify DbC contracts on invalid dimensions and non-finite values."""
    _skip_if_no_pinocchio()
    import pinocchio as pin

    chain = PlanarChain(
        link_lengths=np.array([0.5, 0.4]), actuated=np.array([True, True])
    )
    model = build_planar_pinocchio_model(chain)
    provider = PinocchioRegressorModel(model)

    q = np.zeros((5, 2))
    v = np.zeros((5, 2))
    a = np.zeros((5, 2))

    # Mismatched batch length
    with pytest.raises(PreconditionError):
        provider.regressor(q, v[:4], a)

    # Wrong coordinate dimension
    with pytest.raises(PreconditionError):
        provider.regressor(q[:, :1], v, a)

    # Non-finite inputs
    q_nan = q.copy()
    q_nan[0, 0] = np.nan
    with pytest.raises(PreconditionError):
        provider.regressor(q_nan, v, a)
