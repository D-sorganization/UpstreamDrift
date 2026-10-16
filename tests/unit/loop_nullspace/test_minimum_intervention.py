"""NSX-93 (#12722): minimum-intervention (task-relevant) delayed TVLQR gains.

The gain is the MOSAIC ``tvlqr_gains`` with a zero running state cost and the
terminal weight ``C_imp^T W_y C_imp``.  With no running cost the optimal
cost-to-go depends on the state only through the weighted outcome that the
state would produce under zero control, so the gain annihilates exactly the
directions that leave the weighted outcome unchanged (the kernel of the NSX-90
state kernel ``S_x[t]``).  That is the testable statement of "corrects only
task-relevant deviations".
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.local_policy import (
    LinearizedDynamics,
    tvlqr_gains,
)
from src.shared.python.loop_nullspace.minimum_intervention import (
    face_only_weights,
    joint_space_gains,
    minimum_intervention_design,
    task_irrelevant_gain_norm,
)
from src.shared.python.loop_nullspace.variance_propagation import adjoint_kernels

pytestmark = pytest.mark.unit

NX, NU, T = 4, 2, 12
KERNEL_TOL = 1e-10  # issue acceptance; observed ~1e-15 (see the test)


def _plant() -> LinearizedDynamics:
    rng = np.random.default_rng(7)
    dt = 0.02
    a = np.stack([np.eye(NX) + dt * rng.normal(size=(NX, NX)) for _ in range(T)])
    b = np.stack([dt * rng.normal(size=(NX, NU)) for _ in range(T)])
    return LinearizedDynamics(a, b)


# outcome rows: face angle and a second (club-head-speed-like) channel
C = np.array([[1.0, 0.5, 0.0, 0.2], [0.0, 1.0, 1.0, 0.0]])
R = 1e-2 * np.eye(NU)


def test_gain_vanishes_on_the_kernel_of_the_face_outcome_to_1e_10() -> None:
    plant = _plant()
    w = face_only_weights(2, face_index=0, weight=4.0)
    design = minimum_intervention_design(plant, C, w, R)
    assert design.gains.shape == (T, NU, NX)
    resid = task_irrelevant_gain_norm(design.gains, plant, C, w)
    assert resid < KERNEL_TOL  # observed ~1e-14: a few ulps of a 4x4 SVD
    # and the statement is not vacuous: the gain is large on relevant directions
    assert np.max(np.abs(design.gains)) > 1.0


def test_the_kernel_is_nontrivial_at_every_step() -> None:
    plant = _plant()
    w = face_only_weights(2, 0)
    rows = np.sqrt(w) @ C
    kernels = adjoint_kernels(plant, rows)
    for t in range(T):
        assert np.linalg.matrix_rank(kernels.state[t]) == 1  # one face row
        # NX - 1 irrelevant directions per step
        assert NX - np.linalg.matrix_rank(kernels.state[t]) == NX - 1


def test_a_second_weighted_outcome_makes_more_directions_relevant() -> None:
    plant = _plant()
    w_face = face_only_weights(2, 0)
    face = minimum_intervention_design(plant, C, w_face, R)
    both = minimum_intervention_design(plant, C, np.eye(2), R)
    assert not np.allclose(face.gains, both.gains)
    # ker(face and chs) is inside ker(face): the face-only gain also vanishes there
    assert task_irrelevant_gain_norm(face.gains, plant, C, np.eye(2)) < KERNEL_TOL
    # the two-outcome gain reacts to deviations that only the second outcome sees
    assert task_irrelevant_gain_norm(both.gains, plant, C, w_face) > 1e-3
    assert task_irrelevant_gain_norm(both.gains, plant, C, np.eye(2)) < KERNEL_TOL


def test_the_terminal_weight_and_last_gain_are_the_closed_form() -> None:
    plant = _plant()
    w = np.diag([3.0, 0.0])
    design = minimum_intervention_design(plant, C, w, R)
    v = C.T @ w @ C
    np.testing.assert_allclose(design.terminal_weight, v)
    a, b = plant.a[-1], plant.b[-1]
    last = np.linalg.solve(R + b.T @ v @ b, b.T @ v @ a)
    np.testing.assert_allclose(design.gains[-1], last, rtol=1e-12)
    # and it is exactly the MOSAIC recursion with zero running state cost
    ref = tvlqr_gains(plant, np.zeros((NX, NX)), R, v)
    np.testing.assert_array_equal(design.gains, ref)


def test_cheap_control_drives_the_weighted_outcome_to_zero() -> None:
    """With ``R -> 0`` the corrected face outcome vanishes for any start."""
    plant = _plant()
    w = face_only_weights(2, 0)
    design = minimum_intervention_design(plant, C, w, 1e-10 * np.eye(NU))
    x = np.array([0.3, -0.2, 0.5, 0.1])
    x_open = x.copy()
    for t in range(T):
        x = plant.a[t] @ x - plant.b[t] @ design.gains[t] @ x
        x_open = plant.a[t] @ x_open
    assert abs(C[0] @ x_open) > 0.05
    assert abs(C[0] @ x) < 1e-6 * abs(C[0] @ x_open)


def test_joint_space_gains_reproduce_the_tangent_state_feedback() -> None:
    rng = np.random.default_rng(3)
    nv, k = 5, 2
    basis, _ = np.linalg.qr(rng.normal(size=(nv, k)))
    offset = rng.normal(size=(nv, k))
    k_xi, k_eta = rng.normal(size=(NU, k)), rng.normal(size=(NU, k))
    kq, kv = joint_space_gains(np.hstack([k_xi, k_eta]), basis, offset)
    xi, eta = rng.normal(size=k), rng.normal(size=k)
    dq = basis @ xi
    dv = basis @ eta + offset @ xi
    # u = -K x for the deviation x; the replay applies +Kq e_q + Kv e_v, e = -dev
    u_replay = kq @ (-dq) + kv @ (-dv)
    np.testing.assert_allclose(u_replay, -(k_xi @ xi + k_eta @ eta), atol=1e-13)


def test_validation() -> None:
    plant = _plant()
    with pytest.raises(ValueError, match="outcome_weights"):
        minimum_intervention_design(plant, C, -np.eye(2), R)
    with pytest.raises(ValueError, match="input_weight"):
        minimum_intervention_design(plant, C, np.eye(2), -R)
    with pytest.raises(ValueError, match="outcome_rows"):
        minimum_intervention_design(plant, C[:, :3], np.eye(2), R)
    with pytest.raises(ValueError, match="outcome_weights"):
        minimum_intervention_design(plant, C, np.eye(3), R)
    with pytest.raises(ValueError, match="face_index"):
        face_only_weights(2, 2)
