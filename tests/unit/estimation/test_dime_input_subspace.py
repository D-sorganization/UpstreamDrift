"""Behavioural tests for torque-independent drift feasibility and missing-data prediction (DIME-16, #11437).

Tests the rank-revealing input-effect subspace decomposition, covariance whitening,
orthogonal-complement torque-independent drift feasibility, bounded control feasibility,
changing contact rank, correlated uncertainty, masked observation prediction, and
runtime factor exclusivity against duplicate full dynamics.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import (
    DimeCompleteState,
    EstimationIntervalFactor,
    RuntimeExclusivityContract,
)
from src.shared.python.estimation.dime_input_subspace import (
    DimeInputSubspaceFactor,
    DimeSubspaceReport,
    MaskedIntervalControlPrior,
    MaskedPredictionResult,
    SubspaceContactInteraction,
    SubspaceDecomposition,
    SubspaceFeasibilityEvaluation,
    decompose_input_subspace,
    evaluate_input_subspace_feasibility,
    predict_masked_interval,
)
from src.shared.python.estimation.drift_prediction import ControlBand

pytestmark = pytest.mark.unit


# ==============================================================================
# Fixtures and Simple Analytical Dynamical Systems
# ==============================================================================


class _UnderactuatedCartPoleFixture:
    """Analytical 2-DOF underactuated cart-pole (x: unactuated base, theta: actuated).

    Dynamics: M(q) ddot{q} + c(q, v) = S^T tau
    Here base x has no motor; motor applies torque to pole theta.
    S = [0, 1].
    """

    def __init__(self) -> None:
        self.mc = 1.0  # cart mass
        self.mp = 0.5  # pole mass
        self.length = 0.8  # pole length
        self.g = 9.81

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        # q = [x, theta]
        theta = q[1]
        m11 = self.mc + self.mp
        m12 = self.mp * self.length * math.cos(theta)
        m21 = m12
        m22 = self.mp * self.length**2
        return np.array([[m11, m12], [m21, m22]], dtype=np.float64)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        # c(q, v) contains Coriolis, centrifugal, and gravity
        theta = q[1]
        theta_dot = v[1]
        c1 = -self.mp * self.length * (theta_dot**2) * math.sin(theta)
        c2 = self.mp * self.g * self.length * math.sin(theta)
        return np.array([c1, c2], dtype=np.float64)

    def selection_matrix(self) -> np.ndarray:
        # Only theta is actuated
        return np.array([[0.0], [1.0]], dtype=np.float64)


# ==============================================================================
# RED Cases
# ==============================================================================


class TestRedCases:
    """Required RED failure/refusal and negative constraint cases."""

    def test_fully_actuated_free_system_projection_adds_no_information(self) -> None:
        """RED: fully actuated free system where projection must add no information.

        When every degree of freedom is actuated with full rank (r == n),
        the orthogonal complement U_perp has dimension 0.
        Testing orthogonal discrepancy must yield zero residual and zero norm,
        confirming projection adds no independent restriction beyond control bounds.
        """
        n = 3
        m = 3
        # Full rank input influence matrix
        B = np.array(
            [[2.0, 0.5, 0.0], [0.5, 1.5, 0.2], [0.0, 0.2, 1.0]], dtype=np.float64
        )
        cov = np.diag([0.04, 0.04, 0.04])
        decomp = decompose_input_subspace(B, cov)

        assert decomp.is_fully_actuated
        assert not decomp.is_underactuated
        assert decomp.rank == n
        assert decomp.orthogonal_basis.shape[1] == 0  # 0 unactuated directions

        drift_acc = np.array([1.0, -9.81, 0.0])
        cand_acc = np.array([4.0, 2.0, -1.0])  # arbitrary candidate acceleration

        eval_res = evaluate_input_subspace_feasibility(
            candidate_accel=cand_acc,
            drift_accel=drift_acc,
            input_matrix=B,
            covariance=cov,
        )
        # Orthogonal residual must be exactly empty / zero norm
        assert eval_res.orthogonal_residual.size == 0
        assert eval_res.orthogonal_chi2 == pytest.approx(0.0, abs=1e-12)
        assert eval_res.is_drift_feasible

    def test_underactuated_cart_pole_torque_cannot_explain_arbitrary_base_acceleration(
        self,
    ) -> None:
        """RED: underactuated pendulum/cart where torque cannot explain arbitrary base acceleration.

        For a cart-pole where only the pole is actuated, an arbitrary base acceleration
        that deviates from the physical dynamic relationship cannot be achieved by ANY torque.
        The orthogonal complement residual must detect this dynamically impossible motion.
        """
        cart_pole = _UnderactuatedCartPoleFixture()
        q = np.array([0.0, 0.3])
        v = np.array([0.5, -0.2])
        M = cart_pole.mass_matrix(q)
        bias = cart_pole.bias_forces(q, v)
        S = cart_pole.selection_matrix()  # shape (2, 1)

        # Drift acceleration: M^{-1} (-bias)
        drift_acc = np.linalg.solve(M, -bias)
        # Control influence B = M^{-1} S
        B = np.linalg.solve(M, S)  # shape (2, 1)

        cov = np.eye(2) * 0.01

        # Feasible acceleration: drift + B * tau_true
        tau_true = np.array([2.5])
        accel_feasible = drift_acc + (B @ tau_true)

        res_feasible = evaluate_input_subspace_feasibility(
            candidate_accel=accel_feasible,
            drift_accel=drift_acc,
            input_matrix=B,
            covariance=cov,
            threshold_chi2=1.0,
        )
        assert res_feasible.is_drift_feasible
        assert res_feasible.orthogonal_chi2 == pytest.approx(0.0, abs=1e-9)

        # Infeasible acceleration: tamper with the unactuated cart acceleration
        accel_infeasible = accel_feasible.copy()
        accel_infeasible[0] += (
            5.0  # fictitious cart acceleration impossible via pole torque
        )

        res_infeasible = evaluate_input_subspace_feasibility(
            candidate_accel=accel_infeasible,
            drift_accel=drift_acc,
            input_matrix=B,
            covariance=cov,
            threshold_chi2=4.0,
        )
        assert not res_infeasible.is_drift_feasible
        assert res_infeasible.orthogonal_chi2 > 10.0
        assert not res_infeasible.is_overall_feasible

    def test_changing_contact_rank_smoothly_updates_subspace_and_receipt(self) -> None:
        """RED: changing contact rank between flight and constrained stance.

        Flight: no contact reactions (contact rank 0).
        Stance: contact constraint introduces constraint Jacobian J_c.
        When contact forces are eliminated, the effective drift and input subspace
        must reflect active contact dimensions without numerical singularity or crashes.
        """
        n = 3
        # 1 joint actuated, floating base 2 DOF unactuated
        B = np.zeros((n, 1), dtype=np.float64)
        B[2, 0] = 1.0  # only joint 2 actuated
        cov = np.eye(n) * 0.05
        drift_acc = np.array([0.0, -9.81, 0.0])
        cand_acc = np.array(
            [0.0, 0.0, 1.5]
        )  # base vertical acceleration = 0 (supported by ground)

        # 1. Flight: no contact forces, candidate vertical accel 0 is infeasible (gravity requires -9.81)
        res_flight = evaluate_input_subspace_feasibility(
            candidate_accel=cand_acc,
            drift_accel=drift_acc,
            input_matrix=B,
            covariance=cov,
            contact=None,
            threshold_chi2=9.0,
        )
        assert not res_flight.is_drift_feasible
        assert res_flight.contact_rank == 0

        # 2. Stance: vertical contact force balances gravity
        # J_c selects vertical coordinate (index 1)
        J_c = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        contact_force = np.array([9.81])  # 9.81 N upward
        mass_mat = np.eye(n)  # unit mass

        res_stance = evaluate_input_subspace_feasibility(
            candidate_accel=cand_acc,
            drift_accel=drift_acc,
            input_matrix=B,
            covariance=cov,
            contact=SubspaceContactInteraction(
                jacobian=J_c,
                forces=contact_force,
                mass_matrix=mass_mat,
            ),
            threshold_chi2=9.0,
        )
        # In stance with contact reaction eliminated, vertical acceleration 0 is physically feasible!
        assert res_stance.is_drift_feasible
        assert res_stance.contact_rank == 1

    def test_correlated_uncertainty_whitening_handles_coupled_sensor_noise(
        self,
    ) -> None:
        """RED: correlated uncertainty requires non-diagonal whitening.

        If observation error has strong covariance off-diagonals, unwhitened
        Euclidean projection gives distorted feasibility metrics.
        The whitened subspace must correctly decorrelate the error.
        """
        n = 2
        B = np.array([[1.0], [0.0]], dtype=np.float64)
        # Correlated covariance: var(1)=1.0, var(2)=1.0, cov(1,2)=0.9
        cov = np.array([[1.0, 0.9], [0.9, 1.0]], dtype=np.float64)

        decomp = decompose_input_subspace(B, cov)
        # Whitening matrix W must satisfy W @ cov @ W.T == eye(2)
        whitened_cov = decomp.whitening_transform @ cov @ decomp.whitening_transform.T
        np.testing.assert_allclose(whitened_cov, np.eye(2), atol=1e-10)

        # An acceleration deviation along the stiff correlated eigenvector should have high chi2
        # while along the compliant eigenvector should have lower chi2
        eigvals, eigvecs = np.linalg.eigh(cov)
        stiff_vec = eigvecs[:, 0]  # eigenvalue ~ 0.1
        compliant_vec = eigvecs[:, 1]  # eigenvalue ~ 1.9

        drift = np.zeros(2)
        eval_stiff = evaluate_input_subspace_feasibility(
            candidate_accel=stiff_vec,
            drift_accel=drift,
            input_matrix=B,
            covariance=cov,
        )
        eval_comp = evaluate_input_subspace_feasibility(
            candidate_accel=compliant_vec,
            drift_accel=drift,
            input_matrix=B,
            covariance=cov,
        )
        # Chi2 along stiff (low-variance) direction must be substantially higher than along compliant direction
        assert eval_stiff.orthogonal_chi2 > eval_comp.orthogonal_chi2

    def test_singular_near_singular_input_maps_drop_directions_into_unactuated_subspace(
        self,
    ) -> None:
        """RED: singular/near-singular input maps must not produce explosive torques.

        Near-singular directions below tolerance must be classified as unactuated (U_perp)
        rather than inverted with 1e12 control efforts.
        """
        n = 2
        # Input matrix where second direction has tiny singular value
        B = np.array([[1.0, 0.0], [0.0, 1e-9]], dtype=np.float64)
        cov = np.eye(2)

        decomp = decompose_input_subspace(B, cov, tolerance=1e-6)
        assert decomp.rank == 1  # near-singular direction dropped
        assert decomp.is_underactuated
        assert decomp.orthogonal_basis.shape[1] == 1  # 1 unactuated direction

        # Candidate acceleration trying to accelerate along the near-singular second coordinate
        cand_acc = np.array([1.0, 10.0])
        drift_acc = np.array([0.0, 0.0])
        band = ControlBand(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0]))

        res = evaluate_input_subspace_feasibility(
            candidate_accel=cand_acc,
            drift_accel=drift_acc,
            input_matrix=B,
            covariance=cov,
            control_band=band,
            tolerance=1e-6,
        )
        # Since the second coordinate is unactuated, the 10.0 acceleration produces high orthogonal residual
        assert not res.is_drift_feasible
        assert res.orthogonal_chi2 > 50.0


# ==============================================================================
# GREEN Cases
# ==============================================================================


class TestGreenCases:
    """Required GREEN analytical, scale invariance, and verification cases."""

    def test_analytic_subspace_projection_matches_ground_truth(self) -> None:
        """GREEN: analytic subspace cases match closed-form orthogonal projection."""
        # 3D space, B = [1, 0, 0]^T, isotropic noise cov = sigma^2 * I
        B = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
        sigma2 = 0.25
        cov = np.eye(3) * sigma2

        decomp = decompose_input_subspace(B, cov)
        assert decomp.rank == 1

        delta_a = np.array([2.5, -1.0, 2.0])
        drift = np.zeros(3)
        cand = delta_a

        eval_res = evaluate_input_subspace_feasibility(
            candidate_accel=cand,
            drift_accel=drift,
            input_matrix=B,
            covariance=cov,
        )
        # Orthogonal components are indices 1 and 2: (-1.0, 2.0)
        # Whitened: (-1.0 / 0.5, 2.0 / 0.5) = (-2.0, 4.0)
        # Chi2 = (-2)^2 + (4)^2 = 4 + 16 = 20.0
        assert eval_res.orthogonal_chi2 == pytest.approx(20.0, rel=1e-9)

    def test_scale_invariant_whitening(self) -> None:
        """GREEN: scale-invariant whitening: scaling covariance scales Mahalanobis chi2 by 1/alpha."""
        B = np.array([[1.0, 0.0], [0.2, 0.8], [0.0, 0.5]], dtype=np.float64)
        base_cov = np.array(
            [[0.1, 0.02, 0.01], [0.02, 0.15, 0.03], [0.01, 0.03, 0.2]], dtype=np.float64
        )
        alpha = 4.0
        scaled_cov = alpha * base_cov

        cand = np.array([1.0, -0.5, 2.0])
        drift = np.array([0.2, 0.1, -0.3])

        res1 = evaluate_input_subspace_feasibility(
            candidate_accel=cand,
            drift_accel=drift,
            input_matrix=B,
            covariance=base_cov,
        )
        res2 = evaluate_input_subspace_feasibility(
            candidate_accel=cand,
            drift_accel=drift,
            input_matrix=B,
            covariance=scaled_cov,
        )
        # Chi2 with scaled covariance must equal res1.orthogonal_chi2 / alpha
        assert res2.orthogonal_chi2 == pytest.approx(
            res1.orthogonal_chi2 / alpha, rel=1e-9
        )

    def test_masked_observation_prediction_with_calibrated_intervals(self) -> None:
        """GREEN: masked-observation prediction retains unactuated drift while broadening control subspace.

        Poor-information intervals retain uncertainty along the actuated directions,
        while unactuated directions strictly follow deterministic drift.
        """
        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.0, 0.5]),
            v=np.array([1.0, -0.5]),
            model_hash="test_model_v1",
        )
        # 2-DOF: index 0 is unactuated drift, index 1 is actuated
        B = np.array([[0.0], [1.0]], dtype=np.float64)
        drift_acc = np.array([-2.0, -9.81])
        u_mean = np.array([9.81])
        u_cov = np.array([[4.0]])  # std = 2.0 N*m
        dt = 0.05
        horizon = 5

        # Masked observation: data missing across entire interval
        mask = np.ones(horizon, dtype=bool)  # True = missing

        pred = predict_masked_interval(
            initial_state=initial_state,
            input_matrix=B,
            drift_accel=drift_acc,
            control_prior=MaskedIntervalControlPrior(mean=u_mean, covariance=u_cov),
            observation_mask=mask,
            dt=dt,
            horizon=horizon,
            model_discrepancy=np.diag([0.001, 0.001, 0.001, 0.001]),
        )
        assert pred.horizon_valid
        assert pred.trajectory_mean.shape == (horizon + 1, 4)
        assert pred.trajectory_covariance.shape == (horizon + 1, 4, 4)

        # Unactuated coordinate (DOF 0) velocity variance grows only due to model discrepancy,
        # whereas actuated coordinate (DOF 1) velocity variance grows strongly with control prior uncertainty
        var_v0_final = pred.trajectory_covariance[-1, 2, 2]
        var_v1_final = pred.trajectory_covariance[-1, 3, 3]
        assert var_v1_final > 8.0 * var_v0_final
        # Unactuated position variance is strictly lower than actuated position variance
        assert (
            pred.trajectory_covariance[-1, 1, 1] > pred.trajectory_covariance[-1, 0, 0]
        )

    def test_exclusivity_contract_rejects_duplicate_full_dynamics_factor(self) -> None:
        """GREEN: runtime factor ownership check rejects duplication between reduced and full dynamics."""
        contract = RuntimeExclusivityContract()

        # Register full dynamics factor
        full_factor = EstimationIntervalFactor(
            name="full_mhe_dynamics",
            factor_type="explicit_input_likelihood",
            t_start=0.0,
            t_end=1.0,
            contributes_to_objective=True,
        )
        contract.register_factor(full_factor)

        # 1. Permitted as diagnostic (contributes_to_objective=False)
        diag_subspace = DimeInputSubspaceFactor(
            name="subspace_screener",
            t_start=0.2,
            t_end=0.8,
            mode="diagnostic",
            contributes_to_objective=False,
        )
        diag_factor = diag_subspace.to_interval_factor()
        contract.register_factor(diag_factor)  # must succeed without error

        # 2. Rejected as competing objective factor on overlapping interval
        competing_subspace = DimeInputSubspaceFactor(
            name="competing_reduced_subspace",
            t_start=0.2,
            t_end=0.8,
            mode="reduced_subspace_dynamics",
            contributes_to_objective=True,
        )
        with pytest.raises(PreconditionError, match="[Rr]untime exclusivity violation"):
            contract.register_factor(competing_subspace.to_interval_factor())

    def test_subspace_report_serialization_roundtrip(self) -> None:
        """GREEN: DimeSubspaceReport serializes and deserializes losslessly."""
        report = DimeSubspaceReport(
            time_s=1.25,
            dimension_n=3,
            actuated_dimension_m=2,
            subspace_rank_r=2,
            unactuated_dimension=1,
            singular_values=(2.5, 1.2),
            condition_number=2.0833,
            drift_acceleration=(0.0, -9.81, 0.5),
            unactuated_residual_norm=0.045,
            is_drift_feasible=True,
            is_control_feasible=True,
            is_overall_feasible=True,
            optimal_torque=(1.2, -0.4),
            prediction_horizon_valid=True,
            contact_active=False,
            contact_rank=0,
            diagnostics={"mode": "flight", "whitening": "cholesky"},
        )
        data = report.to_dict()
        restored = DimeSubspaceReport.from_dict(data)
        assert restored == report

    def test_finite_difference_cross_checks_in_smooth_modes(self) -> None:
        """GREEN: finite-difference cross-check of subspace residual Jacobian."""
        n = 3
        B = np.array([[1.0, 0.2], [0.0, 0.8], [0.0, 0.0]], dtype=np.float64)
        cov = np.array(
            [[0.2, 0.05, 0.0], [0.05, 0.3, 0.02], [0.0, 0.02, 0.1]], dtype=np.float64
        )
        drift = np.array([0.5, -9.81, 1.2])
        cand = np.array([1.2, -8.0, 0.8])

        decomp = decompose_input_subspace(B, cov)
        u_perp = decomp.orthogonal_basis
        w_mat = decomp.whitening_transform
        # Analytic Jacobian of orthogonal residual r_perp w.r.t candidate accel
        analytic_jac = u_perp.T @ w_mat  # shape: (n_unactuated, n)

        # Finite difference Jacobian
        eps = 1e-6
        k_perp = u_perp.shape[1]
        fd_jac = np.zeros((k_perp, n), dtype=np.float64)

        base_res = evaluate_input_subspace_feasibility(
            candidate_accel=cand,
            drift_accel=drift,
            input_matrix=B,
            covariance=cov,
        ).orthogonal_residual

        for j in range(n):
            cand_pert = cand.copy()
            cand_pert[j] += eps
            pert_res = evaluate_input_subspace_feasibility(
                candidate_accel=cand_pert,
                drift_accel=drift,
                input_matrix=B,
                covariance=cov,
            ).orthogonal_residual
            fd_jac[:, j] = (pert_res - base_res) / eps

        np.testing.assert_allclose(fd_jac, analytic_jac, rtol=1e-5, atol=1e-7)
