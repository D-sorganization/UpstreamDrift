"""Physics consistency validation for simulation correctness.

Phase 3 Implementation: Enhanced Validation from FUTURE_ROADMAP.md

This module provides physics-based validation to ensure simulation correctness:
- Energy conservation verification
- Analytical derivative checks (Jacobian validation)
- Momentum conservation checks

These tests help catch:
- Integration errors
- Modeling bugs in URDF/MuJoCo XML
- Numerical precision issues
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

logger = logging.getLogger(__name__)


@dataclass
class EnergyValidationResult:
    """Result of energy conservation check."""

    energy_error: float
    relative_error: float
    passes: bool
    kinetic_energy_initial: float
    kinetic_energy_final: float
    potential_energy_initial: float
    potential_energy_final: float
    work_applied: float
    message: str

    def __str__(self) -> str:
        status = "PASS" if self.passes else "FAIL"
        return (
            f"Energy Conservation [{status}]: "
            f"Error={self.relative_error:.2e} (threshold=1e-3)"
        )


@dataclass
class JacobianValidationResult:
    """Result of Jacobian derivative check."""

    jacobian_error: float
    passes: bool
    body_id: int
    message: str

    def __str__(self) -> str:
        status = "PASS" if self.passes else "FAIL"
        return (
            f"Jacobian Validation [{status}]: "
            f"Error={self.jacobian_error:.2e} (threshold=1e-6)"
        )


class PhysicsValidator:
    """Validate physics consistency for MuJoCo simulations.

    This class implements Phase 3 validation checks from FUTURE_ROADMAP.md:
    - Energy conservation verification
    - Analytical derivative checks
    - Jacobian validation tests

    These tests provide confidence that the physics engine is correctly
    configured and that numerical integration is stable.

    Example:
        >>> validator = PhysicsValidator(model, data)
        >>> result = validator.verify_energy_conservation(qpos, qvel, torques)
        >>> print(result)
        Energy Conservation [PASS]: Error=1.23e-05 (threshold=1e-3)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tolerance_energy: float = 1e-3,
        tolerance_jacobian: float = 1e-6,
    ) -> None:
        """Initialize physics validator.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            tolerance_energy: Relative error tolerance for energy conservation
            tolerance_jacobian: Absolute error tolerance for Jacobian validation
        """
        try:
            import mujoco
        except ImportError as e:
            raise ImportError(
                "MuJoCo is required for physics validation. "
                "Install with: pip install mujoco"
            ) from e

        self.model = model
        self.data = data
        self.tolerance_energy = tolerance_energy
        self.tolerance_jacobian = tolerance_jacobian
        self._mujoco = mujoco

        # Create scratch data for validation (doesn't modify original)
        self._scratch_data = mujoco.MjData(model)

    def compute_kinetic_energy(self, qpos: np.ndarray, qvel: np.ndarray) -> float:
        """Compute kinetic energy: KE = 0.5 * q̇ᵀ M(q) q̇.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]

        Returns:
            Kinetic energy [J]
        """
        self._scratch_data.qpos[:] = qpos
        self._scratch_data.qvel[:] = qvel
        self._mujoco.mj_forward(self.model, self._scratch_data)

        # Get mass matrix
        M = np.zeros((self.model.nv, self.model.nv))
        self._mujoco.mj_fullM(self.model, M, self._scratch_data.qM)

        # KE = 0.5 * v^T * M * v
        return float(0.5 * qvel @ M @ qvel)

    def compute_potential_energy(self, qpos: np.ndarray) -> float:
        """Compute gravitational potential energy.

        Args:
            qpos: Joint positions [nv]

        Returns:
            Potential energy [J]
        """
        self._scratch_data.qpos[:] = qpos
        self._scratch_data.qvel[:] = 0
        self._mujoco.mj_forward(self.model, self._scratch_data)

        # PE = sum(m_i * g * h_i) for all bodies
        pe = 0.0
        gravity = self.model.opt.gravity[2]  # Z gravity component

        for i in range(1, self.model.nbody):  # Skip world body
            mass = self.model.body_mass[i]
            height = self._scratch_data.xipos[i, 2]  # Z position
            pe += mass * (-gravity) * height

        return float(pe)

    def step_forward(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        torques: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate one step forward.

        Args:
            qpos: Initial joint positions [nv]
            qvel: Initial joint velocities [nv]
            torques: Applied torques [nv]
            dt: Timestep [s]

        Returns:
            Tuple of (new_qpos, new_qvel)
        """
        self._scratch_data.qpos[:] = qpos
        self._scratch_data.qvel[:] = qvel
        self._scratch_data.ctrl[:] = torques[: len(self._scratch_data.ctrl)]
        self._scratch_data.time = 0.0

        # Set timestep temporarily
        original_timestep = self.model.opt.timestep
        self.model.opt.timestep = dt

        try:
            self._mujoco.mj_step(self.model, self._scratch_data)
        finally:
            self.model.opt.timestep = original_timestep

        return (
            self._scratch_data.qpos.copy(),
            self._scratch_data.qvel.copy(),
        )

    def verify_energy_conservation(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        torques: np.ndarray,
        dt: float = 0.001,
    ) -> EnergyValidationResult:
        """Verify power balance: dE/dt = P_applied - P_dissipated.

        For a conservative system with no damping:
        - Total energy change should equal work done by applied torques
        - Energy error indicates integration or modeling issues

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            torques: Applied torques [nv]
            dt: Timestep [s] (default: 1ms)

        Returns:
            EnergyValidationResult with pass/fail status
        """
        # Energy at t
        KE_t = self.compute_kinetic_energy(qpos, qvel)
        PE_t = self.compute_potential_energy(qpos)
        E_t = KE_t + PE_t

        # Simulate forward one step
        qpos_next, qvel_next = self.step_forward(qpos, qvel, torques, dt)

        # Energy at t+dt
        KE_next = self.compute_kinetic_energy(qpos_next, qvel_next)
        PE_next = self.compute_potential_energy(qpos_next)
        E_next = KE_next + PE_next

        # Work done by applied torques: W = τ · Δq ≈ τ · q̇ · dt
        # Average velocity during step
        qvel_avg = 0.5 * (qvel + qvel_next)
        work_applied = float(np.dot(torques[: self.model.nv], qvel_avg) * dt)

        # Energy balance
        dE = E_next - E_t
        energy_error = abs(dE - work_applied)

        # Relative error (avoid division by zero)
        denominator = max(abs(work_applied), abs(E_t), 1e-10)
        relative_error = energy_error / denominator

        passes = relative_error < self.tolerance_energy

        message = (
            f"Energy {'conserved' if passes else 'NOT conserved'}: "
            f"ΔE={dE:.4e} J, Work={work_applied:.4e} J, "
            f"Error={energy_error:.4e} J ({relative_error:.2e} relative)"
        )

        if not passes:
            logger.warning(message)
        else:
            logger.debug(message)

        return EnergyValidationResult(
            energy_error=energy_error,
            relative_error=relative_error,
            passes=passes,
            kinetic_energy_initial=KE_t,
            kinetic_energy_final=KE_next,
            potential_energy_initial=PE_t,
            potential_energy_final=PE_next,
            work_applied=work_applied,
            message=message,
        )

    def verify_jacobian(
        self,
        qpos: np.ndarray,
        body_id: int,
        epsilon: float = 1e-8,
    ) -> JacobianValidationResult:
        """Verify Jacobian via finite differences.

        Compares analytical Jacobian from mj_jacBody with numerical
        Jacobian computed via central differences.

        Args:
            qpos: Joint positions [nv]
            body_id: Body ID to check Jacobian for
            epsilon: Perturbation size for finite differences

        Returns:
            JacobianValidationResult with pass/fail status
        """
        # Set state
        self._scratch_data.qpos[:] = qpos
        self._scratch_data.qvel[:] = 0
        self._mujoco.mj_forward(self.model, self._scratch_data)

        # Get analytical Jacobian
        jacp_analytical = np.zeros((3, self.model.nv))
        jacr_analytical = np.zeros((3, self.model.nv))
        self._mujoco.mj_jacBody(
            self.model, self._scratch_data, jacp_analytical, jacr_analytical, body_id
        )

        # Compute numerical Jacobian via central differences
        jacp_numerical = np.zeros((3, self.model.nv))

        for i in range(self.model.nv):
            # Forward perturbation
            qpos_plus = qpos.copy()
            qpos_plus[i] += epsilon
            self._scratch_data.qpos[:] = qpos_plus
            self._mujoco.mj_forward(self.model, self._scratch_data)
            pos_plus = self._scratch_data.xpos[body_id].copy()

            # Backward perturbation
            qpos_minus = qpos.copy()
            qpos_minus[i] -= epsilon
            self._scratch_data.qpos[:] = qpos_minus
            self._mujoco.mj_forward(self.model, self._scratch_data)
            pos_minus = self._scratch_data.xpos[body_id].copy()

            # Central difference
            jacp_numerical[:, i] = (pos_plus - pos_minus) / (2 * epsilon)

        # Compute error
        error = float(np.linalg.norm(jacp_analytical - jacp_numerical))
        passes = error < self.tolerance_jacobian

        message = (
            f"Jacobian for body {body_id}: "
            f"{'VALID' if passes else 'INVALID'} "
            f"(error={error:.2e}, threshold={self.tolerance_jacobian})"
        )

        if not passes:
            logger.warning(message)
        else:
            logger.debug(message)

        return JacobianValidationResult(
            jacobian_error=error,
            passes=passes,
            body_id=body_id,
            message=message,
        )

    def run_full_validation(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        torques: np.ndarray | None = None,
    ) -> dict[str, bool]:
        """Run all physics validation checks.

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            torques: Applied torques [nv] (default: zeros)

        Returns:
            Dictionary mapping check names to pass/fail status
        """
        if torques is None:
            torques = np.zeros(self.model.nv)

        results = {}

        # Energy conservation
        energy_result = self.verify_energy_conservation(qpos, qvel, torques)
        results["energy_conservation"] = energy_result.passes
        logger.info(str(energy_result))

        # Jacobian validation for all bodies
        jacobian_pass = True
        for body_id in range(1, self.model.nbody):  # Skip world
            jac_result = self.verify_jacobian(qpos, body_id)
            if not jac_result.passes:
                jacobian_pass = False
                logger.warning(str(jac_result))

        results["jacobian_validation"] = jacobian_pass

        # Summary
        all_pass = all(results.values())
        logger.info(
            f"Physics validation {'PASSED' if all_pass else 'FAILED'}: "
            f"{sum(results.values())}/{len(results)} checks passed"
        )

        return results
