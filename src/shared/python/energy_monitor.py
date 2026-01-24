"""Energy conservation monitoring for physics simulations.

Assessment B Finding B-006 / Guideline O3 Implementation

This module provides real-time energy drift monitoring to detect integration
failures and ensure conservative system behaviors remain physically valid.

Per Guideline O3:
- Energy drift < 1% for conservative systems (passive pendulum, no damping)
- Drift > 1% triggers warning with corrective action recommendations
- Drift > 5% triggers critical error (integration failure)
"""

from __future__ import annotations

from src.shared.python.logging_config import get_logger
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shared.python.interfaces import PhysicsEngine

logger = get_logger(__name__)

# Conservation tolerances from Guideline O3
ENERGY_DRIFT_TOLERANCE_PCT = 1.0  # [%]
# Source: Guideline O3, conservative system tolerance

ENERGY_DRIFT_CRITICAL_PCT = 5.0  # [%]
# Source: Assessment B recommendation, integration failure threshold


@dataclass
class EnergySnapshot:
    """Snapshot of energy components at a specific time.

    Attributes:
        time: Simulation time [s]
        kinetic: Kinetic energy KE = 0.5 · v^T · M(q) · v [J]
        potential: Potential energy PE [J] (gravity + elastic)
        total: Total mechanical energy E = KE + PE [J]
    """

    time: float  # [s]
    kinetic: float  # [J]
    potential: float  # [J]

    @property
    def total(self) -> float:
        """Total mechanical energy E = KE + PE [J]."""
        return self.kinetic + self.potential


@dataclass
class ConservationMonitor:
    """Monitor energy conservation during simulation.

    Tracks energy drift and warns if integration quality degrades.

    Attributes:
        engine: Physics engine to monitor
        E_initial: Initial total energy [J] (set on first check)
        drift_history: List of (time, drift_pct) tuples
        max_drift_pct: Maximum allowed drift before warning [%]
        critical_drift_pct: Maximum drift before critical error [%]
    """

    engine: PhysicsEngine
    E_initial: float | None = None
    drift_history: list[tuple[float, float]] = field(default_factory=list)
    max_drift_pct: float = ENERGY_DRIFT_TOLERANCE_PCT
    critical_drift_pct: float = ENERGY_DRIFT_CRITICAL_PCT

    def initialize(self) -> None:
        """Initialize monitor with current energy as baseline.

        Call this at the start of simulation after setting initial conditions.

        Example:
            >>> monitor = ConservationMonitor(engine)
            >>> engine.set_state(q0, v0)
            >>> monitor.initialize()
        """
        snapshot = self.get_energy_snapshot()
        self.E_initial = snapshot.total
        self.drift_history.clear()

        logger.info(
            f"Energy monitor initialized: E₀ = {self.E_initial:.6f} J "
            f"(KE = {snapshot.kinetic:.6f} J, PE = {snapshot.potential:.6f} J)"
        )

    def get_energy_snapshot(self) -> EnergySnapshot:
        """Capture current energy state.

        Returns:
            EnergySnapshot with current KE, PE, time

        Note:
            Assumes engine has:
            - get_time() -> float
            - get_state() -> (q, v)
            - compute_mass_matrix() -> M(q)
            - compute_gravity_forces() -> g(q)
        """
        t = self.engine.get_time()
        q, v = self.engine.get_state()

        # Kinetic energy: KE = 0.5 · v^T · M(q) · v
        M = self.engine.compute_mass_matrix()
        KE = 0.5 * v.T @ M @ v

        # Exact PE would require integrating g(q) along path from a reference
        # configuration. For monitoring drift, we use the first-order form:
        # PE ≈ -q^T · g(q), assuming g(q) is approximately constant over small
        # displacements (e.g., point masses in a uniform gravity field near a
        # reference configuration).
        g = self.engine.compute_gravity_forces()
        PE = -q.T @ g  # First-order approximation, sufficient for drift monitoring

        return EnergySnapshot(time=t, kinetic=float(KE), potential=float(PE))

    def check_and_warn(self) -> float:
        """Check current energy drift and warn if exceeds tolerance.

        Guideline O3: Conservative systems should have <1% energy drift.

        Returns:
            Current drift percentage [%]

        Raises:
            IntegrationFailureError: If drift exceeds critical threshold (5%)

        Example:
            >>> monitor = ConservationMonitor(engine)
            >>> monitor.initialize()
            >>> for _ in range(1000):
            ...     engine.step(dt=0.001)
            ...     drift_pct = monitor.check_and_warn()  # Warns if drift > 1%
        """
        if self.E_initial is None:
            raise RuntimeError(
                "ConservationMonitor not initialized. Call initialize() first."
            )

        snapshot = self.get_energy_snapshot()
        E_current = snapshot.total
        drift_pct = (E_current - self.E_initial) / abs(self.E_initial) * 100

        # Record history
        self.drift_history.append((snapshot.time, drift_pct))

        # Critical drift (integration failure)
        if abs(drift_pct) > self.critical_drift_pct:
            raise IntegrationFailureError(
                f"❌ INTEGRATION FAILURE (Energy drift > {self.critical_drift_pct:.1f}%):\\n"
                f"  Time: {snapshot.time:.3f} s\\n"
                f"  Initial energy: {self.E_initial:.6f} J\\n"
                f"  Current energy: {E_current:.6f} J\\n"
                f"  Drift: {drift_pct:+.2f}% (critical threshold: {self.critical_drift_pct:.1f}%)\\n"
                f"  Recommended actions:\\n"
                f"    1. Reduce timestep by factor of 4\\n"
                f"    2. Switch to higher-order integrator (RK4)\\n"
                f"    3. Check constraint satisfaction (may be failing)\\n"
                f"  Cannot continue simulation - results unreliable."
            )

        # Warning drift (exceeds tolerance)
        elif abs(drift_pct) > self.max_drift_pct:
            logger.warning(
                f"⚠️ Energy conservation violated (Guideline O3):\\n"
                f"  Time: {snapshot.time:.3f} s\\n"
                f"  Initial energy: {self.E_initial:.6f} J\\n"
                f"  Current energy: {E_current:.6f} J\\n"
                f"  Drift: {drift_pct:+.2f}% (tolerance: {self.max_drift_pct:.1f}%)\\n"
                f"  Breakdown:\\n"
                f"    Kinetic: {snapshot.kinetic:.6f} J\\n"
                f"    Potential: {snapshot.potential:.6f} J\\n"
                f"  Likely causes:\\n"
                f"    - Timestep too large (try dt < {self.estimate_max_stable_timestep():.2e} s)\\n"
                f"    - Integrator unsuitable (use RK4 for conservative systems)\\n"
                f"    - Constraint violations accumulating\\n"
                f"  Recommendation: Reduce timestep and monitor constraint residuals"
            )

        return float(drift_pct)

    def estimate_max_stable_timestep(self) -> float:
        """Estimate maximum stable timestep for explicit integrators.

        Theory: Explicit Euler stable if dt < 2/λ_max, where λ_max is largest
        eigenvalue of linearized dynamics Jacobian.

        For typical biomechanical systems:
            - Natural frequencies: 1-10 Hz → dt < 0.01-0.1 s
            - Stiff systems (muscles, contacts): dt < 0.001 s

        Returns:
            Recommended maximum timestep [s]

        Note:
            This is a heuristic estimate. Actual stability depends on system
            dynamics and integration method.

        Reference:
            Hairer, Wanner (1996), "Solving ODEs II: Stiff Problems"
        """
        # Heuristic: For biomechanical systems, estimate based on velocity magnitude
        _, v = self.engine.get_state()
        v_norm = np.linalg.norm(v)

        # Typical angular velocity: ω ~ 10 rad/s → dt < 0.01 s
        # High-speed motion: ω ~ 100 rad/s → dt < 0.001 s
        if v_norm < 1.0:
            return 0.01  # Slow motion
        elif v_norm < 10.0:
            return 0.001  # Normal motion
        else:
            return 0.0001  # High-speed motion

    def project_to_energy_manifold(self) -> None:
        """Scale velocities to restore energy (variational integrator approximation).

        WARNING: This is a heuristic correction, not a rigorous variational integrator.
        Only use as emergency fix, not as primary integration method.

        Theory:
            E = 0.5·v^T·M·v + PE(q) ≈ 0.5·v^T·M·v (if KE >> PE)
            To restore E₀: scale v by √(E₀/E_current)

        Note:
            This assumes kinetic energy dominates. For systems where PE is
            significant (slow motion), this approximation is poor.
        """
        if self.E_initial is None:
            raise RuntimeError(
                "ConservationMonitor not initialized. Call initialize() first."
            )

        snapshot = self.get_energy_snapshot()
        E_current = snapshot.total

        if abs(E_current) < 1e-12:
            logger.warning("Cannot project to energy manifold: E_current ≈ 0")
            return

        # Scale velocities to restore total energy
        scale = np.sqrt(abs(self.E_initial / E_current))
        q, v = self.engine.get_state()
        self.engine.set_state(q, scale * v)

        logger.warning(
            f"⚠️ Energy manifold projection applied (heuristic fix):\\n"
            f"  Velocity scaled by factor: {scale:.6f}\\n"
            f"  This is NOT a rigorous variational integrator.\\n"
            f"  Recommendation: Fix root cause (reduce timestep, improve integrator)"
        )


class IntegrationFailureError(Exception):
    """Raised when energy drift indicates integration failure (>5% drift)."""

    pass
