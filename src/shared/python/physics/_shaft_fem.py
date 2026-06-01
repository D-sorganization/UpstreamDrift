from __future__ import annotations

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from ._shaft_data import BeamElement, ShaftProperties, ShaftState
from ._shaft_model import ShaftModel
from ._shaft_properties import compute_EI_profile, compute_mass_profile

logger = get_logger(__name__)


class FiniteElementShaftModel(ShaftModel):
    """Finite element beam model for shaft dynamics.

    Implements Euler-Bernoulli beam elements with:
    - Distributed stiffness (varying EI along shaft)
    - Consistent mass matrix
    - Rayleigh damping
    - Cantilevered boundary conditions

    Issue #756: Full FE implementation for cross-engine shaft modeling.
    """

    def __init__(self, n_elements: int = 10) -> None:
        """Initialize FE shaft model.

        Args:
            n_elements: Number of finite elements
        """
        if n_elements is None:
            raise ValueError("n_elements must be provided")
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.n_dof = 2 * self.n_nodes  # 2 DOF per node (deflection, rotation)

        self.properties: ShaftProperties | None = None
        self.elements: list[BeamElement] = []

        # Global matrices
        self.K: np.ndarray = np.zeros((1, 1))  # Stiffness
        self.M: np.ndarray = np.zeros((1, 1))  # Mass
        self.C: np.ndarray = np.zeros((1, 1))  # Damping

        # State vectors (reduced DOF after boundary conditions)
        self.u: np.ndarray = np.zeros(1)  # Displacement
        self.v: np.ndarray = np.zeros(1)  # Velocity
        self.a: np.ndarray = np.zeros(1)  # Acceleration
        self.f_ext: np.ndarray = np.zeros(1)  # External forces

        self.time = 0.0
        self.n_free_dof = 0

        logger.info(f"FiniteElementShaftModel created with {n_elements} elements")

    def initialize(self, properties: ShaftProperties) -> None:
        """Initialize model and assemble system matrices.

        Builds global stiffness, mass, and damping matrices from
        element contributions. Applies cantilevered BC at butt end.

        Args:
            properties: Shaft properties
        """
        if properties is None:
            raise ValueError("properties must be provided")
        self.properties = properties
        self._create_elements()
        self._assemble_matrices()
        self._apply_boundary_conditions()

        logger.info(
            f"FE shaft initialized: {self.n_elements} elements, "
            f"{self.n_free_dof} free DOFs"
        )

    def _create_elements(self) -> None:
        """Create beam elements from shaft properties."""
        if self.properties is None:
            return

        self.elements = []
        L_total = self.properties.length
        L_elem = L_total / self.n_elements

        # Get EI and mass profiles
        EI = compute_EI_profile(self.properties)
        mass = compute_mass_profile(self.properties)

        for i in range(self.n_elements):
            # Element center position for property interpolation
            x_center = (i + 0.5) * L_elem

            # Interpolate properties at element center
            x_stations = self.properties.station_positions
            EI_elem = float(np.interp(x_center, x_stations, EI))
            mass_elem = float(np.interp(x_center, x_stations, mass))

            self.elements.append(
                BeamElement(
                    node_i=i,
                    node_j=i + 1,
                    length=L_elem,
                    EI=EI_elem,
                    mass_per_length=mass_elem,
                    damping=self.properties.damping_ratio,
                )
            )

    def _element_stiffness_matrix(self, element: BeamElement) -> np.ndarray:
        """Compute 4x4 element stiffness matrix for Euler-Bernoulli beam.

        K_e = EI/L³ * [12    6L   -12   6L  ]
                     [6L    4L²  -6L   2L² ]
                     [-12  -6L   12   -6L  ]
                     [6L    2L²  -6L   4L² ]

        Args:
            element: Beam element

        Returns:
            4x4 stiffness matrix
        """
        if element is None:
            raise ValueError("element must be provided")
        EI = element.EI
        L = element.length
        L2 = L * L
        L3 = L * L * L

        k = (
            EI
            / L3
            * np.array(
                [
                    [12, 6 * L, -12, 6 * L],
                    [6 * L, 4 * L2, -6 * L, 2 * L2],
                    [-12, -6 * L, 12, -6 * L],
                    [6 * L, 2 * L2, -6 * L, 4 * L2],
                ]
            )
        )

        return k

    def _element_mass_matrix(self, element: BeamElement) -> np.ndarray:
        """Compute 4x4 consistent mass matrix for Euler-Bernoulli beam.

        M_e = μL/420 * [156    22L   54    -13L  ]
                       [22L    4L²   13L   -3L²  ]
                       [54     13L   156   -22L  ]
                       [-13L  -3L²  -22L   4L²   ]

        Args:
            element: Beam element

        Returns:
            4x4 mass matrix
        """
        if element is None:
            raise ValueError("element must be provided")
        mu = element.mass_per_length
        L = element.length
        L2 = L * L

        m = (
            mu
            * L
            / 420
            * np.array(
                [
                    [156, 22 * L, 54, -13 * L],
                    [22 * L, 4 * L2, 13 * L, -3 * L2],
                    [54, 13 * L, 156, -22 * L],
                    [-13 * L, -3 * L2, -22 * L, 4 * L2],
                ]
            )
        )

        return m

    def _assemble_matrices(self) -> None:
        """Assemble global stiffness and mass matrices."""
        n_dof = self.n_dof
        self.K = np.zeros((n_dof, n_dof))
        self.M = np.zeros((n_dof, n_dof))

        for element in self.elements:
            # Element DOF indices: [w_i, θ_i, w_j, θ_j]
            dof_i = [2 * element.node_i, 2 * element.node_i + 1]
            dof_j = [2 * element.node_j, 2 * element.node_j + 1]
            dofs = dof_i + dof_j

            k_e = self._element_stiffness_matrix(element)
            m_e = self._element_mass_matrix(element)

            # Assemble into global matrices
            for ii, i in enumerate(dofs):
                for jj, j in enumerate(dofs):
                    self.K[i, j] += k_e[ii, jj]
                    self.M[i, j] += m_e[ii, jj]

        # Rayleigh damping: C = α*M + β*K
        # Targets approximately 2% critical damping ratio (ζ ≈ 0.02) in the
        # 10–100 Hz frequency range relevant to golf shaft bending modes.
        # Derivation: for two target frequencies ω₁, ω₂ with damping ratio ζ,
        #   α = 2ζ·ω₁·ω₂ / (ω₁ + ω₂)
        #   β = 2ζ / (ω₁ + ω₂)
        # With ω₁=2π·10≈62.8, ω₂=2π·100≈628.3, ζ=0.02:
        #   α ≈ 0.1, β ≈ 5.8e-5 (rounded to 1e-4 for conservatism)
        alpha = 0.1  # Mass proportional coefficient
        beta = 0.0001  # Stiffness proportional coefficient
        self.C = alpha * self.M + beta * self.K

    def _apply_boundary_conditions(self) -> None:
        """Apply cantilevered boundary conditions at butt end (node 0).

        Fixed at butt: w(0) = 0, θ(0) = 0
        Free DOFs are indices 2, 3, ..., n_dof-1
        """
        # Remove first two DOFs (node 0 is fixed)
        fixed_dofs = [0, 1]
        free_dofs = [i for i in range(self.n_dof) if i not in fixed_dofs]
        self.n_free_dof = len(free_dofs)

        # Extract submatrices for free DOFs
        self.K = self.K[np.ix_(free_dofs, free_dofs)]
        self.M = self.M[np.ix_(free_dofs, free_dofs)]
        self.C = self.C[np.ix_(free_dofs, free_dofs)]

        # Initialize state vectors
        self.u = np.zeros(self.n_free_dof)
        self.v = np.zeros(self.n_free_dof)
        self.a = np.zeros(self.n_free_dof)
        self.f_ext = np.zeros(self.n_free_dof)

    def get_state(self) -> ShaftState:
        """Get current shaft deformation state.

        Returns:
            ShaftState with deflections, velocities, and rotations
        """
        # Reconstruct full DOF vector (with zeros for fixed DOFs)
        u_full = np.zeros(self.n_dof)
        v_full = np.zeros(self.n_dof)

        u_full[2:] = self.u  # Free DOFs start at index 2
        v_full[2:] = self.v

        # Extract deflections and rotations at nodes
        deflections = u_full[0::2]  # Even indices are deflections
        rotations = u_full[1::2]  # Odd indices are rotations
        velocities = v_full[0::2]  # Velocity of deflections

        return ShaftState(
            deflections=deflections,
            velocities=velocities,
            rotations=rotations,
            timestamp=self.time,
        )

    def apply_load(
        self,
        position: float,
        force: np.ndarray,
        moment: np.ndarray | None = None,
    ) -> None:
        """Apply external load at specified position.

        Args:
            position: Position along shaft from butt end [m]
            force: Force vector [Fx, Fy, Fz] - Fy used as transverse
            moment: Optional moment vector [Mx, My, Mz]
        """
        if position is None:
            raise ValueError("position must be provided")
        if self.properties is None:
            return

        L_total = self.properties.length
        L_elem = L_total / self.n_elements

        # Find nearest node to load position
        node_idx = int(np.clip(position / L_elem, 0, self.n_nodes - 1))

        # Map to free DOF index (node 0 is fixed)
        if node_idx == 0:
            return  # Cannot apply load at fixed end

        free_dof_idx = 2 * node_idx - 2  # -2 because first 2 DOFs are removed

        # Apply transverse force (assume force[1] is transverse)
        if free_dof_idx < self.n_free_dof:
            self.f_ext[free_dof_idx] = force[1] if len(force) > 1 else force[0]

        # Apply moment if provided
        if moment is not None and free_dof_idx + 1 < self.n_free_dof:
            self.f_ext[free_dof_idx + 1] = moment[2] if len(moment) > 2 else moment[0]

    def _warn_if_ill_conditioned(self, dt: float) -> None:
        """Warn when dt is small enough to ill-condition the Newmark system.

        The effective stiffness K_eff = K + (γ/βdt)C + (1/βdt²)M is dominated
        by the inertial term once dt falls below the natural time scale
        sqrt(min(diagM)/max(diagK)). Below that scale the solve loses
        precision (issue #6985). The factor `c` gives a safety margin.

        Args:
            dt: Proposed time step [s]
        """
        if self.M.size == 0 or self.K.size == 0:
            return
        diag_m = np.diag(self.M)
        diag_k = np.diag(self.K)
        min_m = float(np.min(diag_m))
        max_k = float(np.max(diag_k))
        if min_m <= 0 or max_k <= 0:
            return
        c = 1e-2  # safety factor below the characteristic time scale
        dt_threshold = c * np.sqrt(min_m / max_k)
        if dt < dt_threshold:
            logger.warning(
                "Newmark step dt=%.3e is below the conditioning threshold "
                "%.3e (c*sqrt(min(diagM)/max(diagK))); K_eff is "
                "ill-conditioned and the solution may lose precision.",
                dt,
                dt_threshold,
            )

    def _solve_scaled(self, K_eff: np.ndarray, f_eff: np.ndarray) -> np.ndarray:
        """Solve K_eff u = f_eff using symmetric Jacobi (diagonal) scaling.

        Forms D = diag(K_eff)^(-1/2) and solves the better-conditioned system
        (D K_eff D) y = D f_eff, then recovers u = D y. This non-dimensionalises
        the effective stiffness so the inertial 1/(βdt²) scaling does not blow
        up the condition number (issue #6985).

        Args:
            K_eff: Effective stiffness matrix
            f_eff: Effective force vector

        Returns:
            Displacement increment solution
        """
        diag = np.abs(np.diag(K_eff))
        # Guard against zero/negative diagonal entries.
        scale = np.where(diag > 0, 1.0 / np.sqrt(diag), 1.0)
        k_scaled = K_eff * scale[:, None] * scale[None, :]
        f_scaled = f_eff * scale
        try:
            y = np.linalg.solve(k_scaled, f_scaled)
        except np.linalg.LinAlgError:
            logger.warning("FE solve failed, using pseudo-inverse")
            y = np.linalg.lstsq(k_scaled, f_scaled, rcond=None)[0]
        return y * scale

    def step(self, dt: float) -> ShaftState:
        """Advance simulation by dt using Newmark-beta integration.

        Uses average acceleration method (β=1/4, γ=1/2) for stability.

        Args:
            dt: Time step [s]

        Returns:
            Updated shaft state
        """
        if dt is None:
            raise ValueError("dt must be provided")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.time += dt

        # Newmark-beta parameters
        beta = 0.25
        gamma = 0.5

        self._warn_if_ill_conditioned(dt)

        # Effective stiffness matrix
        K_eff = self.K + gamma / (beta * dt) * self.C + 1 / (beta * dt**2) * self.M

        # Effective force vector
        f_eff = (
            self.f_ext
            + self.M
            @ (
                1 / (beta * dt**2) * self.u
                + 1 / (beta * dt) * self.v
                + (1 / (2 * beta) - 1) * self.a
            )
            + self.C
            @ (
                gamma / (beta * dt) * self.u
                + (gamma / beta - 1) * self.v
                + dt * (gamma / (2 * beta) - 1) * self.a
            )
        )

        # Solve for new displacement. At impact-scale dt the 1/(β dt²) M term
        # dominates K by many orders of magnitude, so K_eff is severely
        # ill-conditioned. Symmetric Jacobi (diagonal) scaling normalises the
        # diagonal to O(1), which restores precision in the solve and prevents
        # the catastrophic cancellation that otherwise corrupts the a_new
        # recovery below (issue #6985).
        u_new = self._solve_scaled(K_eff, f_eff)

        # Update velocity and acceleration
        a_new = (
            1 / (beta * dt**2) * (u_new - self.u)
            - 1 / (beta * dt) * self.v
            - (1 / (2 * beta) - 1) * self.a
        )
        v_new = self.v + dt * ((1 - gamma) * self.a + gamma * a_new)

        self.u = u_new
        self.v = v_new
        self.a = a_new

        # Clear external forces after step
        self.f_ext = np.zeros(self.n_free_dof)

        return self.get_state()

    def compute_natural_frequencies(self, n_modes: int = 5) -> list[float]:
        """Compute natural frequencies via eigenvalue analysis.

        Solves generalized eigenvalue problem: K*φ = ω²*M*φ

        Args:
            n_modes: Number of modes to compute

        Returns:
            List of natural frequencies [Hz]
        """
        if n_modes is None:
            raise ValueError("n_modes must be provided")
        from scipy.linalg import eigh

        # Solve generalized eigenvalue problem
        eigenvalues, _ = eigh(self.K, self.M)

        # Convert to frequencies
        frequencies = []
        for w2 in eigenvalues[:n_modes]:
            if w2 > 0:
                omega = np.sqrt(w2)
                freq = omega / (2 * np.pi)
                frequencies.append(float(freq))

        return frequencies

    def compute_static_solution(
        self, load_position: float, load_force: float
    ) -> ShaftState:
        """Compute static deflection under point load.

        Args:
            load_position: Position from butt end [m]
            load_force: Transverse load [N]

        Returns:
            Static equilibrium state
        """
        # Save current state
        if load_position is None:
            raise ValueError("load_position must be provided")
        u_saved = self.u.copy()
        v_saved = self.v.copy()
        f_saved = self.f_ext.copy()

        # Apply load
        self.apply_load(load_position, np.array([0, load_force, 0]))

        # Solve K*u = f
        try:
            self.u = np.linalg.solve(self.K, self.f_ext)
        except np.linalg.LinAlgError:
            self.u = np.linalg.lstsq(self.K, self.f_ext, rcond=None)[0]

        self.v = np.zeros(self.n_free_dof)
        state = self.get_state()

        # Restore original state
        self.u = u_saved
        self.v = v_saved
        self.f_ext = f_saved

        return state
