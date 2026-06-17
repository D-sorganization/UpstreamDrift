# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Lagrangian inverse dynamics for a 3-link planar chain.

Contains ``LagrangianDynamics``.  Balance utilities (``balance_pose``,
``_standing_balanced``) live in ``lagrangian_balance`` and are re-exported
here for backward compatibility.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from ..backend import PhysicsBackend
from ..constants import CORIOLIS_SLOW_LIMIT_RAD_S
from .body_model import BodyModel, ChainGeometry
from .lagrangian_balance import _standing_balanced, balance_pose
from .lagrangian_batch import (
    batch_coriolis_torques,
    batch_gravity_torques,
    batch_inertia_torques,
    numpy_inverse_dynamics_batch,
)
from .lagrangian_kinematics import LagrangianKinematicsMixin

logger = logging.getLogger(__name__)

__all__ = [
    "LagrangianDynamics",
    "_standing_balanced",
    "balance_pose",
]


class LagrangianDynamics(LagrangianKinematicsMixin, PhysicsBackend):
    """Analytical inverse dynamics for a 3-link planar chain.

    Computes M(q)*qdd + C(q,qd) + G(q) = tau analytically.

    Inertia convention:
        ``I_segments`` are **centroidal** moments of inertia (about each
        segment's own COM), i.e. ``I_com = m * (rho * L)**2``. The diagonal
        mass-matrix construction adds the parallel-axis term ``m * d**2``
        itself, so callers must NOT pre-add it. Passing proximal-axis inertia
        (``I_com + m * d**2``) double-counts the parallel-axis term and
        overestimates the diagonal inertia (issue #490).

    Preconditions:
        body is a valid BodyModel
        m_segments, I_segments are length-3 arrays
        load_mass >= 0

    Complexity:
        Scalar dynamics methods are O(1) for the fixed 3-link chain.  Batch
        inverse dynamics is O(N) time and O(N) memory for ``N`` trajectory
        samples because each sample is evaluated independently.
    """

    # Process-wide latch so the "Rust accelerator unavailable" warning is
    # emitted at most once per run regardless of how many instances exist.
    _warned_rust_fallback: bool = False

    def __init__(
        self,
        body: BodyModel,
        m_segments: NDArray,
        I_segments: NDArray,
        load_mass: float,
        chain_geometry: ChainGeometry | None = None,
        supine: bool = False,
    ) -> None:
        """Initialise Lagrangian dynamics for a 3-link planar chain.

        Parameters:
            body: Anthropometric model (used for g, BOS, and default geometry).
            m_segments: Length-3 array of segment masses (kg).
            I_segments: Length-3 array of **centroidal** segment moments of
                inertia (about each segment COM, kg·m²). The parallel-axis
                term is added internally; do not pre-add it.
            load_mass: Mass of the external load at the chain tip (kg).
            chain_geometry: Optional ChainGeometry providing segment lengths,
                COM distances, and joint names for non-leg kinematic chains
                (e.g. the arm chain for bench press).
            supine: If True, gravity acts perpendicular to the chain axis
                (the lifter is lying down).  Gravity torque uses cos(q)
                instead of sin(q).  Used for bench press.
        """
        if len(m_segments) != 3:
            raise ValueError("need 3 segment masses")
        if load_mass < 0:
            raise ValueError("load_mass cannot be negative")
        self.body = body
        self.m = m_segments
        self.I = I_segments
        self.m_load = load_mass
        self.g = body.g
        self.supine = supine
        # Emit the "fast movement" Coriolis warning at most once per instance to
        # avoid flooding logs during an optimization sweep (issue #491).
        self._warned_fast_coriolis = False

        L, d = self._init_chain_geometry(body, chain_geometry)
        self._init_coupling_coefficients(m_segments, load_mass, L, d)
        self._init_diagonal_mass_constants(m_segments, I_segments, load_mass, L, d)
        self._init_gravity_coefficients(body.g, m_segments, load_mass, L, d)

    # -- init helpers -----------------------------------------------

    def _init_chain_geometry(
        self,
        body: BodyModel,
        chain_geometry: ChainGeometry | None,
    ) -> tuple[NDArray, NDArray]:
        """Set segment lengths, COM distances, and joint names.

        Returns the (L, d) arrays used by subsequent coefficient setup.
        """
        if chain_geometry is not None:
            L = chain_geometry.L
            d = chain_geometry.d
            self.L = L
            self.L_eff = L.copy()
            self.d = d
            self.d_eff = d.copy()
            self.joint_names = list(chain_geometry.joint_names)
        else:
            self.L = body.L
            self.L_eff = body.L_eff
            self.d = body.d
            self.d_eff = body.d_eff
            self.joint_names = ["ankle", "knee", "hip", "shoulder"]
            L = body.L
            d = body.d
        return L, d

    def _init_coupling_coefficients(
        self,
        m: NDArray,
        m_load: float,
        L: NDArray,
        d: NDArray,
    ) -> None:
        """Pre-compute off-diagonal coupling coefficients (constant for a given model)."""
        self._a01 = (m[1] * d[1] + (m[2] + m_load) * L[1]) * L[0]
        self._a02 = (m[2] * d[2] + m_load * L[2]) * L[0]
        self._a12 = (m[2] * d[2] + m_load * L[2]) * L[1]

    def _init_diagonal_mass_constants(
        self,
        m: NDArray,
        inertia: NDArray,
        m_load: float,
        L: NDArray,
        d: NDArray,
    ) -> None:
        """Pre-compute diagonal mass-matrix constants."""
        self._M00 = m[0] * d[0] ** 2 + (m[1] + m[2] + m_load) * L[0] ** 2 + inertia[0]
        self._M11 = m[1] * d[1] ** 2 + (m[2] + m_load) * L[1] ** 2 + inertia[1]
        self._M22 = m[2] * d[2] ** 2 + m_load * L[2] ** 2 + inertia[2]

    def _init_gravity_coefficients(
        self,
        g: float,
        m: NDArray,
        m_load: float,
        L: NDArray,
        d: NDArray,
    ) -> None:
        """Pre-compute gravity torque coefficients."""
        self._g0 = g * (m[0] * d[0] + (m[1] + m[2] + m_load) * L[0])
        self._g1 = g * (m[1] * d[1] + (m[2] + m_load) * L[1])
        self._g2 = g * (m[2] * d[2] + m_load * L[2])

    @property
    def segment_lengths(self) -> NDArray:
        """Lengths of the active kinematic chain segments."""
        return self.L

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom for this dynamics model (always 3)."""
        return 3

    @property
    def name(self) -> str:
        """Human-readable name identifying this dynamics backend."""
        return "Lagrangian (3-link planar)"

    def mass_matrix(self, q: NDArray) -> NDArray:
        """Compute the symmetric 3x3 joint-space mass (inertia) matrix.

        Args:
            q: Joint angle vector of shape (3,) in radians.

        Returns:
            Symmetric (3, 3) mass matrix M(q).

        Complexity:
            O(1) time and memory for the fixed 3-link model.
        """
        M = np.zeros((3, 3))
        M[0, 0] = self._M00
        M[1, 1] = self._M11
        M[2, 2] = self._M22
        M[0, 1] = M[1, 0] = self._a01 * math.cos(q[0] - q[1])
        M[0, 2] = M[2, 0] = self._a02 * math.cos(q[0] - q[2])
        M[1, 2] = M[2, 1] = self._a12 * math.cos(q[1] - q[2])
        return M

    def _coriolis_vector(self, q: NDArray, qd: NDArray) -> NDArray:
        """Centrifugal terms of the Coriolis/centrifugal vector.

        NOTE: This implementation includes only the centrifugal (qd_j^2)
        terms and omits the cross-velocity Coriolis terms (qd_i * qd_j,
        i != j).  This is a known simplification that is acceptable for
        slow barbell movements where cross-velocity products are small
        relative to centrifugal and gravitational terms.  A full
        Christoffel-symbol formulation would be needed for fast movements.
        """
        s01 = math.sin(q[0] - q[1])
        s02 = math.sin(q[0] - q[2])
        s12 = math.sin(q[1] - q[2])
        C = np.zeros(3)
        C[0] = self._a01 * s01 * qd[1] ** 2 + self._a02 * s02 * qd[2] ** 2
        C[1] = -self._a01 * s01 * qd[0] ** 2 + self._a12 * s12 * qd[2] ** 2
        C[2] = -self._a02 * s02 * qd[0] ** 2 - self._a12 * s12 * qd[1] ** 2
        return C

    def _check_coriolis_slow_assumption(self, max_abs_qd: float) -> None:
        """Warn once if joint speed exceeds the slow-movement Coriolis assumption.

        The Coriolis model omits cross-velocity terms (see ``_coriolis_vector``).
        Above ``CORIOLIS_SLOW_LIMIT_RAD_S`` those terms can be material and the
        reported torques may be underestimated (issue #491).
        """
        if max_abs_qd > CORIOLIS_SLOW_LIMIT_RAD_S and not self._warned_fast_coriolis:
            self._warned_fast_coriolis = True
            logger.warning(
                "max |qd| = %.2f rad/s exceeds slow-movement assumption "
                "(%.2f rad/s); Coriolis cross-velocity terms are omitted, joint "
                "torques may be underestimated",
                max_abs_qd,
                CORIOLIS_SLOW_LIMIT_RAD_S,
            )

    def _gravity_vector(self, q: NDArray) -> NDArray:
        """Compute the gravity loading vector G(q).

        Args:
            q: Joint angle vector of shape (3,) in radians.

        Returns:
            Gravity vector G(q) of shape (3,).
        """
        G = np.zeros(3)
        trig = math.cos if self.supine else math.sin
        G[0] = self._g0 * trig(q[0])
        G[1] = self._g1 * trig(q[1])
        G[2] = self._g2 * trig(q[2])
        return G

    def inverse_dynamics(self, q: NDArray, qd: NDArray, qdd: NDArray) -> NDArray:
        # Performance optimization:
        # In highly called scalar physical calculations like inverse_dynamics, avoid
        # composing intermediate structural arrays (e.g., 3x3 mass matrices via mass_matrix(),
        # 3x1 vectors via _coriolis_vector() and _gravity_vector()) for immediate matrix
        # multiplication. Instead, unroll the operations into simple scalars to save memory
        # allocation and loop overhead.
        q0, q1, q2 = q
        qd0, qd1, qd2 = qd
        qdd0, qdd1, qdd2 = qdd

        self._check_coriolis_slow_assumption(max(abs(qd0), abs(qd1), abs(qd2)))

        c01 = math.cos(q0 - q1)
        c02 = math.cos(q0 - q2)
        c12 = math.cos(q1 - q2)
        s01 = math.sin(q0 - q1)
        s02 = math.sin(q0 - q2)
        s12 = math.sin(q1 - q2)

        qd2_0 = qd0 * qd0
        qd2_1 = qd1 * qd1
        qd2_2 = qd2 * qd2

        a01_c01 = self._a01 * c01
        a02_c02 = self._a02 * c02
        a12_c12 = self._a12 * c12

        a01_s01 = self._a01 * s01
        a02_s02 = self._a02 * s02
        a12_s12 = self._a12 * s12

        trig = math.cos if self.supine else math.sin
        sq0 = trig(q0)
        sq1 = trig(q1)
        sq2 = trig(q2)

        tau0 = (
            self._M00 * qdd0
            + a01_c01 * qdd1
            + a02_c02 * qdd2
            + a01_s01 * qd2_1
            + a02_s02 * qd2_2
            + self._g0 * sq0
        )
        tau1 = (
            a01_c01 * qdd0
            + self._M11 * qdd1
            + a12_c12 * qdd2
            - a01_s01 * qd2_0
            + a12_s12 * qd2_2
            + self._g1 * sq1
        )
        tau2 = (
            a02_c02 * qdd0
            + a12_c12 * qdd1
            + self._M22 * qdd2
            - a02_s02 * qd2_0
            - a12_s12 * qd2_1
            + self._g2 * sq2
        )

        return np.array([tau0, tau1, tau2])

    def _batch_inertia_torques(
        self,
        qdd: NDArray,
        c01: NDArray,
        c02: NDArray,
        c12: NDArray,
    ) -> NDArray:
        """Inertia contribution — delegates to :func:`lagrangian_batch.batch_inertia_torques`."""
        return batch_inertia_torques(
            qdd,
            c01,
            c02,
            c12,
            M00=self._M00,
            M11=self._M11,
            M22=self._M22,
            a01=self._a01,
            a02=self._a02,
            a12=self._a12,
        )

    def _batch_coriolis_torques(
        self,
        qd: NDArray,
        s01: NDArray,
        s02: NDArray,
        s12: NDArray,
    ) -> NDArray:
        """Coriolis contribution — delegates to :func:`lagrangian_batch.batch_coriolis_torques`."""
        return batch_coriolis_torques(
            qd,
            s01,
            s02,
            s12,
            a01=self._a01,
            a02=self._a02,
            a12=self._a12,
        )

    def _batch_gravity_torques(self, q: NDArray) -> NDArray:
        """Gravity contribution — delegates to :func:`lagrangian_batch.batch_gravity_torques`."""
        return batch_gravity_torques(
            q,
            g0=self._g0,
            g1=self._g1,
            g2=self._g2,
            supine=self.supine,
        )

    def _numpy_inverse_dynamics_batch(self, q: NDArray, qd: NDArray, qdd: NDArray) -> NDArray:
        """NumPy fallback — delegates to :func:`lagrangian_batch.numpy_inverse_dynamics_batch`."""
        return numpy_inverse_dynamics_batch(
            q,
            qd,
            qdd,
            M00=self._M00,
            M11=self._M11,
            M22=self._M22,
            a01=self._a01,
            a02=self._a02,
            a12=self._a12,
            g0=self._g0,
            g1=self._g1,
            g2=self._g2,
            supine=self.supine,
        )

    def inverse_dynamics_batch(self, q: NDArray, qd: NDArray, qdd: NDArray) -> NDArray:
        """Vectorised batch torques (N, 3) for q/qd/qdd each (N, 3).

        Tries the Rust accelerator; falls back to _numpy_inverse_dynamics_batch.

        Preconditions:
            q, qd, qdd are all finite. A non-finite input (e.g. a NaN from a
            degenerate spline evaluation) is rejected with a ``ValueError``
            naming the offending array rather than silently propagating NaN
            torques into the cost (issue #499).

        Complexity:
            O(N) time and O(N) output memory for ``N`` trajectory samples.  The
            Rust and NumPy paths have the same asymptotic complexity.
        """
        self._require_finite_batch_inputs(q, qd, qdd)
        self._check_coriolis_slow_assumption(float(np.max(np.abs(qd))) if qd.size else 0.0)
        try:
            from movement_optimizer_core import inverse_dynamics_batch_rs  # type: ignore[import-not-found]  # noqa: I001

            return inverse_dynamics_batch_rs(
                q,
                qd,
                qdd,
                self._M00,
                self._M11,
                self._M22,
                self._a01,
                self._a02,
                self._a12,
                self._g0,
                self._g1,
                self._g2,
            )
        except ImportError:
            # The Rust accelerator is optional, but on a host where it *should*
            # be installed a failed import is a real (silent) performance and
            # parity regression — surface it at WARNING, once per process
            # (issue #494).
            if not LagrangianDynamics._warned_rust_fallback:
                LagrangianDynamics._warned_rust_fallback = True
                logger.warning(
                    "Rust accelerator (movement_optimizer_core) unavailable; "
                    "using NumPy batch inverse dynamics. Build with "
                    "`maturin develop --release` for the accelerated path."
                )
        return self._numpy_inverse_dynamics_batch(q, qd, qdd)

    @staticmethod
    def _require_finite_batch_inputs(q: NDArray, qd: NDArray, qdd: NDArray) -> None:
        """Reject non-finite batch inputs with a clear, located error.

        Performs one cheap ``np.isfinite(...).all()`` check per array so a NaN
        introduced upstream fails loudly at the dynamics boundary instead of
        propagating into NaN torques and a spurious infinite cost (issue #499).
        """
        for name, arr in (("q", q), ("qd", qd), ("qdd", qdd)):
            if not np.isfinite(arr).all():
                bad = int(np.count_nonzero(~np.isfinite(arr)))
                raise ValueError(
                    f"inverse_dynamics_batch received {bad} non-finite value(s) "
                    f"in '{name}'; refusing to compute NaN torques"
                )


# Kinematic methods (com_x_batch, forward_kinematics, bar_position,
# com_position) are inherited from LagrangianKinematicsMixin.
# Balance helpers (balance_pose, _standing_balanced) are imported from
# lagrangian_balance and re-exported via __all__ for backward compatibility.
