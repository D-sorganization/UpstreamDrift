"""Vectorized planar serial-chain dynamics in inertial-regressor form.

This is the analytic reference fixture for MOSAIC.  A planar chain of ``n``
revolute links (the golf double/triple pendulum lineage of AffineDrift) has
inverse dynamics that are *exactly linear* in the stacked inertial parameters
``pi = (m_i, h_ix, h_iy, I_io)`` of its links::

    tau = Y(q, v, a) pi

The same regressor yields the mass matrix (``a`` columns with gravity off),
the bias ``h(q, v) = Y(q, v, 0) pi`` and therefore the pointwise drift of the
control-affine form.  Forward dynamics, RK4 rollout, marker forward kinematics
and analytic marker Jacobians are provided so the estimator can be tested
end-to-end against a known ground truth without any physics engine.

Conventions: link ``i`` frame has its origin at joint ``i`` and its x-axis
along the link; absolute angles are measured from the world x-axis; gravity
defaults to ``(0, -9.81)``.  All kinematic methods are vectorized over a
leading batch of ``N`` samples (time nodes, trials, or both flattened).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.inertial import PLANAR_PARAMETERS_PER_BODY

FloatArray: TypeAlias = npt.NDArray[np.float64]

GRAVITY_DOWN: FloatArray = np.array([0.0, -9.81], dtype=np.float64)
_RK4_WEIGHTS = (1.0, 2.0, 2.0, 1.0)


def _cross2(x: FloatArray, y: FloatArray) -> FloatArray:
    """Scalar z-component of the planar cross product, batched on the last axis."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def _require_same_batch(*arrays: FloatArray) -> int:
    lengths = {array.shape[0] for array in arrays}
    require(len(lengths) == 1, "all inputs must share the batch length", lengths)
    return lengths.pop()


@dataclass(frozen=True)
class PlanarMarkerSet:
    """Markers rigidly attached to links at fixed local offsets (metres)."""

    link_index: npt.NDArray[np.int64]
    offsets: FloatArray

    def __post_init__(self) -> None:
        require(self.link_index.ndim == 1, "link_index must be 1-D")
        require(
            self.offsets.shape == (self.link_index.size, 2),
            "offsets must be (markers, 2)",
        )
        require(bool(np.all(self.link_index >= 0)), "link indices must be non-negative")

    @property
    def n_markers(self) -> int:
        return int(self.link_index.size)


@dataclass(frozen=True)
class DriftDecomposition:
    """Pointwise control-affine drift split: ZTCF = ZVCF + velocity part."""

    ztcf: FloatArray
    zvcf: FloatArray
    velocity_part: FloatArray


@dataclass(frozen=True)
class _ChainKinematics:
    phi: FloatArray
    omega: FloatArray
    alpha: FloatArray
    e: FloatArray
    n: FloatArray
    origin: FloatArray
    origin_acc: FloatArray


@dataclass(frozen=True)
class PlanarChain:
    """Pinned planar revolute chain with an actuation mask."""

    link_lengths: FloatArray
    actuated: npt.NDArray[np.bool_]
    gravity: FloatArray = field(default_factory=lambda: GRAVITY_DOWN.copy())

    def __post_init__(self) -> None:
        require(
            self.link_lengths.ndim == 1 and self.link_lengths.size > 0, "need >= 1 link"
        )
        require(bool(np.all(self.link_lengths > 0.0)), "link lengths must be positive")
        require(
            self.actuated.shape == self.link_lengths.shape, "actuated mask per joint"
        )
        require(self.gravity.shape == (2,), "gravity is a planar vector")

    @property
    def n_links(self) -> int:
        return int(self.link_lengths.size)

    @property
    def n_dof(self) -> int:
        """Generalized-coordinate count (one revolute joint per link)."""
        return self.n_links

    @property
    def n_parameters(self) -> int:
        return PLANAR_PARAMETERS_PER_BODY * self.n_links

    @property
    def n_inputs(self) -> int:
        return int(np.count_nonzero(self.actuated))

    @property
    def unactuated_indices(self) -> npt.NDArray[np.int64]:
        return np.flatnonzero(~self.actuated)

    @property
    def input_matrix(self) -> FloatArray:
        """Selection matrix ``B`` (n_links x n_inputs) mapping inputs to joints."""
        return np.eye(self.n_links)[:, self.actuated]

    # ------------------------------------------------------------------ kinematics
    def _kinematics(
        self, q: FloatArray, v: FloatArray, a: FloatArray
    ) -> _ChainKinematics:
        batch = _require_same_batch(q, v, a)
        require(q.shape == (batch, self.n_links), "q must be (N, n_links)", q.shape)
        phi, omega, alpha = (
            np.cumsum(q, axis=1),
            np.cumsum(v, axis=1),
            np.cumsum(a, axis=1),
        )
        e = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
        normal = np.stack([-np.sin(phi), np.cos(phi)], axis=-1)
        lengths = self.link_lengths[None, :, None]
        step = lengths * e
        origin = np.concatenate(
            [np.zeros((batch, 1, 2)), np.cumsum(step, axis=1)], axis=1
        )
        acc_step = lengths * (alpha[..., None] * normal - (omega**2)[..., None] * e)
        origin_acc = np.concatenate(
            [np.zeros((batch, 1, 2)), np.cumsum(acc_step, axis=1)], axis=1
        )
        return _ChainKinematics(
            phi, omega, alpha, e, normal, origin, origin_acc[:, :-1]
        )

    def regressor(
        self,
        q: FloatArray,
        v: FloatArray,
        a: FloatArray,
        gravity: FloatArray | None = None,
    ) -> FloatArray:
        """Return ``Y`` with shape ``(N, n_links, 4 n_links)`` so ``tau = Y @ pi``."""
        kin = self._kinematics(q, v, a)
        grav = self.gravity if gravity is None else gravity
        n, batch = self.n_links, q.shape[0]
        regressor = np.zeros(
            (batch, n, PLANAR_PARAMETERS_PER_BODY * n), dtype=np.float64
        )
        for i in range(n):
            e_i, n_i = kin.e[:, i], kin.n[:, i]
            alpha_i, omega_sq = kin.alpha[:, i, None], (kin.omega[:, i] ** 2)[:, None]
            net_acc = kin.origin_acc[:, i] - grav
            acc_hx = alpha_i * n_i - omega_sq * e_i
            acc_hy = -alpha_i * e_i - omega_sq * n_i
            col = PLANAR_PARAMETERS_PER_BODY * i
            for j in range(i + 1):
                lever = kin.origin[:, i] - kin.origin[:, j]
                regressor[:, j, col] = _cross2(lever, net_acc)
                regressor[:, j, col + 1] = _cross2(e_i, net_acc) + _cross2(
                    lever, acc_hx
                )
                regressor[:, j, col + 2] = _cross2(n_i, net_acc) + _cross2(
                    lever, acc_hy
                )
                regressor[:, j, col + 3] = kin.alpha[:, i]
        return regressor

    # -------------------------------------------------------------------- dynamics
    def mass_matrix(self, q: FloatArray, pi: FloatArray) -> FloatArray:
        """Return ``M(q)`` with shape ``(N, n, n)`` assembled from regressor columns."""
        zeros = np.zeros_like(q)
        no_gravity = np.zeros(2)
        columns = [
            self.regressor(q, zeros, np.tile(unit, (q.shape[0], 1)), no_gravity) @ pi
            for unit in np.eye(self.n_links)
        ]
        mass = np.stack(columns, axis=-1)
        ensure(
            bool(np.allclose(mass, np.swapaxes(mass, 1, 2), atol=1e-9)), "M symmetric"
        )
        return mass

    def bias(self, q: FloatArray, v: FloatArray, pi: FloatArray) -> FloatArray:
        """Return ``h(q, v) = C(q, v) v + g(q)`` with shape ``(N, n)``."""
        return self.regressor(q, v, np.zeros_like(q)) @ pi

    def drift_decomposition(
        self, q: FloatArray, v: FloatArray, pi: FloatArray
    ) -> DriftDecomposition:
        """Pointwise ZTCF/ZVCF accelerations: ``-M^{-1} h`` and ``-M^{-1} g``."""
        mass = self.mass_matrix(q, pi)
        ztcf = -np.linalg.solve(mass, self.bias(q, v, pi)[..., None])[..., 0]
        zvcf = -np.linalg.solve(mass, self.bias(q, np.zeros_like(q), pi)[..., None])[
            ..., 0
        ]
        return DriftDecomposition(ztcf=ztcf, zvcf=zvcf, velocity_part=ztcf - zvcf)

    def forward_dynamics(
        self, q: FloatArray, v: FloatArray, u: FloatArray, pi: FloatArray
    ) -> FloatArray:
        """Return joint accelerations ``M^{-1}(B u - h)`` with shape ``(N, n)``."""
        require(
            u.shape == (q.shape[0], self.n_inputs), "u must be (N, n_inputs)", u.shape
        )
        rhs = u @ self.input_matrix.T - self.bias(q, v, pi)
        return np.linalg.solve(self.mass_matrix(q, pi), rhs[..., None])[..., 0]

    def step(
        self, x: FloatArray, u: FloatArray, pi: FloatArray, dt: float
    ) -> FloatArray:
        """One RK4 step of the state ``x = (q, v)`` with zero-order-hold input."""
        require(dt > 0.0, "dt must be positive", dt)
        n = self.n_links

        def rate(state: FloatArray) -> FloatArray:
            q, v = state[:, :n], state[:, n:]
            return np.concatenate([v, self.forward_dynamics(q, v, u, pi)], axis=1)

        k1 = rate(x)
        k2 = rate(x + 0.5 * dt * k1)
        k3 = rate(x + 0.5 * dt * k2)
        k4 = rate(x + dt * k3)
        return x + (dt / 6.0) * sum(
            w * k for w, k in zip(_RK4_WEIGHTS, (k1, k2, k3, k4), strict=True)
        )

    def rollout(
        self, x0: FloatArray, u: FloatArray, pi: FloatArray, dt: float
    ) -> tuple[FloatArray, FloatArray]:
        """Integrate ``T`` steps for a batch of trials; returns ``(q, v)`` of ``T+1`` nodes."""
        require(x0.shape[1] == 2 * self.n_links, "x0 must be (K, 2 n_links)", x0.shape)
        require(
            u.ndim == 3 and u.shape[0] == x0.shape[0],
            "u must be (K, T, n_inputs)",
            u.shape,
        )
        states = [x0]
        for t in range(u.shape[1]):
            states.append(self.step(states[-1], u[:, t], pi, dt))
        trajectory = np.stack(states, axis=1)
        return trajectory[..., : self.n_links], trajectory[..., self.n_links :]

    def total_energy(self, q: FloatArray, v: FloatArray, pi: FloatArray) -> FloatArray:
        """Kinetic plus gravitational potential energy per sample."""
        kin = self._kinematics(q, np.zeros_like(q), np.zeros_like(q))
        kinetic = 0.5 * np.einsum("ni,nij,nj->n", v, self.mass_matrix(q, pi), v)
        params = pi.reshape(self.n_links, PLANAR_PARAMETERS_PER_BODY)
        mass_moment = params[None, :, 0:1] * kin.origin[:, :-1] + (
            params[None, :, 1:2] * kin.e + params[None, :, 2:3] * kin.n
        )
        potential = -np.einsum("nij,j->n", mass_moment, self.gravity)
        return kinetic + potential

    # --------------------------------------------------------------------- markers
    def marker_positions(self, q: FloatArray, markers: PlanarMarkerSet) -> FloatArray:
        """World positions ``(N, n_markers, 2)`` of attached markers."""
        require(
            bool(np.all(markers.link_index < self.n_links)), "marker link out of range"
        )
        kin = self._kinematics(q, np.zeros_like(q), np.zeros_like(q))
        idx = markers.link_index
        offsets = markers.offsets[None]
        return (
            kin.origin[:, idx]
            + offsets[..., 0:1] * kin.e[:, idx]
            + offsets[..., 1:2] * kin.n[:, idx]
        )

    def marker_jacobians(
        self, q: FloatArray, markers: PlanarMarkerSet
    ) -> tuple[FloatArray, FloatArray]:
        """Analytic Jacobians of marker positions w.r.t. joint angles and link lengths.

        Returns ``(d/dq, d/dl)`` each of shape ``(N, n_markers, 2, n_links)``.
        """
        kin = self._kinematics(q, np.zeros_like(q), np.zeros_like(q))
        positions = self.marker_positions(q, markers)
        joints = np.arange(self.n_links)
        distal = (joints[None, :] <= markers.link_index[:, None]).astype(np.float64)
        proximal = (joints[None, :] < markers.link_index[:, None]).astype(np.float64)
        radius = positions[:, :, None, :] - kin.origin[:, None, :-1, :]
        rotated = np.stack([-radius[..., 1], radius[..., 0]], axis=-1)
        jac_q = np.swapaxes(rotated * distal[None, :, :, None], 2, 3)
        jac_l = np.swapaxes(kin.e[:, None, :, :] * proximal[None, :, :, None], 2, 3)
        return jac_q, jac_l
