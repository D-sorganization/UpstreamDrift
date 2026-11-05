"""Cross-module ZVCF consistency on the planar oracle (issues #11553, #14494).

ZVCF is the control component ``M(q)^-1 tau`` (owner decision, #14494). The
zero-velocity, zero-torque acceleration ``-M(q)^-1 Y(q, 0, 0) pi`` keeps its
value under ``zero_velocity_zero_torque_acceleration``; the planar oracle's
``drift_decomposition`` still reports it as ``zvcf``.
:class:`motion_matching.counterfactual.AccelerationDecomposition` must report
the same values as :mod:`simulation_backends.ztcf_zvcf` for the same state.
The control-preserved variant keeps its own name and is never called ZVCF.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.planar_chain import PlanarChain
from src.shared.python.motion_matching.counterfactual import (
    AccelerationDecomposition,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    zero_velocity_control_preserved_acceleration,
    zero_velocity_zero_torque_acceleration,
    ztcf_acceleration,
    zvcf_acceleration,
)

pytestmark = pytest.mark.unit

RTOL = 1e-10
ATOL = 1e-12

Q = np.array([0.7, -0.4])
V = np.array([2.3, -1.1])
TAU = np.array([4.0, -1.5])


class _PlanarOracleProvider:
    """``EquationsOfMotionProvider`` view of the planar double pendulum at fixed ``pi``."""

    def __init__(self) -> None:
        m1, m2, lc1, lc2, i1c, i2c = 1.3, 0.7, 0.45, 0.25, 0.08, 0.02
        self.chain = PlanarChain(
            link_lengths=np.array([0.9, 0.6]), actuated=np.array([True, True])
        )
        self.pi = np.array(
            [m1, m1 * lc1, 0.0, i1c + m1 * lc1**2, m2, m2 * lc2, 0.0, i2c + m2 * lc2**2]
        )

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return self.chain.mass_matrix(q[None, :], self.pi)[0]

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.chain.bias(q[None, :], v[None, :], self.pi)[0]


@pytest.fixture
def provider() -> _PlanarOracleProvider:
    return _PlanarOracleProvider()


def _oracle_zero_velocity_zero_torque(provider: _PlanarOracleProvider) -> np.ndarray:
    drift = provider.chain.drift_decomposition(Q[None, :], V[None, :], provider.pi)
    return drift.zvcf[0]


def test_zero_velocity_zero_torque_matches_oracle(
    provider: _PlanarOracleProvider,
) -> None:
    """The renamed gravity-only term is the projection ``-M^-1 Y(q,0,0) pi``."""
    np.testing.assert_allclose(
        zero_velocity_zero_torque_acceleration(provider, Q),
        _oracle_zero_velocity_zero_torque(provider),
        rtol=RTOL,
        atol=ATOL,
    )


def test_canonical_zvcf_is_mass_inverse_torque(provider: _PlanarOracleProvider) -> None:
    """``zvcf_acceleration`` is the control component ``M^-1 tau``."""
    np.testing.assert_allclose(
        zvcf_acceleration(provider, Q, TAU),
        np.linalg.solve(provider.mass_matrix(Q), TAU),
        rtol=RTOL,
        atol=ATOL,
    )


def test_decomposition_zvcf_from_documented_components(
    provider: _PlanarOracleProvider,
) -> None:
    """Built from its documented components, the decomposition's ZVCF is canonical.

    ``a_grav`` is the zero-velocity, zero-control acceleration; ``a_drift`` the
    velocity-dependent remainder of the ZTCF; ``a_ctrl = M^-1 tau``.
    """
    a_grav = zero_velocity_zero_torque_acceleration(provider, Q)
    a_drift = ztcf_acceleration(provider, Q, V) - a_grav
    a_ctrl = np.linalg.solve(provider.mass_matrix(Q), TAU)

    decomp = AccelerationDecomposition(a_grav=a_grav, a_drift=a_drift, a_ctrl=a_ctrl)

    np.testing.assert_allclose(
        decomp.zvcf, zvcf_acceleration(provider, Q, TAU), rtol=RTOL, atol=ATOL
    )


def test_both_entry_points_agree_for_same_state(
    provider: _PlanarOracleProvider,
) -> None:
    """``AccelerationDecomposition.from_dynamics`` delegates to the canonical module."""
    decomp = AccelerationDecomposition.from_dynamics(provider, Q, V, TAU)

    np.testing.assert_allclose(
        decomp.zvcf, zvcf_acceleration(provider, Q, TAU), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        decomp.zero_velocity_zero_torque,
        _oracle_zero_velocity_zero_torque(provider),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        decomp.ztcf, ztcf_acceleration(provider, Q, V), rtol=RTOL, atol=ATOL
    )
    forward = provider.chain.forward_dynamics(
        Q[None, :], V[None, :], TAU[None, :], provider.pi
    )[0]
    np.testing.assert_allclose(decomp.total_accel, forward, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(decomp.ztcf + decomp.zvcf, forward, rtol=RTOL, atol=ATOL)


def test_control_preserved_variant_keeps_its_own_name(
    provider: _PlanarOracleProvider,
) -> None:
    """The ``v = 0`` acceleration that keeps ``tau`` is not ZVCF."""
    decomp = AccelerationDecomposition.from_dynamics(provider, Q, V, TAU)
    expected = zero_velocity_control_preserved_acceleration(provider, Q, TAU)

    np.testing.assert_allclose(
        decomp.zero_velocity_control_preserved, expected, rtol=RTOL, atol=ATOL
    )
    assert not np.allclose(decomp.zvcf, expected)
