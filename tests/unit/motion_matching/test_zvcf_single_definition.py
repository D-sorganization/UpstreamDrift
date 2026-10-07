"""Cross-module ZVCF consistency on the planar oracle (issue #11553).

The methods reference (``model_aware_matching.tex``, eq. ``eq:zvcf``) defines
ZVCF as velocity *and* control set to zero:
``a_ZVCF = -M(q)^-1 Y(q, 0, 0) pi = -M(q)^-1 h(q, 0)`` in free flight.  The
canonical implementation is
:func:`simulation_backends.ztcf_zvcf.zvcf_acceleration`;
:class:`motion_matching.counterfactual.AccelerationDecomposition` must report
the same value for the same state.  The control-preserved variant keeps its own
name and is never called ZVCF.
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
    """``DynamicsProvider`` view of the planar double pendulum at fixed ``pi``."""

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


def _oracle_zvcf(provider: _PlanarOracleProvider) -> np.ndarray:
    drift = provider.chain.drift_decomposition(Q[None, :], V[None, :], provider.pi)
    return drift.zvcf[0]


def test_canonical_zvcf_matches_oracle(provider: _PlanarOracleProvider) -> None:
    """``zvcf_acceleration`` is the regressor projection ``-M^-1 Y(q,0,0) pi``."""
    np.testing.assert_allclose(
        zvcf_acceleration(provider, Q), _oracle_zvcf(provider), rtol=RTOL, atol=ATOL
    )


def test_decomposition_zvcf_from_documented_components(
    provider: _PlanarOracleProvider,
) -> None:
    """Built from its documented components, the decomposition's ZVCF is canonical.

    ``a_grav`` is the zero-velocity, zero-control acceleration; ``a_drift`` the
    velocity-dependent remainder of the ZTCF; ``a_ctrl = M^-1 tau``.
    """
    a_grav = zvcf_acceleration(provider, Q)
    a_drift = ztcf_acceleration(provider, Q, V) - a_grav
    a_ctrl = np.linalg.solve(provider.mass_matrix(Q), TAU)

    decomp = AccelerationDecomposition(a_grav=a_grav, a_drift=a_drift, a_ctrl=a_ctrl)

    np.testing.assert_allclose(
        decomp.zvcf, zvcf_acceleration(provider, Q), rtol=RTOL, atol=ATOL
    )


def test_both_entry_points_agree_for_same_state(
    provider: _PlanarOracleProvider,
) -> None:
    """``AccelerationDecomposition.from_dynamics`` delegates to the canonical module."""
    decomp = AccelerationDecomposition.from_dynamics(provider, Q, V, TAU)

    np.testing.assert_allclose(
        decomp.zvcf, zvcf_acceleration(provider, Q), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        decomp.zvcf, _oracle_zvcf(provider), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        decomp.ztcf, ztcf_acceleration(provider, Q, V), rtol=RTOL, atol=ATOL
    )
    forward = provider.chain.forward_dynamics(
        Q[None, :], V[None, :], TAU[None, :], provider.pi
    )[0]
    np.testing.assert_allclose(decomp.total_accel, forward, rtol=RTOL, atol=ATOL)


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
