"""Shaft springs and dampers in the Pinocchio force adapter (#15150 Track Z, Z6).

The flexible shaft's springs and dampers are passive, so they belong in the
drift (ZTCF) and in the inverse-dynamics right-hand side, never in the
player's torques. The adapter's bias therefore carries ``-tau_spring``: a
deflected, moving shaft at the same state differs from a straight, still one
by exactly ``k (q - q_ref) + c v`` on the chain rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

SEEDS = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
ATOL = 1e-9


def _adapter(shaft: str):
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.shared.python.motion_matching.execution.spec_builder import (
        with_shaft_model,
    )

    doc = json.loads((SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes())
    return PinocchioForceAdapter(
        json.dumps(with_shaft_model(doc, "driver", shaft)).encode()
    )


@pytest.fixture(scope="module")
def chain():
    return _adapter("chain")


def _shaft_slots(adapter) -> np.ndarray:
    names = adapter.coordinate_order
    springs = adapter.shaft_springs()
    return np.array([names.index(n) for n in springs], dtype=int)


def _springs(adapter) -> np.ndarray:
    """``(k, rest, c)`` rows aligned with :func:`_shaft_slots`."""
    return np.array(list(adapter.shaft_springs().values()), dtype=float).T


def _state(adapter) -> tuple[np.ndarray, np.ndarray]:
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        q = np.zeros(adapter.nv)
        q[: rec["q"].shape[1]] = rec["q"][400]
        v = np.zeros(adapter.nv)
        v[: rec["v"].shape[1]] = rec["v"][400]
    return q, v


def _nle(adapter, q, v) -> np.ndarray:
    return np.asarray(
        adapter._pin.nonLinearEffects(adapter._model, adapter._data, q, v)
    )


def test_bias_carries_the_spring_and_damper_on_the_chain_rows(chain) -> None:
    slots = _shaft_slots(chain)
    k, rest, c = _springs(chain)
    q, v = _state(chain)
    q[slots], v[slots] = rest + 0.02, 0.5
    _, bias = chain.compute_mass_and_bias(q, v)
    extra = bias - _nle(chain, q, v)
    np.testing.assert_allclose(extra[slots], k * 0.02 + c * 0.5, rtol=1e-9)
    others = np.setdiff1d(np.arange(chain.nv), slots)
    np.testing.assert_allclose(extra[others], 0.0, atol=ATOL)
    q[slots], v[slots] = rest, 0.0
    _, bias = chain.compute_mass_and_bias(q, v)
    np.testing.assert_allclose(bias - _nle(chain, q, v), 0.0, atol=ATOL)


def test_inverse_dynamics_includes_the_spring(chain) -> None:
    slots = _shaft_slots(chain)
    k, rest, _ = _springs(chain)
    q, _ = _state(chain)
    q[slots] = rest - 0.01
    zero = np.zeros(chain.nv)
    tau = chain.compute_inverse_dynamics(q, zero, zero)
    mass, bias = chain.compute_mass_and_bias(q, zero)
    np.testing.assert_allclose(tau, mass @ zero + bias, atol=ATOL)
    rnea = np.asarray(chain._pin.rnea(chain._model, chain._data, q, zero, zero))
    np.testing.assert_allclose((tau - rnea)[slots], -k * 0.01, rtol=1e-9)


def test_rigid_club_bias_is_unchanged() -> None:
    rigid = _adapter("rigid")
    assert rigid.shaft_springs() == {}
    q, v = _state(rigid)
    _, bias = rigid.compute_mass_and_bias(q, v)
    nle = np.asarray(rigid._pin.nonLinearEffects(rigid._model, rigid._data, q, v))
    np.testing.assert_allclose(bias, nle, atol=0.0)


def test_drift_returns_a_bent_still_shaft_toward_rest(chain) -> None:
    """ZTCF (tau = 0) of a bent shaft accelerates it back: the spring is drift."""
    from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider
    from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
        constrained_acceleration_at,
    )

    slots = _shaft_slots(chain)
    _, rest, _ = _springs(chain)
    q, _ = _state(chain)
    q[slots] = rest + 0.05
    zero = np.zeros(chain.nv)
    provider = AdapterConstrainedProvider(chain)
    bent_a, _ = constrained_acceleration_at(provider, q, zero, zero)
    q[slots] = rest
    rest_a, _ = constrained_acceleration_at(provider, q, zero, zero)
    assert float(np.dot(bent_a[slots] - rest_a[slots], np.ones(len(slots)))) < 0.0


@pytest.mark.requires_mujoco
def test_mujoco_adapter_matches_pinocchio_spring_and_leaves_the_shaft_unactuated(
    chain,
) -> None:
    """The kinetic match runs on MuJoCo: same passive law, no shaft actuator."""
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.execution.spec_builder import (
        with_shaft_model,
    )
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MujocoForceAdapter,
    )

    doc = json.loads((SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes())
    mj = MujocoForceAdapter(
        json.dumps(with_shaft_model(doc, "driver", "chain")).encode()
    )
    names = mj.coordinate_order
    springs = chain.shaft_springs()
    slots = np.array([names.index(n) for n in springs], dtype=int)
    assert not set(slots) & set(mj.actuated_indices)
    k, rest, c = _springs(chain)
    q, v = np.zeros(mj.nv), np.zeros(mj.nv)
    q[slots], v[slots] = rest + 0.02, 0.5
    _, bias = mj.compute_mass_and_bias(q, v)
    _, raw = mj._mass_and_bias(q, v)
    np.testing.assert_allclose((bias - raw)[slots], k * 0.02 + c * 0.5, rtol=1e-9)
    others = np.setdiff1d(np.arange(mj.nv), slots)
    np.testing.assert_allclose((bias - raw)[others], 0.0, atol=0.0)
