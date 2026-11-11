"""JaxSim ``canonical-v2`` round-trip (#14936 follow-up)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.engines.test_jaxsim_backend import _backend
from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.canonical_state import (
    CONVENTION_TAG_V2,
    CanonicalState,
)

pytestmark = pytest.mark.unit


def test_jaxsim_canonical_state_roundtrip_spinning_translating_base() -> None:
    backend = _backend()

    rng = np.random.default_rng(14936)
    quat = se3.quat_exp(rng.normal(scale=0.35, size=3))
    q = np.concatenate([rng.normal(size=3), quat, [0.2]])
    v = np.concatenate(
        [
            rng.normal(size=3) * 2.0,
            rng.normal(size=3) * 1.5,
            [-0.4],
        ]
    )
    from src.shared.python.pose_interchange.canonical_state import CanonicalState

    a = rng.normal(size=v.shape[0]) * 0.75
    seed = CanonicalState(q=q, v=v, a=a, t=0.5)
    assert seed.convention == CONVENTION_TAG_V2

    backend.set_canonical_state(seed)
    recovered = backend.get_canonical_state()

    np.testing.assert_allclose(recovered.q, seed.q, atol=1.0e-12)
    np.testing.assert_allclose(recovered.v, seed.v, atol=1.0e-12)
    np.testing.assert_allclose(recovered.a, seed.a, atol=1.0e-12)
    assert recovered.t == pytest.approx(seed.t)


def _seed_with_nonzero_acceleration() -> CanonicalState:
    rng = np.random.default_rng(14969)
    quat = se3.quat_exp(rng.normal(scale=0.2, size=3))
    q = np.concatenate([rng.normal(size=3) * 0.1, quat, [0.3]])
    v = np.concatenate([rng.normal(size=3), rng.normal(size=3), [0.1]])
    a = rng.normal(size=v.shape[0]) * 1.25
    return CanonicalState(q=q, v=v, a=a, t=0.25)


def test_set_state_clears_cached_canonical_acceleration() -> None:
    backend = _backend()
    seed = _seed_with_nonzero_acceleration()
    backend.set_canonical_state(seed)
    np.testing.assert_allclose(backend.get_canonical_state().a, seed.a, atol=1.0e-12)

    q_new = seed.q.copy()
    q_new[0] += 0.05
    v_new = seed.v * 0.5
    backend.set_state(q_new, v_new)

    q_back, v_back = backend.get_state()
    np.testing.assert_allclose(q_back, q_new, atol=1.0e-12)
    np.testing.assert_allclose(v_back, v_new, atol=1.0e-12)
    np.testing.assert_allclose(backend.get_canonical_state().a, 0.0, atol=1.0e-12)


def test_step_clears_cached_canonical_acceleration() -> None:
    backend = _backend()
    seed = _seed_with_nonzero_acceleration()
    backend.set_canonical_state(seed)
    np.testing.assert_allclose(backend.get_canonical_state().a, seed.a, atol=1.0e-12)

    backend.step(dt=0.01)

    recovered = backend.get_canonical_state()
    np.testing.assert_allclose(recovered.a, 0.0, atol=1.0e-12)


def test_set_canonical_state_rejects_non_v2_convention() -> None:
    backend = _backend()
    seed = _seed_with_nonzero_acceleration()
    foreign = SimpleNamespace(
        q=seed.q,
        v=seed.v,
        a=seed.a,
        t=seed.t,
        convention="canonical-v1",
    )
    with pytest.raises(ValueError, match="convention must be 'canonical-v2'"):
        backend.set_canonical_state(foreign)  # type: ignore[arg-type]
