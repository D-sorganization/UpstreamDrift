"""JaxSim ``canonical-v2`` round-trip (#14936 follow-up)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.engines.test_jaxsim_backend import _backend
from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.canonical_state import CONVENTION_TAG_V2

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
