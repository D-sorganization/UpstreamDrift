"""Live JaxSim canonical-v2 angular-frame round-trip (#14936)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.jaxsim import JaxSimBackend
from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.canonical_state import CanonicalState

pytestmark = [pytest.mark.unit, pytest.mark.requires_jaxsim]


def test_live_jaxsim_canonical_state_roundtrip_non_identity_rotation() -> None:
    pytest.importorskip("jaxsim")

    backend = JaxSimBackend()
    backend.load_from_path("tests/fixtures/jaxsim/single_link.sdf")
    q0, _ = backend.get_state()
    n_j = q0.shape[0] - 7

    quat = se3.quat_exp(np.array([0.25, 0.5, -0.15]))
    omega_body = np.array([2.0, -1.0, 1.25])
    q = np.concatenate([np.zeros(3), quat, q0[7:]])
    v = np.concatenate(
        [np.array([0.5, -0.25, 0.1]), omega_body, np.zeros(n_j, dtype=np.float64)]
    )
    seed = CanonicalState(q=q, v=v, a=np.zeros_like(v), t=0.0)

    backend.set_canonical_state(seed)
    recovered = backend.get_canonical_state()

    np.testing.assert_allclose(recovered.q, seed.q, atol=1.0e-9)
    np.testing.assert_allclose(recovered.v, seed.v, atol=1.0e-9)
