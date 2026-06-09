"""Cross-engine canonical-v2 velocity-layout conformance (issue #7144).

Regression guard for the BLOCKER where the Pinocchio reference adapter
documented and implemented its native base motion as ``[angular; linear]`` and
swapped the first six velocity components, while the spec
(``docs/conventions/canonical-v2.md`` §2) and the MuJoCo adapter use
``[linear; angular]``. The swap is self-inverse, so each adapter's own
round-trip passed while every *cross-engine* exchange silently transposed base
linear and angular velocity.

The canonical state below uses distinguishable linear ``[1, 2, 3]`` and angular
``[10, 20, 30]`` blocks so a swap is detectable. Both adapters must agree that a
canonical state maps to natives that preserve the linear block at indices 0..2.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.adapters.mujoco import (
    CanonicalV2State as MujocoCanonicalV2State,
)
from src.shared.python.pose_interchange.adapters.mujoco import MujocoAdapter
from src.shared.python.pose_interchange.adapters.pinocchio_reference import (
    CanonicalV2State as PinocchioCanonicalV2State,
)
from src.shared.python.pose_interchange.adapters.pinocchio_reference import (
    PinocchioReferenceAdapter,
)
from src.shared.python.pose_interchange.canonical_layout import ANGULAR, LINEAR

pytestmark = pytest.mark.unit

# Canonical-v2 state with distinguishable base blocks plus one joint.
_Q = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
_V = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 0.7], dtype=np.float64)
_A = np.array([4.0, 5.0, 6.0, 40.0, 50.0, 60.0, 0.9], dtype=np.float64)

_LINEAR_EXPECTED = np.array([1.0, 2.0, 3.0])
_ANGULAR_EXPECTED = np.array([10.0, 20.0, 30.0])


class _IdentityRneaBackend:
    def fk(self, q: np.ndarray, frame_name: str) -> np.ndarray:  # pragma: no cover
        return np.eye(4)

    def jacobian(
        self, q: np.ndarray, frame_name: str
    ) -> np.ndarray:  # pragma: no cover
        return np.zeros((6, q.shape[0] - 1))

    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        return a.copy()

    def aba(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return tau.copy()


def test_pinocchio_native_velocity_keeps_linear_first() -> None:
    adapter = PinocchioReferenceAdapter(_IdentityRneaBackend())
    canonical = PinocchioCanonicalV2State(q=_Q.copy(), v=_V.copy(), a=_A.copy())

    native = adapter.from_canonical_v2(canonical)

    # Pinocchio free-flyer motion layout equals canonical: [linear; angular].
    np.testing.assert_allclose(native.v[LINEAR], _LINEAR_EXPECTED)
    np.testing.assert_allclose(native.v[ANGULAR], _ANGULAR_EXPECTED)
    np.testing.assert_allclose(native.a[LINEAR], _A[LINEAR])
    np.testing.assert_allclose(native.a[ANGULAR], _A[ANGULAR])

    recovered = adapter.to_canonical_v2(native)
    np.testing.assert_allclose(recovered.v, _V)
    np.testing.assert_allclose(recovered.a, _A)


def test_mujoco_native_velocity_keeps_linear_first() -> None:
    adapter = MujocoAdapter()
    canonical = MujocoCanonicalV2State(q=_Q.copy(), v=_V.copy(), a=_A.copy())

    native = adapter.from_canonical_v2(canonical)

    np.testing.assert_allclose(native.qvel[LINEAR], _LINEAR_EXPECTED)
    np.testing.assert_allclose(native.qvel[ANGULAR], _ANGULAR_EXPECTED)


def test_mujoco_and_pinocchio_agree_on_canonical_velocity() -> None:
    """A canonical state handed to either engine yields the same native linear
    block — the cross-engine exchange that was corrupted before the fix."""

    pin = PinocchioReferenceAdapter(_IdentityRneaBackend())
    muj = MujocoAdapter()

    pin_native = pin.from_canonical_v2(
        PinocchioCanonicalV2State(q=_Q.copy(), v=_V.copy(), a=_A.copy())
    )
    muj_native = muj.from_canonical_v2(
        MujocoCanonicalV2State(q=_Q.copy(), v=_V.copy(), a=_A.copy())
    )

    np.testing.assert_allclose(pin_native.v[LINEAR], muj_native.qvel[LINEAR])
    np.testing.assert_allclose(pin_native.v[ANGULAR], muj_native.qvel[ANGULAR])

    # Round-trip a MuJoCo-produced canonical state through Pinocchio.
    canonical_from_mujoco = muj.to_canonical_v2(muj_native)
    pin_from_mujoco = pin.from_canonical_v2(
        PinocchioCanonicalV2State(
            q=canonical_from_mujoco.q,
            v=canonical_from_mujoco.v,
            a=canonical_from_mujoco.a,
        )
    )
    np.testing.assert_allclose(pin_from_mujoco.v[LINEAR], _LINEAR_EXPECTED)
    np.testing.assert_allclose(pin_from_mujoco.v[ANGULAR], _ANGULAR_EXPECTED)
