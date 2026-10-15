"""The torque allocator consumes only canonical closure Jacobians (NSX-13, R8).

The allocation must be invariant to the engine's native closure convention:
row order, frame, reference point and sign.  A fixture adapter serves the same
physical closure in several native conventions; the allocator output must not
change.  Engines that do not implement the closed-loop frame protocol are a
legacy passthrough, so the unconverted fixture is the baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.closure_convention import (
    NativeClosureConvention,
    from_canonical_closure,
)
from src.shared.python.loop_nullspace.snapshot import canonical_closure_jacobian
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    EngineType,
    MultiEngineTorqueAllocator,
    PinocchioSyntheticForceAdapter,
)

pytestmark = pytest.mark.unit

NV = 44


def _rot(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


GRIP_R = _rot((0.3, -0.5, 0.8), 0.7)
GRIP_P = np.array([0.4, -0.2, 1.1])


class ConventionedFixture(PinocchioSyntheticForceAdapter):
    """Synthetic closure served in an arbitrary native convention."""

    def __init__(
        self,
        native: NativeClosureConvention,
        native_rot: np.ndarray,
        native_point: np.ndarray,
    ) -> None:
        super().__init__(nv=NV, n_spheres=6)
        self.native_closure_convention = native
        self._native_rot = native_rot
        self._native_point = native_point

    @property
    def engine_type(self) -> EngineType:
        return EngineType.PINOCCHIO

    def native_closure_frame(self, q):
        return self._native_rot, self._native_point

    def closure_wrench_frame(self, q):
        return GRIP_R, GRIP_P

    def compute_grip_jacobian(self, q):
        canonical = PinocchioSyntheticForceAdapter.compute_grip_jacobian(self, q)
        native, _ = from_canonical_closure(
            canonical,
            np.zeros(6),
            native=self.native_closure_convention,
            native_frame_rotation=self._native_rot,
            grip_frame_rotation=GRIP_R,
            native_reference_point=self._native_point,
            grip_reference_point=GRIP_P,
        )
        return native


CONVENTIONS = {
    "pinocchio_like": (
        NativeClosureConvention("lin_ang", "LOCAL", "frame_origin", 1),
        _rot((1, 0, 0), 0.4),
        np.array([0.1, 0.2, 0.3]),
    ),
    "mujoco_like": (
        NativeClosureConvention("ang_lin", "WORLD", "site", -1),
        np.eye(3),
        np.array([-0.3, 0.1, 0.9]),
    ),
    "drake_like": (
        NativeClosureConvention("ang_lin", "LOCAL", "frame_origin", -1),
        _rot((0, 1, 1), -1.1),
        GRIP_P,
    ),
}


def _state(nv: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12558)
    return rng.normal(size=nv), 0.1 * rng.normal(size=nv), 0.5 * rng.normal(size=nv)


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_canonical_jacobian_is_independent_of_the_native_convention(name) -> None:
    q, _, _ = _state(NV)
    adapter = ConventionedFixture(*CONVENTIONS[name])
    legacy = PinocchioSyntheticForceAdapter(nv=NV, n_spheres=6)
    native = adapter.compute_grip_jacobian(q)
    canonical = canonical_closure_jacobian(adapter, q)
    assert np.abs(native - legacy.compute_grip_jacobian(q)).max() > 1e-3  # differs
    np.testing.assert_allclose(
        canonical, legacy.compute_grip_jacobian(q), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_allocator_output_is_invariant_to_the_native_convention(name) -> None:
    q, v, a = _state(NV)
    baseline = MultiEngineTorqueAllocator(
        PinocchioSyntheticForceAdapter(nv=NV, n_spheres=6)
    ).allocate_frame(q, v, a)
    assert np.abs(baseline.lambda_grip).max() > 0.0  # non-trivial allocation
    other = MultiEngineTorqueAllocator(
        ConventionedFixture(*CONVENTIONS[name])
    ).allocate_frame(q, v, a)
    for field in ("tau_actuated", "f_ground", "lambda_grip", "delta_tau_root"):
        np.testing.assert_allclose(
            getattr(other, field), getattr(baseline, field), rtol=1e-9, atol=1e-9
        )
    assert other.success == baseline.success


def test_legacy_adapter_without_frame_protocol_passes_through() -> None:
    q, _, _ = _state(NV)
    legacy = PinocchioSyntheticForceAdapter(nv=NV, n_spheres=6)
    np.testing.assert_array_equal(
        canonical_closure_jacobian(legacy, q), legacy.compute_grip_jacobian(q)
    )
