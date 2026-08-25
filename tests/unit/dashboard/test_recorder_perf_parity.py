"""Numerical parity tests for the recorder performance fixes (issue #8931).

These tests pin down that the vectorized / factorized implementations in
``_recorder_recording.py`` (induced accelerations via a single Cholesky
factorization instead of a full matrix inverse per frame) and
``_recorder_analysis.py`` (vectorized swing-plane wrench decomposition
instead of a per-frame loop) produce numerically identical output to the
original, straightforward formulations.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit
from src.shared.python.dashboard._recorder_analysis import _AnalysisMixin
from src.shared.python.dashboard._recorder_recording import _RecordingMixin
from src.shared.python.spatial_algebra.reference_frames import (
    ReferenceFrame,
    ReferenceFrameTransformer,
    SwingPlaneFrame,
    WrenchInFrame,
)


def _reference_induced_accelerations(
    M: np.ndarray, tau: np.ndarray, sources: list[int]
) -> dict[int, np.ndarray]:
    """Pre-fix reference: full inverse, one column read per source."""
    M_inv = np.linalg.inv(M)
    return {src_idx: M_inv[:, src_idx] * tau[src_idx] for src_idx in sources}


class _FakeRecorder(_RecordingMixin):
    """Minimal harness exercising only `_record_induced_accelerations`."""

    def __init__(self, nv: int, sources: list[int]) -> None:
        self.analysis_config = {"induced_accel_sources": sources}
        self.data = {
            "induced_accelerations": {src: np.zeros((1, nv)) for src in sources}
        }


class TestInducedAccelerationParity:
    """Parity between np.linalg.inv column-read and cho_factor/cho_solve."""

    def test_matches_full_inverse_reference(self) -> None:
        rng = np.random.default_rng(1234)
        nv = 6
        # Build a random SPD matrix (as the mass matrix contract requires).
        a = rng.normal(size=(nv, nv))
        M = a @ a.T + nv * np.eye(nv)
        tau = rng.normal(size=nv)
        sources = [0, 2, 5]

        expected = _reference_induced_accelerations(M, tau, sources)

        recorder = _FakeRecorder(nv, sources)
        recorder._record_induced_accelerations(0, tau, M)

        for src_idx in sources:
            np.testing.assert_allclose(
                recorder.data["induced_accelerations"][src_idx][0],
                expected[src_idx],
                atol=1e-10,
                rtol=1e-10,
            )

    def test_matches_full_inverse_reference_identity_mass(self) -> None:
        nv = 4
        M = np.eye(nv)
        tau = np.array([1.0, -2.0, 3.5, 0.25])
        sources = [0, 1, 2, 3]

        expected = _reference_induced_accelerations(M, tau, sources)

        recorder = _FakeRecorder(nv, sources)
        recorder._record_induced_accelerations(0, tau, M)

        for src_idx in sources:
            np.testing.assert_allclose(
                recorder.data["induced_accelerations"][src_idx][0],
                expected[src_idx],
                atol=1e-12,
                rtol=1e-12,
            )

    def test_no_op_when_source_not_in_buffer(self) -> None:
        nv = 3
        M = np.eye(nv)
        tau = np.array([1.0, 2.0, 3.0])
        # analysis_config requests source 1, but no buffer was allocated for it.
        recorder = _FakeRecorder(nv, sources=[])
        recorder.analysis_config = {"induced_accel_sources": [1]}
        recorder._record_induced_accelerations(0, tau, M)
        assert recorder.data["induced_accelerations"] == {}


def _reference_wrench_decomposition(
    forces: np.ndarray, moments: np.ndarray, fsp: SwingPlaneFrame | None, n: int
) -> dict[str, np.ndarray]:
    """Pre-fix reference: per-frame WrenchInFrame + get_swing_plane_decomposition."""
    transformer = ReferenceFrameTransformer()
    if fsp is not None:
        transformer.set_swing_plane(fsp)

    decomp_keys = [
        "force_in_plane",
        "force_out_of_plane",
        "force_along_grip",
        "torque_in_plane",
        "torque_out_of_plane",
        "torque_about_grip",
    ]

    decompositions = []
    for i in range(n):
        wrench = WrenchInFrame(
            force=forces[i],
            torque=moments[i],
            frame=ReferenceFrame.GLOBAL,
            body_name="ground",
        )
        if fsp is not None:
            decomp = transformer.get_swing_plane_decomposition(wrench)
        else:
            decomp = dict.fromkeys(decomp_keys, 0.0)
        decompositions.append(decomp)

    return {k: np.array([d[k] for d in decompositions]) for k in decomp_keys}


class TestWrenchDecompositionParity:
    """Parity between the per-frame loop and the vectorized decomposition."""

    @pytest.fixture
    def swing_plane(self) -> SwingPlaneFrame:
        # An orthonormal-ish, non-axis-aligned plane so all dot products
        # actually exercise nontrivial coefficients.
        normal = np.array([0.0, 0.0, 1.0])
        in_plane_x = np.array([1.0, 0.0, 0.0])
        in_plane_y = np.array([0.0, 1.0, 0.0])
        grip_axis = np.array([0.7071, 0.0, 0.7071])
        return SwingPlaneFrame(
            origin=np.zeros(3),
            normal=normal,
            in_plane_x=in_plane_x,
            in_plane_y=in_plane_y,
            grip_axis=grip_axis,
        )

    def test_matches_per_frame_reference(self, swing_plane: SwingPlaneFrame) -> None:
        rng = np.random.default_rng(42)
        n = 250
        forces = rng.normal(size=(n, 3)) * 100.0
        moments = rng.normal(size=(n, 3)) * 10.0

        expected = _reference_wrench_decomposition(forces, moments, swing_plane, n)
        actual = _AnalysisMixin._compute_wrench_decomposition(
            forces, moments, swing_plane, n
        )

        for key in expected:
            np.testing.assert_allclose(
                actual[key], expected[key], atol=1e-10, rtol=1e-10
            )

    def test_matches_reference_when_no_swing_plane(self) -> None:
        n = 10
        forces = np.ones((n, 3))
        moments = np.ones((n, 3))

        expected = _reference_wrench_decomposition(forces, moments, None, n)
        actual = _AnalysisMixin._compute_wrench_decomposition(forces, moments, None, n)

        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
