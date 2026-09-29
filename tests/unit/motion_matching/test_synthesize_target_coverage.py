"""Coverage tests for ``synthesize_target_from_coefficients``."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.cost import SimOutput
from src.shared.python.motion_matching.synthesize_target_from_coefficients import (
    EngineSimulator,
    SynthOptions,
    synthesize_target_from_coefficients,
)
from src.shared.python.motion_matching.target import AlignOptions


def _make_simout(n: int = 301, t_end: float = 0.3) -> SimOutput:
    time = np.linspace(0.0, t_end, n)
    speed_curve = np.sin(np.pi * time / t_end)
    butt = np.zeros((n, 3))
    clubhead = np.column_stack([speed_curve * 0.5, np.zeros(n), np.zeros(n)])
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return SimOutput(time=time, butt=butt, clubhead=clubhead, club_quat=quat)


class _Fake:
    """Engine simulator returning a canned SimOut."""

    def __call__(
        self,
        theta: np.ndarray,
        *,
        sample_rate_hz: float,
        simulation_time_s: float,
    ) -> SimOutput:
        n = int(round(sample_rate_hz * simulation_time_s)) + 1
        return _make_simout(n=n, t_end=simulation_time_s)


def test_engine_protocol_runtime_check() -> None:
    """Pin: ``EngineSimulator`` is a runtime-checkable protocol."""
    assert isinstance(_Fake(), EngineSimulator)


def test_basic_roundtrip() -> None:
    """Pin: valid theta + engine produces a validated ClubTarget."""
    theta = np.zeros(7 * 3, dtype=np.float64)
    out = synthesize_target_from_coefficients(theta, _Fake())
    assert out.time.shape[0] >= 2


def test_engine_must_be_callable() -> None:
    """Pin: non-callable engine raises TypeError."""
    with pytest.raises(TypeError, match="EngineSimulator"):
        synthesize_target_from_coefficients(np.zeros(7), 42)  # type: ignore[arg-type]


def test_theta_bounds_violation() -> None:
    """Pin: out-of-bound coefficient is rejected pre-engine."""
    bad = np.zeros(7)
    bad[0] = 5000.0  # exceeds A bound 1000
    with pytest.raises(ValueError, match="coefficient A"):
        synthesize_target_from_coefficients(bad, _Fake())


def test_theta_bad_length_rejected() -> None:
    """Pin: non-multiple-of-7 length rejected."""
    with pytest.raises(ValueError, match="multiple of 7"):
        synthesize_target_from_coefficients(np.zeros(5), _Fake())


def test_engine_must_return_simout() -> None:
    """Pin: engine returning non-SimOutput raises TypeError."""

    def bad_engine(theta, *, sample_rate_hz, simulation_time_s):
        return "not a simout"

    with pytest.raises(TypeError, match="must return SimOut"):
        synthesize_target_from_coefficients(np.zeros(7), bad_engine)


def test_engine_simout_must_have_time() -> None:
    """Pin: SimOut with ``time=None`` is rejected."""

    def bad_engine(theta, *, sample_rate_hz, simulation_time_s):
        n = 301
        return SimOutput(
            time=None,
            butt=np.zeros((n, 3)),
            clubhead=np.zeros((n, 3)),
            club_quat=np.tile([1, 0, 0, 0], (n, 1)),
        )

    with pytest.raises(ValueError, match="SimOut.time is required"):
        synthesize_target_from_coefficients(np.zeros(7), bad_engine)


def test_quat_normalisation_and_sign_flip() -> None:
    """Pin: non-unit, w<0 quaternions are normalised and sign-flipped."""

    def neg_w_engine(theta, *, sample_rate_hz, simulation_time_s):
        n = int(round(sample_rate_hz * simulation_time_s)) + 1
        time = np.linspace(0.0, simulation_time_s, n)
        quat = np.tile([-2.0, 0.0, 0.0, 0.0], (n, 1))
        butt = np.zeros((n, 3))
        clubhead = np.column_stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)])
        return SimOutput(time=time, butt=butt, clubhead=clubhead, club_quat=quat)

    out = synthesize_target_from_coefficients(np.zeros(7), neg_w_engine)
    # After normalisation + flip, w should be +1.0.
    assert np.allclose(out.club_quat[:, 0], 1.0)


def test_quat_finite_required() -> None:
    """Pin: NaN quaternion rejected."""

    def nan_quat_engine(theta, *, sample_rate_hz, simulation_time_s):
        n = 301
        time = np.linspace(0.0, simulation_time_s, n)
        quat = np.full((n, 4), np.nan)
        return SimOutput(
            time=time,
            butt=np.zeros((n, 3)),
            clubhead=np.column_stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)]),
            club_quat=quat,
        )

    with pytest.raises(ValueError, match="finite"):
        synthesize_target_from_coefficients(np.zeros(7), nan_quat_engine)


def test_align_opts_overrides_impact() -> None:
    """Pin: ``align_opts.impact_target_t_s`` selects the impact frame."""
    align = AlignOptions(impact_target_t_s=0.10)
    out = synthesize_target_from_coefficients(np.zeros(7), _Fake(), align_opts=align)
    # impact_idx is 1-based; sample 0.1s on a 1kHz grid is index ~100,
    # plus 1 for 1-based. Allow some tolerance.
    expected = int(np.argmin(np.abs(out.time - 0.10))) + 1
    assert out.impact_idx == expected


def test_add_noise_branch() -> None:
    """Pin: ``add_noise=True`` perturbs positions reproducibly."""
    opts = SynthOptions(add_noise=True, noise_sigma_m=1e-4, noise_seed=42)
    out1 = synthesize_target_from_coefficients(np.zeros(7), _Fake(), opts=opts)
    out2 = synthesize_target_from_coefficients(np.zeros(7), _Fake(), opts=opts)
    # Same seed -> identical noise.
    assert np.allclose(out1.butt, out2.butt)
