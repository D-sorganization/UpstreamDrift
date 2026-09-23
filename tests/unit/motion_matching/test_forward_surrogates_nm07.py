"""NM-07 (#10622): Forward surrogates, real-clock resampling, and physics-structured checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.surrogate.validate import (
    ValidationReport,
    check_contact_boundary_failure,
    check_trust_region,
    compare_gradient_fidelity,
    compute_directional_derivative,
    quaternion_geodesic_error_rad,
    resample_to_timegrid,
    validate_against_simscape,
)

pytestmark = pytest.mark.unit


def _dummy_target(
    times: np.ndarray,
    clubhead: np.ndarray | None = None,
    quats: np.ndarray | None = None,
    impact_idx: int = 5,
) -> ClubTarget:
    n = len(times)
    ch = clubhead if clubhead is not None else np.zeros((n, 3))
    q = quats if quats is not None else np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    butt = np.zeros((n, 3))
    prov = SourceProvenance("test.c3d", "c3d", "test", "trial1", "a" * 64)
    return ClubTarget(
        time=times,
        butt=butt,
        clubhead=ch,
        club_quat=q,
        impact_idx=impact_idx,
        source=prov,
    )


def test_real_clock_versus_index_timing() -> None:
    """Non-uniform source timestamps resample accurately with real-clock timegrid."""
    src_times = np.array([0.0, 0.1, 0.2, 0.8, 1.0])
    src_values = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )

    dst_times = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    resampled_real = resample_to_timegrid(
        src_values, src_time=src_times, dst_time=dst_times
    )

    # At t=0.5 (halfway between 0.2 and 0.8), value should be 2 + (0.5-0.2)/(0.8-0.2) * 6 = 5.0
    assert np.isclose(resampled_real[2, 0], 5.0, atol=1e-5)

    # Index-based uniform resampling ignores non-uniform real-time timestamps
    resampled_index = resample_to_timegrid(
        src_values, src_time=None, dst_time=len(dst_times)
    )
    # In index-based, index 2 is t=0.5 in [0, 4] -> index 2 in src_values -> value 2.0 (distorted)
    assert not np.isclose(resampled_index[2, 0], 5.0, atol=0.5)


def test_real_clock_resampling_dbc_rejects_non_monotonic() -> None:
    """DbC: non-monotonic timestamps raise ValueError."""
    bad_times = np.array([0.0, 0.2, 0.1, 0.5])
    values = np.zeros((4, 3))
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        resample_to_timegrid(values, src_time=bad_times, dst_time=5)


def test_quaternion_rotation_sign_invariance() -> None:
    """Rotations q and -q represent identical SO(3) orientations (zero geodesic error)."""
    q1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0, 0.0]])
    # Antipodal quaternions represent the exact same physical rotations
    q2 = -q1.copy()

    err = quaternion_geodesic_error_rad(q1, q2)
    assert np.isclose(err, 0.0, atol=1e-6)

    # Truly orthogonal rotation: 90 degrees around X -> geodesic error is pi/2
    q_diff = np.array([[0.70710678, 0.70710678, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    err_diff = quaternion_geodesic_error_rad(q1, q_diff)
    assert err_diff > 0.5


def test_trust_region_acceptance_and_rejection() -> None:
    """Coefficients within radius are accepted; outside radius are rejected."""
    anchor = np.array([1.0, 2.0, 3.0])
    inside = np.array([1.1, 2.1, 2.9])
    outside = np.array([3.0, 5.0, 7.0])

    assert check_trust_region(inside, anchor, trust_radius=0.5) is True
    assert check_trust_region(outside, anchor, trust_radius=0.5) is False


def test_native_directional_derivatives() -> None:
    """Directional derivative matches analytical gradient projection."""

    def quad_fn(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + 2 * x[1], 3 * x[0] - x[1] ** 2])

    theta = np.array([2.0, 3.0])
    direction = np.array([1.0, 0.0])  # d/dx[0]

    d_val = compute_directional_derivative(quad_fn, theta, direction, eps=1e-5)
    # df/dx[0] at (2, 3): [2*x[0], 3] = [4.0, 3.0]
    expected = np.array([4.0, 3.0])
    assert np.allclose(d_val, expected, atol=1e-4)


def test_adversarial_surrogate_exploitation_rejected() -> None:
    """When surrogate gradient opposes native dynamics derivative, flag fidelity rejection."""
    surrogate_grad = np.array([1.0, 0.0, 0.0])
    # Case 1: Aligned gradient -> accepted
    native_aligned = np.array([0.9, 0.1, 0.0])
    cos_sim, accepted = compare_gradient_fidelity(
        surrogate_grad, native_aligned, cos_sim_threshold=0.3
    )
    assert accepted is True
    assert cos_sim > 0.8

    # Case 2: Opposing gradient (adversarial shortcut) -> rejected
    native_opposing = np.array([-1.0, 0.0, 0.0])
    cos_sim_bad, accepted_bad = compare_gradient_fidelity(
        surrogate_grad, native_opposing, cos_sim_threshold=0.3
    )
    assert accepted_bad is False
    assert cos_sim_bad < 0.0


def test_contact_boundary_failure_detection() -> None:
    """Smooth models evaluating through impact contact transitions report contact boundary failure."""
    times = np.linspace(0.0, 0.3, 10)
    target = _dummy_target(times, impact_idx=5)

    # Smooth rigid model without contact dynamics cannot cover impact transition
    assert (
        check_contact_boundary_failure(target, model_type="smooth_rigid_surrogate")
        is True
    )
    # Contact-capable model passes boundary check
    assert (
        check_contact_boundary_failure(target, model_type="contact_compliant_model")
        is False
    )
