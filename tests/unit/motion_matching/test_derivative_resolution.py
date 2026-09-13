"""Measured-floor verdicts for central-difference derivative audit blocks."""

import numpy as np
import pytest

from src.shared.python.motion_matching import derivative_resolution as module

pytestmark = pytest.mark.unit


def test_central_difference_floor_scales_replay_error_by_step() -> None:
    assert module.central_difference_floor(4e-10, 1e-5) == pytest.approx(4e-5)
    assert module.central_difference_floor(0.0, 1e-5) == 0.0


@pytest.mark.parametrize("error,step", [(-1.0, 1e-5), (1e-9, 0.0), (np.nan, 1e-5)])
def test_central_difference_floor_rejects_invalid(error: float, step: float) -> None:
    with pytest.raises(ValueError):
        module.central_difference_floor(error, step)


def _verdict(analytic, central, **kwargs):
    return module.classify_derivative_block(
        np.asarray(analytic, dtype=float), np.asarray(central, dtype=float), **kwargs
    )


def test_resolved_agreement_passes_with_floor_below_gate() -> None:
    verdict = _verdict([1.0, 2.0], [1.0, 2.0 + 1e-4], step=1e-4, replay_error=1e-9)
    assert verdict.verdict == "passed"
    assert verdict.resolution_floor == pytest.approx(1e-5)
    assert verdict.relative_l2_error == pytest.approx(1e-4 / np.sqrt(5))
    assert verdict.gate == 1e-3


def test_agreement_inside_floor_is_unresolved_not_passed() -> None:
    # Numbers agree, but the measured floor exceeds the gate times the norm.
    verdict = _verdict([1e-2, 0.0], [1e-2, 0.0], step=1e-5, replay_error=1e-6)
    assert verdict.verdict == "unresolved_at_step"


def test_disagreement_inside_floor_is_unresolved() -> None:
    verdict = _verdict(
        [3.2e-2, 0.0], [3.2e-2 + 3.9e-5, 0.0], step=1e-5, replay_error=4e-10
    )
    assert verdict.verdict == "unresolved_at_step"
    assert verdict.absolute_l2_error <= verdict.resolution_floor


def test_disagreement_above_floor_fails() -> None:
    verdict = _verdict(
        [3.2e-2, 0.0], [3.2e-2 + 3.9e-5, 0.0], step=1e-5, replay_error=1e-11
    )
    assert verdict.verdict == "failed"


def test_structural_zero_requires_declaration_and_both_norms_inside_floor() -> None:
    kwargs = {"step": 1e-5, "replay_error": 3e-15}
    undeclared = _verdict([5e-14, 0.0], [1.6e-10, 0.0], **kwargs)
    assert undeclared.verdict == "unresolved_at_step"
    declared = _verdict([5e-14, 0.0], [1.6e-10, 0.0], expected_zero=True, **kwargs)
    assert declared.verdict == "structural_zero"
    violated = _verdict([5e-14, 0.0], [1e-3, 0.0], expected_zero=True, **kwargs)
    assert violated.verdict == "failed"


def test_classification_rejects_bad_shapes_and_gates() -> None:
    with pytest.raises(ValueError):
        _verdict([1.0], [1.0, 2.0], step=1e-5, replay_error=1e-9)
    with pytest.raises(ValueError):
        _verdict([1.0], [1.0], step=1e-5, replay_error=1e-9, gate=0.0)
    with pytest.raises(ValueError):
        _verdict([np.nan], [1.0], step=1e-5, replay_error=1e-9)


def test_direction_qualifies_only_with_a_resolved_step_and_no_failure() -> None:
    passed = _verdict([1.0], [1.0], step=1e-4, replay_error=1e-9)
    weak = _verdict([1e-2, 0.0], [1e-2, 0.0], step=1e-5, replay_error=1e-6)
    failed = _verdict([1.0], [1.5], step=1e-4, replay_error=1e-9)
    assert module.qualify_direction([weak, passed]) is True
    assert module.qualify_direction([weak]) is False
    assert module.qualify_direction([passed, failed]) is False
    assert module.qualify_direction([]) is False


def test_cross_block_bound_follows_scaled_orthonormality() -> None:
    rng = np.random.default_rng(7)
    scales = np.r_[np.full(3, 0.1), np.ones(3)]
    raw = rng.normal(size=(6, 4))
    raw[:, 0] = [1, 0, 0, 0, 0, 0]
    basis, _ = np.linalg.qr(raw)
    jacobian = scales[:, None] * basis
    direction = rng.normal(size=4)
    direction /= np.linalg.norm(direction)
    bound_v = module.cross_block_norm_bound(
        jacobian, scales, direction, block=slice(0, 3), other=slice(3, 6)
    )
    actual_v = np.linalg.norm(jacobian[3:] @ direction)
    assert actual_v <= bound_v + 1e-12
    # A pure-position chart direction leaves the velocity block structurally zero.
    position_only = np.linalg.svd(jacobian[:3], full_matrices=False)[2][0]
    assert (
        module.cross_block_norm_bound(
            jacobian, scales, position_only, block=slice(0, 3), other=slice(3, 6)
        )
        <= 1e-7
    )
    with pytest.raises(ValueError):
        module.cross_block_norm_bound(
            jacobian, scales[:3], direction, block=slice(0, 3), other=slice(3, 6)
        )
