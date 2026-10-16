"""Characterization of trajectory-IK option handling (issue #12845)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.full_body_ik import (
    solve_full_body_ik_trajectory,
)
from src.shared.python.motion_matching.marker_calibration import Array, Pose
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

OFFSETS = {"marker": ("body", (0.0, 0.0, 0.0))}


def _capture() -> TourCapture:
    return TourCapture(
        np.array([0.0]),
        ("marker",),
        np.array([[[2.0, 0.0, 0.0]]]),
        np.ones((1, 1), dtype=bool),
        "a" * 64,
    )


def _poses(q: Array) -> dict[str, Pose]:
    return {"body": (np.eye(3), np.array([q[0], 0.0, 0.0]))}


def _closure(q: Array) -> Array:
    return np.array([q[0] - 1.0])


@pytest.mark.unit
def test_default_options_solve_exactly() -> None:
    q = solve_full_body_ik_trajectory(_poses, OFFSETS, _capture(), np.array([0.0]))
    assert q[0, 0] == pytest.approx(2.0, abs=1e-4)


@pytest.mark.unit
def test_closure_pulls_solution_by_weight() -> None:
    kw = {"closure_fn": _closure, "reg_weight": 0.0}
    light = solve_full_body_ik_trajectory(
        _poses, OFFSETS, _capture(), np.array([0.0]), closure_weight=1.0, **kw
    )
    heavy = solve_full_body_ik_trajectory(
        _poses, OFFSETS, _capture(), np.array([0.0]), closure_weight=3.0, **kw
    )
    assert light[0, 0] == pytest.approx(1.5, abs=1e-6)
    assert heavy[0, 0] == pytest.approx(1.1, abs=1e-6)


@pytest.mark.unit
def test_zero_closure_weight_disables_closure() -> None:
    q = solve_full_body_ik_trajectory(
        _poses,
        OFFSETS,
        _capture(),
        np.array([0.0]),
        closure_fn=_closure,
        closure_weight=0.0,
        reg_weight=0.0,
    )
    assert q[0, 0] == pytest.approx(2.0, abs=1e-6)


@pytest.mark.unit
def test_max_nfev_caps_work() -> None:
    calls = []

    def poses(q: Array) -> dict[str, Pose]:
        calls.append(1)
        return _poses(q)

    solve_full_body_ik_trajectory(
        poses, OFFSETS, _capture(), np.array([0.0]), max_nfev=3, reg_weight=0.0
    )
    assert 0 < len(calls) <= 12


@pytest.mark.unit
@pytest.mark.parametrize("name", ["closure_weight", "reg_weight"])
def test_negative_weights_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must be non-negative"):
        solve_full_body_ik_trajectory(
            _poses, OFFSETS, _capture(), np.array([0.0]), **{name: -1.0}
        )


@pytest.mark.unit
def test_unknown_option_is_type_error() -> None:
    with pytest.raises(TypeError):
        solve_full_body_ik_trajectory(
            _poses, OFFSETS, _capture(), np.array([0.0]), bogus=1
        )


@pytest.mark.unit
def test_input_validation_messages_preserved() -> None:
    with pytest.raises(ValueError, match="finite 1D"):
        solve_full_body_ik_trajectory(_poses, OFFSETS, _capture(), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="missing from offsets"):
        solve_full_body_ik_trajectory(_poses, {}, _capture(), np.array([0.0]))
