"""Markdown report of a speed-continuation run (SHOW-C3c slice 3, #14559)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.continuation_report import (
    render_continuation_report,
)
from src.shared.python.impact_optimization.kinematic_binding import KinematicBinding
from src.shared.python.impact_optimization.speed_continuation import (
    ContinuationPoint,
    ContinuationResult,
)

pytestmark = pytest.mark.unit


def _point(
    speed: float, achieved: float, feasible: bool, torque: tuple[str, ...] = ()
) -> ContinuationPoint:
    sol = SimpleNamespace(
        achieved_speed_mph=achieved,
        binding_joints=torque,
        face_to_path_deg=-0.25,
    )
    return ContinuationPoint(speed, sol, feasible)  # type: ignore[arg-type]


def _result() -> ContinuationResult:
    pts = (
        _point(120.0, 120.0, True, ("LeftShoulderZ",)),
        _point(125.0, 121.4, False),
    )
    return ContinuationResult(pts, 120.0)


ROM = KinematicBinding("SpineInputZ", "ROM", 50.2, 50.0, 1.004)
VEL = KinematicBinding("NeckInputZ", "velocity", 410.0, 400.0, 1.025)


def _render(**kw: object) -> str:
    return render_continuation_report(
        _result(),
        [[], [ROM, VEL]],
        {"SpineInputZ": "prior-derived split; probable miscalibration"},
        **kw,  # type: ignore[arg-type]
    )


def test_one_row_per_rung_with_speed_achieved_and_face_to_path() -> None:
    text = _render()
    assert "| 120 | 120.0 |" in text
    assert "| 125 | 121.4 |" in text
    assert "-0.25" in text


def test_feasible_rung_is_not_labelled_blocked() -> None:
    row = next(r for r in _render().splitlines() if r.startswith("| 120 "))
    assert "blocked" not in row
    assert "LeftShoulderZ" in row


def test_blocked_rung_names_the_subset_and_each_binding_with_kind() -> None:
    row = next(r for r in _render().splitlines() if r.startswith("| 125 "))
    assert "blocked (within the controlled subset)" in row
    assert "SpineInputZ ROM 50.2 / 50.0 deg (100%)" in row
    assert "NeckInputZ velocity 410.0 / 400.0 deg/s (102%)" in row


def test_limit_provenance_is_attached_to_the_bound_joint_only() -> None:
    text = _render()
    assert text.count("prior-derived split; probable miscalibration") == 1
    assert "SpineInputZ" in text.split("Limit provenance")[1]


def test_held_joints_and_fidelity_note_are_reported() -> None:
    text = _render(held_joints=["LeftWristZ"], fidelity_note="full-fidelity: ok")
    assert "Held joints: LeftWristZ" in text
    assert "full-fidelity: ok" in text


def test_defaults_state_nothing_held_and_no_fidelity_claim() -> None:
    text = _render()
    assert "Held joints: none" in text
    assert "Fidelity" not in text


def test_rung_count_mismatch_fails_closed() -> None:
    with pytest.raises(ContractViolationError):
        render_continuation_report(_result(), [[]], {})
