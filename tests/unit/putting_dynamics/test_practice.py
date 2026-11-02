"""Practice feedback and putt-result documents (#14141 step 10c).

Ported from the legacy engine's practice mode (ADR-0058: no unique
feature is deleted). Feedback cases use hand-built results, so each
expected suggestion follows from the geometry alone, not a model run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.putting_dynamics import (
    USGA_STIMP_ROLL_MODEL,
    BallState,
    Mode,
    PutterState,
    PuttResult,
    SurfaceSpec,
    TrajectorySample,
    load_putt_result,
    practice_feedback,
    putt_result_from_dict,
    putt_result_to_dict,
    save_putt_result,
    simulate_ball,
    simulate_strike,
)

pytestmark = pytest.mark.unit

_HOLE = (4.0, 0.0)


def _result(rest_x: float, rest_y: float, *, holed: bool = False) -> PuttResult:
    start = TrajectorySample(0.0, 0.0, 0.0, 0.0, 2.0, 0.0, Mode.ROLL)
    rest = TrajectorySample(2.0, rest_x, rest_y, 0.0, 0.0, 0.0, Mode.REST)
    return PuttResult(
        samples=(start, rest),
        collision=None,
        holed=holed,
        rest_x_m=rest_x,
        rest_y_m=rest_y,
        total_distance_m=math.hypot(rest_x, rest_y),
        time_s=2.0,
        skid_distance_m=0.0,
        speed_at_hole_mps=None,
    )


@pytest.mark.parametrize(
    ("rest", "holed", "suggestion"),
    [
        ((4.0, 0.0), True, "Great putt"),
        ((3.7, 0.1), False, "Hit slightly firmer"),
        ((4.3, -0.1), False, "Hit slightly softer"),
        ((3.0, 1.5), False, "Check your aim line"),
    ],
)
def test_feedback_suggestion_follows_the_miss_geometry(
    rest: tuple[float, float], holed: bool, suggestion: str
) -> None:
    feedback = practice_feedback(_result(*rest, holed=holed), _HOLE)
    assert feedback.suggestion == suggestion
    assert feedback.holed is holed
    assert feedback.roll_model == USGA_STIMP_ROLL_MODEL


def test_feedback_reports_the_distance_from_the_hole() -> None:
    feedback = practice_feedback(_result(3.7, 0.4), _HOLE)
    assert feedback.distance_from_hole_m == pytest.approx(0.5)


def test_feedback_rejects_a_non_finite_hole() -> None:
    with pytest.raises(ContractViolationError):
        practice_feedback(_result(3.0, 0.0), (math.nan, 0.0))


def _rolled_putt() -> PuttResult:
    ball = BallState(x_m=0.0, y_m=0.0, vx_mps=1.2, vy_mps=0.1)
    return simulate_ball(ball, SurfaceSpec.flat_uniform())


def test_result_document_round_trips_and_names_its_roll_model() -> None:
    result = _rolled_putt()
    document = putt_result_to_dict(result)
    assert document["roll_model"] == USGA_STIMP_ROLL_MODEL
    assert putt_result_from_dict(json.loads(json.dumps(document))) == result


def test_struck_putt_round_trips_with_its_collision_report() -> None:
    result = simulate_strike(PutterState.typical(1.5), SurfaceSpec.flat_uniform())
    assert result.collision is not None
    document = json.loads(json.dumps(putt_result_to_dict(result)))
    assert putt_result_from_dict(document) == result


def test_saved_result_file_round_trips(tmp_path: Path) -> None:
    result = _rolled_putt()
    path = tmp_path / "putt.json"
    save_putt_result(result, path)
    assert load_putt_result(path) == result


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"roll_model": None}, "names no roll model"),
        ({"roll_model": "ud-legacy-roll/1"}, "ud-legacy-roll/1"),
        ({"schema": "something/else"}, "schema"),
    ],
)
def test_reader_refuses_documents_it_cannot_reproduce(
    change: dict[str, object], message: str
) -> None:
    document = putt_result_to_dict(_result(3.0, 0.0))
    document.update(change)
    if change.get("roll_model", "") is None:
        del document["roll_model"]
    with pytest.raises(ValueError, match=message):
        putt_result_from_dict(document)
