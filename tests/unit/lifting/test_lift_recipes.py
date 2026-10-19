"""Per-lift bundle recipes, windows and deferred lifts (LIFT-5, #11745)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.lift_recipes import (
    DEFERRED_LIFTS,
    RECIPES,
    LiftRecipe,
    bundle_lifts,
    knee_pass_time,
    recipe,
    window_centre_time,
    window_start_time,
)
from src.shared.python.lifting.pack_audit.names import LIFTS
from src.shared.python.lifting.sagittal_retarget import SagittalSolution

pytestmark = pytest.mark.unit

_SHANK = 0.44


def _solution(bar_up: np.ndarray, shank_rad: float = 0.0) -> SagittalSolution:
    n = bar_up.size
    angles = np.zeros((n, 3))
    angles[:, 0] = shank_rad
    return SagittalSolution(
        time_s=np.linspace(0.0, 0.01 * (n - 1), n),
        segment_angles_rad=angles,
        bar_m=np.column_stack((np.zeros(n), bar_up)),
        source={"segment_lengths_m": [_SHANK, 0.44, 0.51]},
    )


def test_every_standard_lift_has_a_recipe_or_a_deferral_reason() -> None:
    assert set(RECIPES) | set(DEFERRED_LIFTS) == set(LIFTS)
    assert not set(RECIPES) & set(DEFERRED_LIFTS)
    for lift, reason in DEFERRED_LIFTS.items():
        assert len(reason) > 80, lift


def test_bundle_lifts_follow_the_standard_order() -> None:
    lifts = bundle_lifts()
    assert lifts == tuple(lift for lift in LIFTS if lift in RECIPES)
    assert lifts[0] == "squat"


def test_squat_recipe_is_the_committed_back_squat() -> None:
    from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE

    squat = recipe("squat")
    assert squat.bundle_path == BACK_SQUAT_BUNDLE
    assert squat.exercise == "full_squat"
    assert not squat.hand_held


def test_deadlift_recipe_hangs_the_bar_and_centres_on_the_knee_pass() -> None:
    pull = recipe("deadlift")
    assert pull.hand_held
    assert pull.bar_convention == "deadlift"
    assert pull.window_event == "knee_pass"
    assert pull.bundle_path.name == f"{pull.bundle_name}_bundle.npz"
    assert pull.parity_record_path.parent == pull.bundle_path.parent


def test_clean_and_jerk_recipe_solves_the_whole_lift_with_a_phase_bar() -> None:
    cj = recipe("clean_and_jerk")
    assert cj.hand_held and not cj.supine
    # The full Tools clean and jerk (floor to overhead lockout), not the clean.
    assert cj.exercise == cj.bar_convention == "clean_and_jerk"
    assert cj.duration_s == 4.0
    assert cj.window_event == "knee_pass"
    assert cj.phase == "first_pull"
    assert "overhead" in cj.rationale and "window" in cj.rationale
    assert "clean_and_jerk" not in DEFERRED_LIFTS


def test_bench_press_recipe_lies_supine_and_centres_on_the_chest() -> None:
    bench = recipe("bench_press")
    assert bench.supine and not bench.hand_held
    assert bench.exercise == bench.bar_convention == "bench_press"
    assert bench.window_event == "bottom"
    assert bench.phase == "chest_touch"
    assert "bench_press" not in DEFERRED_LIFTS
    assert not recipe("squat").supine


def test_snatch_recipe_holds_a_phase_correct_bar_and_centres_on_the_knee_pass() -> None:
    snatch = recipe("snatch")
    assert snatch.hand_held and not snatch.supine
    assert snatch.exercise == snatch.bar_convention == "snatch"
    assert snatch.window_event == "knee_pass"
    assert snatch.phase == "first_pull"
    assert "snatch" not in DEFERRED_LIFTS


def test_recipe_names_the_reason_for_a_deferred_lift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.lifting import lift_recipes

    monkeypatch.setattr(lift_recipes, "DEFERRED_LIFTS", {"snatch": "measured reason"})
    with pytest.raises(ValueError, match="no bundle: measured reason"):
        recipe("snatch")
    with pytest.raises(ValueError, match="unknown lift"):
        recipe("curl")


def test_recipe_validates_its_fields() -> None:
    base: dict[str, object] = {
        "lift": "deadlift",
        "bundle_name": "x",
        "exercise": "deadlift",
        "bar_convention": "deadlift",
        "window_event": "knee_pass",
        "phase": "knee_pass",
        "duration_s": 1.0,
        "bar_mass_kg": 50.0,
        "rationale": "r",
    }
    with pytest.raises(ValueError, match="lift"):
        LiftRecipe(**{**base, "lift": "curl"})
    with pytest.raises(ValueError, match="window_event"):
        LiftRecipe(**{**base, "window_event": "lockout"})
    with pytest.raises(ValueError, match="positive"):
        LiftRecipe(**{**base, "duration_s": 0.0})


def test_knee_pass_is_the_first_sample_with_the_bar_at_knee_height() -> None:
    bar = np.linspace(0.2, 0.7, 51)
    solution = _solution(bar)
    first = int(np.flatnonzero(bar >= _SHANK)[0])
    assert knee_pass_time(solution) == pytest.approx(solution.time_s[first])


def test_knee_pass_uses_the_shank_angle() -> None:
    bar = np.linspace(0.2, 0.7, 51)
    tilted = _solution(bar, shank_rad=np.pi / 4)
    first = int(np.flatnonzero(bar >= _SHANK * np.cos(np.pi / 4))[0])
    assert knee_pass_time(tilted) == pytest.approx(tilted.time_s[first])


def test_knee_pass_needs_a_bar_that_reaches_the_knee() -> None:
    with pytest.raises(ValueError, match="knee"):
        knee_pass_time(_solution(np.full(5, 0.1)))


def test_window_centres_on_the_bottom_or_the_knee_pass() -> None:
    bar = np.concatenate((np.linspace(0.8, 0.3, 26), np.linspace(0.3, 0.8, 26)[1:]))
    solution = _solution(bar)
    assert window_centre_time(solution, "bottom") == pytest.approx(0.25)
    with pytest.raises(ValueError, match="event"):
        window_centre_time(solution, "lockout")


def test_window_start_is_a_sample_that_keeps_the_window_in_the_lift() -> None:
    solution = _solution(np.linspace(0.2, 0.7, 101))
    start = window_start_time(solution, "knee_pass", 0.2)
    assert start in solution.time_s
    assert start + 0.2 <= solution.time_s[-1] + 1e-12
    centre = knee_pass_time(solution)
    assert abs(start - (centre - 0.1)) <= 0.005 + 1e-12
    late = _solution(np.linspace(0.0, _SHANK, 101))
    assert window_start_time(late, "knee_pass", 0.2) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="window_s"):
        window_start_time(solution, "knee_pass", 0.0)
    with pytest.raises(ValueError, match="longer"):
        window_start_time(solution, "knee_pass", 5.0)
