"""Cross-engine bar-path parity of the pull bundles (LIFT-5, #11745).

Same protocol and the same never-loosened tolerances as the back squat
(``test_bar_path_parity.py``): every engine replays the committed bundle open
loop and places the bar (``Grip`` frame) with its own forward kinematics.

* CI replay of the first :data:`CI_STEPS` steps: 1e-9 m on the bar path.
* Full 200-step window and 20-step re-anchored segments (committed record,
  ``scripts/measure_lift_bar_path_parity.py --lift ...``): epic #11605
  level-L2 frame bound (1e-3 m) and coordinate bound (1e-6 rad).  Where a
  coordinate bound is not met the test is a strict xfail with the measured
  numbers.
"""

from __future__ import annotations

import json

import pytest

from tests.support.real_engines import real_module

from src.shared.python.lifting.bar_path_parity import REQUIRED_ENGINES, replay_bar_path
from src.shared.python.lifting.lift_recipes import recipe
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.motion_matching.same_input.scoring import (
    COORDINATE_BOUND_RAD,
    FRAME_BOUND_M,
)

pytestmark = pytest.mark.unit

CI_STEPS = 20
SEGMENT_STEPS = 20
CI_BAR_PATH_ATOL_M = 1e-9
_ENGINE_MODULE = {
    "mujoco": "mujoco",
    "drake": "pydrake",
    "pinocchio": "pinocchio",
    "opensim": "opensim",
}
_DEADLIFT_COORDINATES = pytest.mark.xfail(
    strict=True,
    reason=(
        "Measured deadlift window: the 140-160 ms re-anchored segment ends "
        "4.4e-5 (Drake), 4.3e-5 (Pinocchio) and 4.2e-5 rad (OpenSim) from "
        "MuJoCo, and the full 0.2 s window 2.0e-4 to 2.1e-4 rad (worst final "
        "coordinate subtalar_angle_r), above the 1e-6 rad L2 coordinate bound; the other nine "
        "segments stay within 6.2e-10 rad and the bar within 1.2e-8 m.  "
        "Same foot-contact signature as the back squat; tracked on #11745, "
        "the bound is not loosened."
    ),
)
PULL_COORDINATE_CASES = [
    pytest.param("deadlift", marks=_DEADLIFT_COORDINATES),
    "clean_and_jerk",
]


def _record(lift: str) -> dict:
    return json.loads(recipe(lift).parity_record_path.read_text(encoding="utf-8"))


def _measured(record: dict) -> dict:
    return {
        engine: entry
        for engine, entry in record["engines"].items()
        if not entry.get("unavailable")
    }


@pytest.fixture(scope="module", params=("deadlift", "clean_and_jerk"))
def lift(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def bundle(lift: str) -> InputBundle:
    return InputBundle.load(recipe(lift).bundle_path)


def test_record_is_for_the_committed_bundle(lift: str, bundle: InputBundle) -> None:
    record = _record(lift)
    assert record["lift"] == recipe(lift).bundle_name
    assert record["bundle_spec_sha256"] == bundle.spec_sha256
    assert record["bundle_steps"] == bundle.steps
    assert record["dt_s"] == bundle.dt_s
    assert len(record["source_commit"]) == 40


def test_record_covers_every_engine_over_the_full_window(
    lift: str, bundle: InputBundle
) -> None:
    record = _record(lift)
    for engine in REQUIRED_ENGINES:
        entry = record["engines"][engine]
        assert entry["failure"] is None, engine
        assert entry["steps"] == bundle.steps, engine
    opensim = record["engines"]["opensim"]
    assert opensim.get("unavailable") or opensim["steps"] == bundle.steps


def test_full_window_and_segmented_frames_are_inside_the_l2_frame_bound(
    lift: str,
) -> None:
    record = _record(lift)
    for engine, entry in _measured(record).items():
        assert entry["max_bar_deviation_m"] <= FRAME_BOUND_M, engine
        segmented = entry["segmented"]
        assert segmented["segment_steps"] == SEGMENT_STEPS, engine
        assert segmented["segments"] == record["bundle_steps"] // SEGMENT_STEPS
        assert segmented["max_frame_error_m"] <= FRAME_BOUND_M, engine


@pytest.mark.parametrize("pull", PULL_COORDINATE_CASES)
def test_coordinates_are_inside_the_l2_coordinate_bound(pull: str) -> None:
    for engine, entry in _measured(_record(pull)).items():
        assert entry["max_coordinate_error_rad"] <= COORDINATE_BOUND_RAD, engine
        segmented = entry["segmented"]["max_coordinate_error_rad"]
        assert segmented <= COORDINATE_BOUND_RAD, engine


@pytest.mark.integration
@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("mujoco", marks=pytest.mark.requires_mujoco),
        pytest.param("drake", marks=pytest.mark.requires_drake),
        pytest.param("pinocchio", marks=pytest.mark.requires_pinocchio),
        pytest.param("opensim", marks=pytest.mark.requires_opensim),
    ],
)
def test_engine_replays_the_recorded_bar_path(bundle: InputBundle, engine: str) -> None:
    real_module(_ENGINE_MODULE[engine])
    replay = replay_bar_path(engine, bundle, steps=CI_STEPS)
    assert replay.failure is None
    assert replay.bar_m.shape == (CI_STEPS + 1, 3)
    assert replay.max_bar_deviation_m <= CI_BAR_PATH_ATOL_M, replay.summary()
    if engine == "mujoco":
        assert replay.max_bar_deviation_m == 0.0
        assert replay.max_coordinate_error_rad == 0.0
