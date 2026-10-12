"""Cross-engine bar-path parity of the back-squat same-input bundle (#11745).

Every engine replays the committed bundle open loop (identical start state,
identical efforts, shared ZOH-RK4 integrator) and places the bar landmark with
its own forward kinematics.  The bar path is compared with the MuJoCo path
recorded in the bundle provenance (``window_bar_path_m``).

Tolerances were fixed before measuring and are never loosened:

* CI replay of the first :data:`CI_STEPS` steps: :data:`CI_BAR_PATH_ATOL_M`
  (1e-9 m), the same order as the 30 ms golf same-input replay (#11607).
* Full 200-step window and the 20-step re-anchored segmented replay
  (committed record, measured by ``scripts/measure_lift_bar_path_parity.py``):
  the epic #11605 level-L2 frame bound (1e-3 m) and coordinate bound
  (1e-6 rad).  The full-window coordinate bound is not met (open-loop
  instability) and is recorded as a strict xfail with the measured numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.lifting.bar_path_parity import (
    BAR_PATH_PARITY_RECORD,
    REQUIRED_ENGINES,
    BarPathReplay,
    replay_bar_path,
)
from src.shared.python.lifting.bundles import BACK_SQUAT_BUNDLE
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


@pytest.fixture(scope="module")
def bundle() -> InputBundle:
    return InputBundle.load(BACK_SQUAT_BUNDLE)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(BAR_PATH_PARITY_RECORD.read_text(encoding="utf-8"))


def test_required_engines_are_the_acceptance_set() -> None:
    assert set(REQUIRED_ENGINES) == {"mujoco", "drake", "pinocchio"}


def test_record_is_for_the_committed_bundle(record, bundle) -> None:
    assert record["bundle_spec_sha256"] == bundle.spec_sha256
    assert record["bundle_steps"] == bundle.steps
    assert record["dt_s"] == bundle.dt_s
    assert len(record["source_commit"]) == 40


def test_record_covers_every_required_engine_over_the_full_window(
    record, bundle
) -> None:
    for engine in REQUIRED_ENGINES:
        entry = record["engines"][engine]
        assert entry["failure"] is None, engine
        assert entry["steps"] == bundle.steps, engine


def _measured(record: dict) -> dict:
    return {
        engine: entry
        for engine, entry in record["engines"].items()
        if not entry.get("unavailable")
    }


def test_full_window_bar_path_is_inside_the_l2_frame_bound(record) -> None:
    for engine, entry in _measured(record).items():
        assert entry["max_bar_deviation_m"] <= FRAME_BOUND_M, engine


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Measured full 0.2 s window: Drake, Pinocchio and OpenSim end up to "
        "4.5e-3 rad from MuJoCo (worst coordinate subtalar_angle_r; the bar "
        "only 5.0e-6 m), above the 1e-6 rad L2 coordinate bound.  The "
        "divergence starts in the 80-100 ms segment; see the segmented xfail."
    ),
)
def test_full_window_coordinates_are_inside_the_l2_coordinate_bound(
    record,
) -> None:
    for engine, entry in _measured(record).items():
        assert entry["max_coordinate_error_rad"] <= COORDINATE_BOUND_RAD, engine


def test_segmented_replay_bar_and_frames_are_inside_the_l2_frame_bound(
    record,
) -> None:
    for engine, entry in _measured(record).items():
        segmented = entry["segmented"]
        assert segmented["segment_steps"] == SEGMENT_STEPS, engine
        assert segmented["segments"] == record["bundle_steps"] // SEGMENT_STEPS
        assert segmented["max_frame_error_m"] <= FRAME_BOUND_M, engine


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Measured 20-step re-anchored replay: the 80-100 ms segment, ending at "
        "the squat bottom, ends 1.8e-4 (Drake), 1.9e-4 (Pinocchio) and 2.0e-4 rad "
        "(OpenSim) from MuJoCo in subtalar_angle_r, above the 1e-6 rad L2 "
        "coordinate bound; the other nine segments stay within 2.3e-7 rad and "
        "every spec frame within 2.2e-7 m.  Tracked on #11745; the bound is "
        "not loosened."
    ),
)
def test_segmented_replay_coordinates_are_inside_the_l2_coordinate_bound(
    record,
) -> None:
    for engine, entry in _measured(record).items():
        segmented = entry["segmented"]
        assert segmented["max_coordinate_error_rad"] <= COORDINATE_BOUND_RAD, engine


def test_opensim_is_recorded_or_unavailable_with_a_reason(record) -> None:
    entry = record["engines"]["opensim"]
    assert entry.get("unavailable") or entry["steps"] > 0
    if entry.get("unavailable"):
        assert entry["reason"]


def test_replay_rejects_an_unknown_engine(bundle) -> None:
    with pytest.raises(ValueError, match="engine"):
        replay_bar_path("bullet", bundle)


def test_replay_rejects_a_bad_step_count(bundle) -> None:
    for steps in (0, bundle.steps + 1):
        with pytest.raises(ValueError, match="steps"):
            replay_bar_path("mujoco", bundle, steps=steps)


def test_replay_rejects_a_bundle_without_a_recorded_bar_path(bundle) -> None:
    provenance = dict(bundle.provenance)
    provenance.pop("window_bar_path_m")
    stripped = InputBundle(
        spec_bytes=bundle.spec_bytes,
        coordinate_order=bundle.coordinate_order,
        dt_s=bundle.dt_s,
        q0=bundle.q0,
        v0=bundle.v0,
        efforts=bundle.efforts,
        reference_q=bundle.reference_q,
        reference_v=bundle.reference_v,
        reference_engine=bundle.reference_engine,
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="window_bar_path_m"):
        replay_bar_path("mujoco", stripped)


@pytest.mark.integration
@pytest.mark.parametrize("engine", ["mujoco", "drake", "pinocchio", "opensim"])
def test_engine_replays_the_recorded_bar_path(bundle, engine) -> None:
    pytest.importorskip(_ENGINE_MODULE[engine])
    replay = replay_bar_path(engine, bundle, steps=CI_STEPS)
    assert isinstance(replay, BarPathReplay)
    assert replay.failure is None
    assert replay.bar_m.shape == (CI_STEPS + 1, 3)
    assert replay.max_bar_deviation_m <= CI_BAR_PATH_ATOL_M, replay.summary()
    if engine == "mujoco":
        assert replay.max_bar_deviation_m == 0.0
        assert replay.max_coordinate_error_rad == 0.0
    summary = replay.summary()
    assert summary["engine"] == engine
    assert np.isfinite(summary["max_bar_deviation_m"])
