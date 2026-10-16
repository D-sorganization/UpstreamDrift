"""Loading real engine results for explorer playback (#12915, epic #12878).

Fixtures under ``tests/fixtures/results_playback`` keep four frames of real
datastore runs (a MuJoCo fit record and a Drake closed-loop rollout) with a
minimal specification, plus two deliberately broken runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.video_frame_source import VideoSourceError
from src.tools.model_explorer.pose_results import (
    PoseFrameSource,
    ResultsSchemaError,
    default_datastore_root,
    discover_result_runs,
    is_pose_results,
    load_pose_frame_source,
)
from src.tools.model_explorer.results_session import ResultsSession

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "results_playback"
MUJOCO_RECORD = FIXTURES / "mujoco_fit" / "dynamics_record.npz"
DRAKE_ROLLOUT = FIXTURES / "drake_rollout" / "closed_loop_drake.npz"
DRAKE_BUNDLE = FIXTURES / "drake_rollout" / "bundle.npz"


@pytest.mark.unit
def test_mujoco_fit_record_loads_with_its_specification() -> None:
    source = load_pose_frame_source(MUJOCO_RECORD)

    assert source.engine == "mujoco"
    assert source.frame_count == 4
    assert source.q.shape == (4, len(source.coordinate_order)) == (4, 44)
    assert source.coordinate_order[0] == "TranslationInputX"
    assert source.is_native_render
    assert source.playback_label == "MuJoCo result"


@pytest.mark.unit
def test_drake_rollout_is_labelled_as_playback_of_drake_result() -> None:
    source = load_pose_frame_source(DRAKE_ROLLOUT)

    assert source.engine == "drake"
    assert not source.is_native_render
    assert source.playback_label.startswith("Playback of Drake result")
    assert "MuJoCo golfer model" in source.playback_label
    np.testing.assert_allclose(source.time_s, [0.0, 0.001, 0.002, 0.003])


@pytest.mark.unit
def test_same_input_bundle_plays_its_reference_trajectory() -> None:
    source = load_pose_frame_source(DRAKE_BUNDLE)

    assert source.engine == "mujoco"
    assert source.frame_count == 4
    np.testing.assert_allclose(np.diff(source.time_s), 0.001)


@pytest.mark.unit
def test_record_without_q_names_the_missing_field() -> None:
    with pytest.raises(ResultsSchemaError) as info:
        load_pose_frame_source(FIXTURES / "missing_q" / "dynamics_record.npz")

    assert info.value.field == "q"
    assert "'q'" in str(info.value)
    assert isinstance(info.value, VideoSourceError)


@pytest.mark.unit
def test_record_without_specification_names_the_spec_field() -> None:
    with pytest.raises(ResultsSchemaError) as info:
        load_pose_frame_source(FIXTURES / "missing_spec" / "dynamics_record.npz")

    assert info.value.field == "spec"
    assert "full_body_spec_hipcal_scaled.json" in str(info.value)


@pytest.mark.unit
def test_q_width_must_match_the_specification(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(MUJOCO_RECORD.parent, run)
    with np.load(MUJOCO_RECORD) as data:
        np.savez(
            run / "dynamics_record.npz", time_s=data["time_s"], q=data["q"][:, :40]
        )

    with pytest.raises(ResultsSchemaError) as info:
        load_pose_frame_source(run / "dynamics_record.npz")

    assert info.value.field == "q"
    assert "44 coordinates" in str(info.value)


@pytest.mark.unit
def test_rollout_without_bundle_names_the_bundle_field(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    for name in ("closed_loop_drake.npz", "closed_loop_drake.json"):
        shutil.copy(DRAKE_ROLLOUT.parent / name, run / name)

    with pytest.raises(ResultsSchemaError) as info:
        load_pose_frame_source(run / "closed_loop_drake.npz")

    assert info.value.field == "bundle"


@pytest.mark.unit
def test_pose_frame_source_rejects_decreasing_time() -> None:
    good = load_pose_frame_source(MUJOCO_RECORD)

    with pytest.raises(ResultsSchemaError) as info:
        PoseFrameSource(
            path=good.path,
            engine="mujoco",
            time_s=good.time_s[::-1].copy(),
            q=good.q,
            coordinate_order=good.coordinate_order,
            spec_bytes=good.spec_bytes,
        )

    assert info.value.field == "time_s"


@pytest.mark.unit
def test_marker_archives_are_not_pose_results(tmp_path: Path) -> None:
    markers = tmp_path / "candidate.npz"
    np.savez(
        markers,
        time_s=np.array([0.0, 0.1]),
        q=np.zeros((2, 3)),
        markers_m=np.zeros((2, 1, 3)),
        target_m=np.zeros((2, 1, 3)),
    )

    assert is_pose_results(MUJOCO_RECORD)
    assert is_pose_results(DRAKE_ROLLOUT)
    assert not is_pose_results(markers)
    assert not ResultsSession.load(markers, "opensim").pose_source


@pytest.mark.unit
def test_discover_lists_playable_runs_only() -> None:
    runs = discover_result_runs(FIXTURES)

    found = {(run.path.parent.name, run.path.name): run for run in runs}
    assert set(found) == {
        ("mujoco_fit", "dynamics_record.npz"),
        ("drake_rollout", "closed_loop_drake.npz"),
        ("missing_q", "dynamics_record.npz"),
    }
    assert found["drake_rollout", "closed_loop_drake.npz"].engine == "drake"
    assert found["mujoco_fit", "dynamics_record.npz"].swing == "mujoco_fit"
    assert "MuJoCo [golfer]" in found["mujoco_fit", "dynamics_record.npz"].label


@pytest.mark.unit
def test_datastore_root_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "swings").mkdir()
    monkeypatch.setenv("UD_SWING_DATASTORE", str(tmp_path))

    assert default_datastore_root() == tmp_path / "swings"


@pytest.mark.unit
def test_session_loads_pose_run_and_scrubs() -> None:
    session = ResultsSession.load(DRAKE_ROLLOUT)

    assert session.pose_source is not None
    assert session.engine == "drake"
    assert session.title.startswith("Playback of Drake result")
    session.scrub(3)
    assert session.current_time_s == pytest.approx(0.003)
    with pytest.raises(ValueError, match="outside"):
        session.scrub(4)


@pytest.mark.unit
def test_time_series_reports_angles_in_degrees_and_translations_in_metres() -> None:
    from src.tools.model_explorer.golfer_playback import results_time_series

    source = load_pose_frame_source(MUJOCO_RECORD)
    series = results_time_series(source, with_club=False)

    assert list(series)[:2] == ["TranslationInputX", "TranslationInputY"]
    assert series["TranslationInputX"].unit == "m"
    np.testing.assert_allclose(series["TranslationInputX"].values, source.q[:, 0])
    hip = source.coordinate_order.index("HipInputX")
    assert series["HipInputX"].unit == "deg"
    np.testing.assert_allclose(series["HipInputX"].values, np.degrees(source.q[:, hip]))
    assert all(len(s.values) == source.frame_count for s in series.values())
