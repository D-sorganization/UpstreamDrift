"""The two-hand club in the musculoskeletal static-optimisation pipeline (OSV-9).

Covers the club option of the muscle-model build and pipeline config, the
shared actuator-torque conversion, and the with/without-club load comparison
that shows the club's inertial load is carried by the upper-body actuators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python import musculoskeletal_swing as ms
from src.engines.physics_engines.opensim.python.musculoskeletal_pipeline import (
    PipelineConfig,
    club_load_comparison,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_solvers import (
    SolveWindow,
    actuator_torques,
    converged_frames,
    parse_static_optimization_log,
)

pytestmark = pytest.mark.unit

GOLF = (
    Path(__file__).resolve().parents[4]
    / "src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim"
)


def _config(tmp_path: Path, club: str | None) -> PipelineConfig:
    states = tmp_path / "states.sto"
    states.write_text("time\tq\n0\t0\n", encoding="utf-8")
    return PipelineConfig(
        golf_model=GOLF,
        states_file=states,
        out_dir=tmp_path / "out",
        window=SolveWindow(0.0, 1.0),
        club=club,
    )


def test_pipeline_config_accepts_known_clubs_and_none(tmp_path: Path) -> None:
    for club in ("driver", "iron7", None):
        _config(tmp_path, club).validate()


def test_pipeline_config_rejects_unknown_club(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="club"):
        _config(tmp_path, "putter").validate()


def test_build_rejects_unknown_club_before_loading_models() -> None:
    with pytest.raises(ValueError, match="club"):
        ms.build_musculoskeletal_model(GOLF, club="putter")


def test_actuator_torques_scales_prefixed_columns_only() -> None:
    acts = {
        "upper_wrist_flex_l": np.array([0.5, -1.0]),
        "reserve_pelvis_tx": np.array([2.0, 2.0]),
        "glmax1_r": np.array([0.1, 0.2]),
    }
    out = actuator_torques(acts, {"upper_wrist_flex_l": 100.0}, "upper_")
    assert list(out) == ["upper_wrist_flex_l"]
    np.testing.assert_allclose(out["upper_wrist_flex_l"], [50.0, -100.0])
    with pytest.raises(ValueError, match="prefix"):
        actuator_torques(acts, {}, "")


def test_club_load_comparison_reports_the_per_frame_difference() -> None:
    t = np.linspace(0.0, 1.0, 5)
    without = {"upper_wrist_flex_l": np.zeros(5), "upper_elbow_flex_r": np.ones(5)}
    with_club = {
        "upper_wrist_flex_l": np.array([0.0, 3.0, -4.0, 0.0, 0.0]),
        "upper_elbow_flex_r": np.ones(5),
    }
    out = club_load_comparison(t, with_club, t, without)
    wrist = out["per_actuator"]["upper_wrist_flex_l"]
    assert wrist["delta_peak"] == pytest.approx(4.0)
    assert wrist["delta_rms"] == pytest.approx(np.sqrt(25.0 / 5.0))
    assert wrist["with_club_rms"] == pytest.approx(np.sqrt(25.0 / 5.0))
    assert wrist["without_club_rms"] == 0.0
    assert out["per_actuator"]["upper_elbow_flex_r"]["delta_peak"] == 0.0
    assert out["largest_delta_rms"] == "upper_wrist_flex_l"
    assert out["n_frames"] == 5 and out["n_frames_excluded"] == 0


def test_parse_static_optimization_log_reads_each_frame(tmp_path: Path) -> None:
    log = tmp_path / "so.log"
    log.write_text(
        "[info] Loading model\n"
        "time = 0.1 Performance = 2.5 Constraint violation = 1e-12\n"
        "   wrist_flex_l: constraint violation = 1.5\n"
        "time = 0.2 Performance = 3.0 Constraint violation = 14.5\n",
        encoding="utf-8",
    )
    t, v = parse_static_optimization_log(log)
    np.testing.assert_allclose(t, [0.1, 0.2])
    np.testing.assert_allclose(v, [1e-12, 14.5])


def test_converged_frames_match_by_time_and_treat_unlogged_as_failed() -> None:
    times = np.array([0.1, 0.2, 0.3])
    mask = converged_frames(times, np.array([0.1, 0.2]), np.array([1e-12, 2.0]))
    assert mask.tolist() == [True, False, False]
    with pytest.raises(ValueError, match="tolerance"):
        converged_frames(times, times, times, tolerance=0.0)


def test_club_load_comparison_uses_only_frames_valid_in_both_runs() -> None:
    t = np.linspace(0.0, 1.0, 4)
    without = {"upper_wrist_flex_l": np.zeros(4)}
    with_club = {"upper_wrist_flex_l": np.array([2.0, 2.0, 9000.0, 2.0])}
    valid = np.array([True, True, False, True])
    out = club_load_comparison(t, with_club, t, without, valid=valid)
    wrist = out["per_actuator"]["upper_wrist_flex_l"]
    assert wrist["delta_peak"] == pytest.approx(2.0)
    assert out["n_frames"] == 3 and out["n_frames_excluded"] == 1
    with pytest.raises(ValueError, match="valid"):
        club_load_comparison(t, with_club, t, without, valid=np.zeros(4, bool))


def test_club_load_comparison_requires_matching_frames_and_actuators() -> None:
    t = np.linspace(0.0, 1.0, 5)
    acts = {"upper_wrist_flex_l": np.zeros(5)}
    with pytest.raises(ValueError, match="time"):
        club_load_comparison(t, acts, t + 0.01, acts)
    with pytest.raises(ValueError, match="actuator"):
        club_load_comparison(t, acts, t, {"upper_elbow_flex_r": np.zeros(5)})


def test_pipeline_config_trail_weld_needs_a_club(tmp_path: Path) -> None:
    cfg = _config(tmp_path, None)
    with pytest.raises(ValueError, match="enforce_trail_weld"):
        PipelineConfig(**{**cfg.__dict__, "enforce_trail_weld": True}).validate()


def test_release_trail_weld_opens_the_two_hand_loop() -> None:
    pytest.importorskip("opensim")
    try:
        ms.resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model, _ = ms.build_musculoskeletal_model(GOLF)
    constraint = model.getConstraintSet().get(ms.TRAIL_GRIP_CONSTRAINT)
    assert constraint.get_isEnforced()
    assert ms.release_trail_weld(model) is True
    assert not constraint.get_isEnforced()
    control, _ = ms.build_musculoskeletal_model(GOLF, club=None)
    assert ms.release_trail_weld(control) is False


def test_build_model_without_club_is_the_control() -> None:
    pytest.importorskip("opensim")
    try:
        ms.resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    with_club, info_club = ms.build_musculoskeletal_model(GOLF)
    control, info = ms.build_musculoskeletal_model(GOLF, club=None)
    assert not control.getBodySet().hasComponent("Club")
    assert info["club"] is None and info["grip_model"] is None
    assert info_club["club"] == "driver" and info_club["grip_model"] == "weld"
    assert info_club["total_mass_kg"] - info["total_mass_kg"] == pytest.approx(
        info_club["club_mass_kg"], abs=1e-9
    )
    assert info["n_muscles"] == info_club["n_muscles"] == 80
