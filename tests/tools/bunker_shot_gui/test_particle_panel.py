"""The particle playback panel inside the shot viewport (issue #9545, slice 9).

Runs on a stock runner: VTK is *forced* absent here (the same ``find_spec``
fake ``test_particle_viewport_degradation`` uses), so every assertion holds
whether or not the optional ``viz3d`` extra is installed.
``test_particle_panel_vtk`` covers the camera and clipping controls that only
a real render can exercise. Every artifact is the SYNTHETIC TEST FIXTURE.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")
pytest.importorskip("pytestqt", reason="the panel tests drive Qt through qtbot")
pytest.importorskip("h5py", reason="the trajectory artifact is HDF5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402

from bunkershot3d.fields.trajectory import final_marker_path  # noqa: E402
from src.tools.bunker_shot_gui import particle_panel, render3d_vtk  # noqa: E402
from src.tools.bunker_shot_gui.particle_panel import (  # noqa: E402
    NO_TRAJECTORY_REASON,
    TIME_SCALES,
    ParticlePlaybackPanel,
    advance_clock,
)
from src.tools.bunker_shot_gui.viewport_widgets import ShotViewportWidget  # noqa: E402
from tests.unit.tools.bunker_shot_gui._synthetic_particle_artifact import (  # noqa: E402
    FRAME_TIMES_S,
    SYNTHETIC_SOLVER_NAME,
    write_synthetic_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_TRANSPORT = ("play", "step_back", "step_forward", "scrub", "speed")
_VTK_ONLY = ("camera", "clipping")


@pytest.fixture(autouse=True)
def _no_vtk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render3d_vtk, "find_spec", lambda name: None)


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    return write_synthetic_artifact(tmp_path / "synthetic.h5")


@pytest.fixture()
def panel(qtbot):  # type: ignore[no-untyped-def]
    widget = ParticlePlaybackPanel()
    qtbot.addWidget(widget)
    yield widget
    widget.unload()


@pytest.fixture()
def loaded(panel, artifact):  # type: ignore[no-untyped-def]
    assert panel.load_trajectory(artifact), panel.status_text
    return panel


# ------------------------------------------------------------------ embedding


def test_the_shot_viewport_embeds_the_particle_panel(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = ShotViewportWidget("Shot")
    qtbot.addWidget(view)
    assert isinstance(view.particles, ParticlePlaybackPanel)
    assert view.particles.parent() is view
    assert view.particles.open_action.text() == "Open particle trajectory…"


# ---------------------------------------------------------- disabled reasons


def test_with_nothing_loaded_every_control_but_open_states_why(panel) -> None:  # type: ignore[no-untyped-def]
    reasons = panel.disabled_reasons()
    assert reasons["open"] is None
    for name in (*_TRANSPORT, *_VTK_ONLY):
        assert reasons[name] == NO_TRAJECTORY_REASON, name
    assert not panel.has_trajectory
    assert NO_TRAJECTORY_REASON in panel.reason_text
    assert "pychrono" in panel.status_text


def test_without_vtk_the_view_controls_name_the_reason(loaded) -> None:  # type: ignore[no-untyped-def]
    reasons = loaded.disabled_reasons()
    for name in _TRANSPORT:
        assert reasons[name] is None, name
    for name in _VTK_ONLY:
        assert reasons[name] is not None
        assert reasons[name].startswith("VTK unavailable: "), reasons[name]
        assert "viz3d" in reasons[name]
    assert "VTK unavailable" in loaded.reason_text
    assert not loaded.viewport_available
    assert "3-D particle viewport unavailable" in loaded.image_text


def test_disabled_controls_are_disabled_widgets(panel) -> None:  # type: ignore[no-untyped-def]
    for name in (*_TRANSPORT, *_VTK_ONLY):
        assert not panel.control_enabled(name), name
    assert panel.control_enabled("open")


# ------------------------------------------------------------------ the stamp


def test_the_stamp_names_tier_verdict_solver_and_decimation(loaded) -> None:  # type: ignore[no-untyped-def]
    stamp = loaded.stamp_text
    assert "F3" in stamp
    assert "BEYOND VALIDATION" in stamp.upper()
    assert SYNTHETIC_SOLVER_NAME in stamp
    assert "drawing 6 of 6 particles" in stamp
    assert "no body tracks" in stamp


# ----------------------------------------------------------------- transport


def test_step_forward_and_back_move_one_frame(loaded) -> None:  # type: ignore[no-untyped-def]
    assert loaded.frame_index == 0
    loaded.step_forward()
    assert loaded.frame_index == 1
    assert "frame 2 of 5" in loaded.stamp_text
    loaded.step_back()
    assert loaded.frame_index == 0


def test_the_scrubber_is_in_simulation_time_not_frame_index(loaded) -> None:  # type: ignore[no-untyped-def]
    # 0.5 of the 0.5 ms span is 0.25 ms, which is frame 2 on the irregular
    # cadence -- a frame-index slider would land on frame 2 by a different
    # route only by luck, so check frame 3's time too.
    loaded.set_scrub_fraction(0.5)
    assert loaded.frame_index == 2
    loaded.set_scrub_fraction(0.6)
    assert loaded.frame_index == 3
    assert f"{FRAME_TIMES_S[3]:.6f} s" in loaded.time_text


def test_stepping_moves_the_scrubber_to_the_frame_time(loaded) -> None:  # type: ignore[no-untyped-def]
    loaded.step_forward()
    expected = FRAME_TIMES_S[1] / FRAME_TIMES_S[-1]
    assert loaded.scrub_fraction == pytest.approx(expected, abs=1e-3)


def test_play_starts_the_timer_and_pause_stops_it(loaded) -> None:  # type: ignore[no-untyped-def]
    loaded.play()
    assert loaded.is_playing
    assert loaded.play_action.text() == "Pause"
    loaded.pause()
    assert not loaded.is_playing
    assert loaded.play_action.text() == "Play"


def test_ticks_advance_by_the_time_scale_and_stop_at_the_end(loaded) -> None:  # type: ignore[no-untyped-def]
    loaded.set_time_scale_index(len(TIME_SCALES) - 1)  # fastest
    loaded.play()
    for _ in range(100):
        if not loaded.is_playing:
            break
        loaded.tick()
    assert loaded.frame_index == len(FRAME_TIMES_S) - 1
    assert not loaded.is_playing


def test_the_time_scale_changes_how_far_a_tick_moves(artifact, qtbot) -> None:  # type: ignore[no-untyped-def]
    reached = []
    for index in (0, len(TIME_SCALES) - 1):
        widget = ParticlePlaybackPanel()
        qtbot.addWidget(widget)
        widget.load_trajectory(artifact)
        widget.set_time_scale_index(index)
        widget.play()
        for _ in range(3):
            widget.tick()
        reached.append(widget.frame_index)
        widget.unload()
    assert reached[0] < reached[1]


def test_play_from_the_last_frame_restarts(loaded) -> None:  # type: ignore[no-untyped-def]
    loaded.set_scrub_fraction(1.0)
    loaded.play()
    assert loaded.frame_index == 0
    loaded.pause()


def test_the_real_timer_drives_playback(loaded, qtbot) -> None:  # type: ignore[no-untyped-def]
    loaded.set_time_scale_index(len(TIME_SCALES) - 1)
    loaded.play()
    qtbot.waitUntil(lambda: loaded.frame_index > 0, timeout=3000)
    loaded.pause()


def test_advance_clock_refuses_a_bad_tick() -> None:
    assert advance_clock(1.0e-4, wall_s=0.04, time_scale=1e-3) == pytest.approx(1.4e-4)
    with pytest.raises(ValueError, match="time_scale"):
        advance_clock(0.0, wall_s=0.04, time_scale=0.0)
    with pytest.raises(ValueError, match="wall_s"):
        advance_clock(0.0, wall_s=-1.0, time_scale=1e-3)


# ---------------------------------------------------------------- rejection


def _corrupt_positions(path: Path) -> None:
    with h5py.File(path, "r+") as handle:
        data = handle["frames/positions"]
        row = np.array(data[0])
        row[0] += 1.0
        data[0] = row


def _wrong_version(path: Path) -> None:
    marker = final_marker_path(path)
    content = json.loads(marker.read_text(encoding="utf-8"))
    content["trajectory_schema_version"] = 99
    marker.write_text(json.dumps(content), encoding="utf-8")


def _missing_marker(path: Path) -> None:
    final_marker_path(path).unlink()


@pytest.mark.parametrize(
    ("damage", "phrase"),
    [
        (_corrupt_positions, "digest"),
        (_wrong_version, "schema v99"),
        (_missing_marker, "no finalisation marker"),
    ],
)
def test_a_rejected_artifact_shows_the_reader_error_and_draws_nothing(
    loaded, artifact, tmp_path, damage, phrase
) -> None:  # type: ignore[no-untyped-def]
    broken = write_synthetic_artifact(tmp_path / "broken.h5")
    damage(broken)
    assert not loaded.load_trajectory(broken)
    assert "Rejected broken.h5" in loaded.status_text
    assert phrase in loaded.status_text.lower() or phrase in loaded.status_text
    # The previously loaded artifact is dropped too: nothing stale, nothing
    # partial stays on screen under the error.
    assert not loaded.has_trajectory
    assert loaded.stamp_text == ""
    assert loaded.disabled_reasons()["play"] == NO_TRAJECTORY_REASON


def test_a_file_that_is_not_hdf5_is_rejected(panel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not an hdf5 file")
    assert not panel.load_trajectory(junk)
    assert "Rejected junk.h5" in panel.status_text


# ---------------------------------------------------------------- open action


def test_the_open_action_loads_the_chosen_file(panel, artifact, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        particle_panel.QFileDialog,
        "getOpenFileName",
        lambda *_a, **_k: (str(artifact), ""),
    )
    panel.open_action.trigger()
    assert panel.has_trajectory


def test_cancelling_the_dialog_changes_nothing(panel, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        particle_panel.QFileDialog, "getOpenFileName", lambda *_a, **_k: ("", "")
    )
    panel.open_action.trigger()
    assert not panel.has_trajectory


# ------------------------------------------------------------------ keyboard


def _focused(panel, qtbot):  # type: ignore[no-untyped-def]
    panel.show()
    qtbot.waitExposed(panel)
    panel.activateWindow()
    qtbot.waitActive(panel)
    target = panel.focus_chain()[0]  # the open button is never disabled
    target.setFocus()
    qtbot.waitUntil(target.hasFocus)
    return target


def test_shortcuts_step_and_toggle_play(loaded, qtbot) -> None:  # type: ignore[no-untyped-def]
    target = _focused(loaded, qtbot)
    qtbot.keyClick(target, Qt.Key.Key_Period)
    assert loaded.frame_index == 1
    qtbot.keyClick(target, Qt.Key.Key_Comma)
    assert loaded.frame_index == 0
    qtbot.keyClick(target, Qt.Key.Key_P)
    assert loaded.is_playing
    qtbot.keyClick(target, Qt.Key.Key_P)
    assert not loaded.is_playing


def test_shortcuts_do_nothing_while_disabled(panel, qtbot) -> None:  # type: ignore[no-untyped-def]
    target = _focused(panel, qtbot)
    qtbot.keyClick(target, Qt.Key.Key_P)
    assert not panel.is_playing


def test_every_control_has_an_accessible_name_and_a_focus_order(panel) -> None:  # type: ignore[no-untyped-def]
    chain = panel.focus_chain()
    assert len(chain) == 8
    names = [widget.accessibleName() for widget in chain]
    assert all(names), names
    assert len(set(names)) == len(names)
    for earlier, later in zip(chain, chain[1:], strict=False):
        assert earlier.nextInFocusChain() is later or later in _reachable(earlier)


def _reachable(start):  # type: ignore[no-untyped-def]
    seen = []
    widget = start.nextInFocusChain()
    while widget is not start and len(seen) < 64:
        seen.append(widget)
        widget = widget.nextInFocusChain()
    return seen


def test_tooltips_name_the_shortcuts(loaded) -> None:  # type: ignore[no-untyped-def]
    assert "(P)" in loaded.play_action.toolTip()
    assert "(.)" in loaded.step_forward_action.toolTip()
    assert "(,)" in loaded.step_back_action.toolTip()
