"""Force and torque vectors drawn at swing events of a manifest run (Issue #14439)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("matplotlib")

from src.sim_run.mujoco_run import TRAJECTORY_FILENAME, run_mujoco
from src.sim_run.vector_overlay import EventRender, render_event_overlays

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_XML = """<mujoco model="overlay_pendulum">
  <option timestep="0.002"/>
  <worldbody>
    <body name="upper" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" mass="2"/>
      <body name="lower" pos="0.4 0 0">
        <joint name="wrist" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
_EVENTS = {"address": 0, "top": 40, "impact": 80}


def _run(out: Path):
    return run_mujoco(
        _XML,
        model_id="test/overlay_pendulum",
        output_dir=out,
        duration=0.2,
        seed=3,
        ud_commit="b" * 40,
    )


def _render(run_dir: Path, out: Path) -> tuple[EventRender, ...]:
    manifest = _run(run_dir)
    return render_event_overlays(manifest, run_dir, _XML, events=_EVENTS, out_dir=out)


def test_one_png_per_event_hashes_to_its_receipt(tmp_path: Path) -> None:
    renders = _render(tmp_path / "run", tmp_path / "png")

    assert [r.event for r in renders] == list(_EVENTS)
    for render in renders:
        png = tmp_path / "png" / render.filename
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert render.sha256 == hashlib.sha256(png.read_bytes()).hexdigest()
    assert renders[1].time_s == pytest.approx(40 * 0.002)


def test_rendering_the_same_run_twice_is_byte_identical(tmp_path: Path) -> None:
    first = _render(tmp_path / "a", tmp_path / "pa")
    second = _render(tmp_path / "b", tmp_path / "pb")

    assert [r.sha256 for r in first] == [r.sha256 for r in second]


def test_legend_states_vector_scale_and_units(tmp_path: Path) -> None:
    renders = _render(tmp_path / "run", tmp_path / "png")

    legend = renders[1].legend
    assert any(line.startswith("Force Ref:") and " N = " in line for line in legend)
    assert any(line.endswith(" m") for line in legend)
    assert all(r.arrows > 0 for r in renders)


def test_snapshot_of_the_glyphs_drawn_at_each_event(tmp_path: Path) -> None:
    renders = _render(tmp_path / "run", tmp_path / "png")
    snapshot = Path(__file__).with_name("vector_overlay_snapshot.json")

    actual = {
        r.event: {a["label"]: [a["magnitude"], *a["tip_m"]] for a in r.glyphs["arrows"]}
        for r in renders
    }
    assert actual == json.loads(snapshot.read_text(encoding="utf-8"))


def test_a_tampered_trajectory_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _run(run_dir)
    (run_dir / TRAJECTORY_FILENAME).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="does not match its manifest digest"):
        render_event_overlays(
            manifest, run_dir, _XML, events=_EVENTS, out_dir=tmp_path / "png"
        )


def test_model_text_must_match_the_recorded_revision(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = _run(run_dir)

    with pytest.raises(ValueError, match="recorded revision"):
        render_event_overlays(
            manifest,
            run_dir,
            _XML.replace('mass="1"', 'mass="2"'),
            events=_EVENTS,
            out_dir=tmp_path / "png",
        )


@pytest.mark.parametrize("events", [{}, {"late": 101}, {"early": -1}])
def test_event_rows_must_lie_inside_the_trajectory(
    tmp_path: Path, events: dict[str, int]
) -> None:
    run_dir = tmp_path / "run"
    manifest = _run(run_dir)

    with pytest.raises(ValueError, match="event"):
        render_event_overlays(
            manifest, run_dir, _XML, events=events, out_dir=tmp_path / "png"
        )


def test_renders_without_a_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "MPLBACKEND"):
        monkeypatch.delenv(name, raising=False)

    (render,) = render_event_overlays(
        _run(tmp_path / "run"),
        tmp_path / "run",
        _XML,
        events={"top": 40},
        out_dir=tmp_path / "png",
    )

    assert (tmp_path / "png" / render.filename).is_file()
