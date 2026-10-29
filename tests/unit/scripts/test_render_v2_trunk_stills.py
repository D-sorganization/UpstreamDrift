"""Address, top and impact stills of the full-body-v2 trunk (HRG-D3, #13174).

GL-free: a fake native backend stands in for the viewer, so these pin the plan,
labels, manifest and fail-closed rules. The real frames need a GL host.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from scripts import render_golfer_realism_showcase as showcase
from scripts import render_v2_trunk_stills as stills
from src.shared.python.motion_matching.full_body_v2_geometry import (
    derive_landmark_v2_spec,
)
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.core import SwingInput

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
V1 = Path(__file__).resolve().parents[3] / (
    "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
N = 9
FRAMES = {"address": 0, "top": 4, "impact": 7}


@pytest.fixture(scope="module")
def v1_doc() -> dict[str, Any]:
    return json.loads(V1.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v2_doc(v1_doc: dict[str, Any]) -> dict[str, Any]:
    return derive_landmark_v2_spec(v1_doc, provenance="HRG-D3 stills fixture")


def _bundle(doc: dict[str, Any], n: int = N) -> InputBundle:
    names = tuple(doc["coordinate_order"])
    q = np.arange(n, dtype=float)[:, None] * np.ones(len(names))
    return InputBundle(
        spec_bytes=json.dumps(doc, sort_keys=True).encode("utf-8"),
        coordinate_order=names,
        dt_s=0.01,
        q0=q[0],
        v0=np.zeros(len(names)),
        efforts=np.zeros((n - 1, len(names))),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
    )


class FakeBackend:
    """Stand-in viewer: constant frames (black when ``blank``), records requests."""

    def __init__(self, engine: str, reason: str | None = None, blank: bool = False):
        self.engine, self.reason, self.blank = engine, reason, blank
        self.requests: list[tuple[int, ...]] = []

    def unavailable_reason(self) -> str | None:
        return self.reason

    def render(self, swing, settings, indices, overlay):  # type: ignore[no-untyped-def]
        self.requests.append(tuple(indices))
        for index in indices:
            shade = 0 if self.blank else 40 + index
            yield {v: np.full((120, 200, 3), shade, np.uint8) for v in settings.views}


def _plan(tmp_path: Path, doc: dict[str, Any]):
    spec, bundle = tmp_path / "spec.json", tmp_path / "driver.npz"
    spec.write_text(json.dumps(doc), encoding="utf-8")
    bundle.write_bytes(b"dummy")
    argv = ["--swing", f"driver={bundle}", "--v2-spec", str(spec), "--out"]
    argv += [str(tmp_path / "out"), "--engines", "mujoco", "--views", "face_on,dtl"]
    plan = showcase.plan_from_args(showcase.build_parser().parse_args(argv))
    return replace(plan, command="python3 -m scripts.render_golfer_realism_showcase")


def _run(plan, bundle, backends, frames=FRAMES) -> Path:  # type: ignore[no-untyped-def]
    settings = replace(showcase.settings_for(plan), overlays=False, hud=False)
    return stills.render_v2_stills(
        plan,
        settings,
        backend_factory=lambda engine: backends[engine],
        event_finder=lambda swing: dict(frames),
        bundle_loader=lambda path: bundle,
    )


def test_non_v2_spec_and_foreign_bundle_are_rejected(tmp_path, v1_doc, v2_doc) -> None:
    with pytest.raises(ValueError, match="full-body-v2"):
        _plan(tmp_path, v1_doc)
    backend = FakeBackend("mujoco")
    with pytest.raises(ValueError, match="does not carry"):
        _run(_plan(tmp_path, v2_doc), _bundle(v1_doc), {"mujoco": backend})
    assert backend.requests == [] and not list(tmp_path.rglob("*.png"))


def test_writes_one_labelled_still_per_event_view_and_engine(tmp_path, v2_doc) -> None:
    plan, bundle, backend = (
        _plan(tmp_path, v2_doc),
        _bundle(v2_doc),
        FakeBackend("mujoco"),
    )
    manifest = _run(plan, bundle, {"mujoco": backend})

    sha12 = bundle.spec_sha256[:12]
    pngs = sorted((tmp_path / "out" / "driver" / "stills").glob("*.png"))
    assert len(pngs) == 6  # address/top/impact x face_on/down_the_line
    assert all(p.name.endswith(f"_{sha12}.png") for p in pngs)
    assert {p.name.split("_")[-2] for p in pngs} == set(FRAMES)
    assert backend.requests == [(0, 4, 7)]

    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert doc["command"] == plan.command
    impact = [s for s in doc["stills"] if s["event"] == "impact"]
    assert {s["frame"] for s in impact} == {7}
    assert {s["spec_sha256"] for s in impact} == {bundle.spec_sha256}
    assert len(doc["stills"]) == 6
    assert all((manifest.parent / s["file"]).is_file() for s in doc["stills"])


def test_missing_viewer_or_blank_frame_fails_closed(tmp_path, v2_doc) -> None:
    plan, bundle = _plan(tmp_path, v2_doc), _bundle(v2_doc)
    with pytest.raises(stills.StillsError, match="no GL context"):
        _run(plan, bundle, {"mujoco": FakeBackend("mujoco", "no GL context")})
    with pytest.raises(stills.StillsError, match="blank"):
        _run(plan, bundle, {"mujoco": FakeBackend("mujoco", blank=True)})
    assert not list(tmp_path.rglob("*.png"))


@pytest.mark.parametrize("top, impact", [(7, 4), (4, 99)])
def test_event_frames_must_increase_inside_the_swing(
    tmp_path, v2_doc, top, impact
) -> None:
    frames = {"address": 0, "top": top, "impact": impact}
    backends = {"mujoco": FakeBackend("mujoco")}
    with pytest.raises(ValueError, match="event frames"):
        _run(_plan(tmp_path, v2_doc), _bundle(v2_doc), backends, frames)


def _synthetic_swing(n: int = 101) -> tuple[SwingInput, np.ndarray]:
    """Slow-at-the-top swing: back and up, then a fast return through the ball."""
    s = np.linspace(0.0, 1.0, n)
    phase = 0.5 * (1.0 - np.cos(np.pi * np.clip(s / 0.5, 0, 1)))  # 0 -> 1 at the top
    phase -= np.clip((s - 0.5) / 0.3, 0, 1) ** 2  # back to 0 at the ball, s = 0.8
    follow = np.clip((s - 0.8) / 0.2, 0, 1)
    head = np.stack([-0.8 * phase + 0.8 * follow, 0 * s, 0.05 + 0.9 * phase], axis=1)
    bundle = _bundle({"coordinate_order": ["index"]}, n)
    return SwingInput(bundle, bundle.reference_q, "synthetic", "Driver", "mujoco"), head


def test_event_finder_uses_the_shared_rules_and_keeps_impact_time(monkeypatch) -> None:
    from src.shared.python.model_appearance import club_face
    from src.shared.python.motion_matching import club_face_target
    from src.tools.native_viewer_export.overlay import detect_impact_time_s

    swing, head = _synthetic_swing()

    class Source:  # MuJoCo overlay source placing ``Clubhead`` on the synthetic path
        def __init__(self, spec_bytes: bytes) -> None: ...

        def frame_poses(self, q: dict[str, float]) -> dict[str, np.ndarray]:
            pose = np.eye(4)
            pose[:3, 3] = head[int(q["index"])]
            return {club_face_target.FACE_FRAME: pose}

    module = ModuleType("overlay_source")
    module.MujocoOverlaySource = Source  # type: ignore[attr-defined]
    path = "src.engines.physics_engines.mujoco.python.overlay_source"
    monkeypatch.setitem(sys.modules, path, module)
    monkeypatch.setattr(
        club_face_target, "face_centre_in_frame", lambda spec: np.zeros(3)
    )

    times = np.asarray(swing.source_times_s)
    t_impact, k, s = club_face.ball_passage(times, head)
    found = stills.detect_event_frames(swing)
    assert found == {
        "address": 0,
        "top": club_face.top_of_backswing_index(times, head),
        "impact": k + int(s >= 0.5),
    }
    assert 0 < found["top"] < found["impact"] < len(times) - 1
    assert detect_impact_time_s(swing) == pytest.approx(t_impact, abs=1e-15)
