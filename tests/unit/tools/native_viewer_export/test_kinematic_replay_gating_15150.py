"""A kinematic replay shows no force-derived visual, and a capacity ellipsoid
without torque limits is unavailable, not a relabelled mobility ellipsoid (#15150).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export import kinematic_replay as kr
from src.tools.native_viewer_export import overlay
from src.tools.native_viewer_export.core import ExportSettings
from src.tools.native_viewer_export.overlay import OverlayOptions
from src.tools.native_viewer_export.runner import ExportJob, run_export

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[4]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
STATES = 9
REASON = "kinematic replay: bundle carries no efforts"


def _swing(efforts_scale: float):
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.full_body_spec import load_full_body_spec
    from src.tools.native_viewer_export.core import SwingInput

    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    order = tuple(spec["coordinate_order"])
    rng = np.random.default_rng(15150)
    q = np.linspace(0.0, 1.0, STATES)[:, None] * rng.normal(0.0, 0.2, len(order))
    v = np.gradient(q, 1.0 / 360.0, axis=0)
    efforts = efforts_scale * rng.normal(0.0, 1.0, (STATES - 1, len(order)))
    bundle = InputBundle(
        spec_bytes=json.dumps(spec).encode("utf-8"),
        coordinate_order=order,
        dt_s=1.0 / 360.0,
        q0=q[0],
        v0=v[0],
        efforts=efforts,
        reference_q=q,
        reference_v=v,
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "sweep", "Driver", "mujoco", v=v)


@pytest.fixture(scope="module")
def kinematic():
    return _swing(0.0)


@pytest.fixture(scope="module")
def dynamic():
    return _swing(1.0)


def _feed(swing, **kw):
    opts = kw.pop("options", None)
    return overlay.build_overlay_feed(
        swing, "mujoco", grip=True, axial=True, palm_thumb=True, options=opts, **kw
    )[0]


def test_a_kinematic_replay_withholds_shading_grip_and_palm_thumb(
    kinematic, caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        feed = _feed(kinematic)
    assert all(feed.axial_at(k) is None for k in range(STATES))
    assert feed.grip_analyses is None
    assert not feed.show_palm_thumb
    note = feed.hud_lines_at(0)[-1]
    assert note.startswith(kr.NOTE)
    assert all(name in note for name in ("shading", "grip", "palm-thumb"))
    assert any("palm-thumb" in r.getMessage() for r in caplog.records)


def test_a_replay_names_only_what_was_requested(kinematic) -> None:
    feed = overlay.build_overlay_feed(kinematic, "mujoco")[0]
    assert feed.hud_lines_at(0) == (kr.NOTE,)


def test_explicit_vector_request_keeps_every_visual(kinematic) -> None:
    feed = _feed(kinematic, options=OverlayOptions(forces_requested=True))
    assert feed.grip_analyses is not None and feed.show_palm_thumb
    assert feed.axial_at(0) is not None
    assert feed.hud_lines_at(0) == ()


def test_a_bundle_with_efforts_is_unchanged(dynamic) -> None:
    feed = _feed(dynamic)
    assert feed.grip_analyses is not None and feed.show_palm_thumb
    assert feed.axial_at(0) is not None
    assert feed.hud_lines_at(0) == ()


def test_rollout_velocity_reaches_the_provider(dynamic, monkeypatch) -> None:
    seen = {}
    real = overlay.BundleOverlayProvider

    class Spy(real):
        def __init__(self, *a, **kw):
            seen["v"] = kw.get("v")
            super().__init__(*a, **kw)

    monkeypatch.setattr(overlay, "BundleOverlayProvider", Spy)
    rollout_v = dynamic.bundle.reference_v * 0.5
    overlay.build_overlay_feed(replace(dynamic, v=rollout_v), "mujoco")
    assert np.array_equal(seen["v"], rollout_v)
    overlay.build_overlay_feed(replace(dynamic, v=None), "mujoco")
    assert seen["v"] is None  # no rollout velocity: the reference stands


# ---- capacity ellipsoids -------------------------------------------------


def _hm(**kw):
    base = {
        "mobility_axes": np.eye(3),
        "mobility_eigenvalues": np.array([4.0, 1.0, 0.25]),
        "force_capacity": None,
        "acceleration_capacity": None,
        "centre": (0.0, 0.0, 1.0),
        "condition": 2.0,
    }
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize("kind", ["force-capacity", "acceleration-capacity"])
def test_capacity_without_limits_is_unavailable_with_a_reason(kind) -> None:
    with pytest.raises(overlay.EllipsoidUnavailable, match="torque limits"):
        overlay._extract_physical_radii(_hm(), kind)


def test_capacity_with_limits_is_unchanged() -> None:
    cap = SimpleNamespace(axes=np.eye(3) * 2, radii=np.array([3.0, 2.0, 1.0]))
    axes, radii = overlay._extract_physical_radii(
        _hm(force_capacity=cap, acceleration_capacity=cap), "force-capacity"
    )
    assert radii == (3.0, 2.0, 1.0) and np.array_equal(axes, cap.axes)
    assert overlay._extract_physical_radii(_hm(), "mobility")[1] == (2.0, 1.0, 0.5)


def test_hud_says_unavailable_instead_of_drawing_a_mislabelled_ellipsoid() -> None:
    pt = SimpleNamespace(name="lead_hand", group="hands")
    hm = _hm()
    scale = SimpleNamespace(unit="N")
    lines = overlay._build_hand_hud_lines_fn(
        lambda i: {"lead_hand": hm}, [pt], ["force-capacity"], {"force-capacity": scale}
    )(0)
    assert "unavailable" in lines[0] and "torque limits" in lines[0]
    assert "lead_hand" in lines[0] or "lead" in lines[0]


# ---- runner: grip record, strip reason, shading legend ---------------------


class _Backend:
    engine = "mujoco"

    def __init__(self):
        self.settings = None

    def unavailable_reason(self):
        return None

    def render(self, swing, settings, indices, ov):
        self.settings = settings
        for _ in indices:
            yield {
                v: np.zeros((settings.height, settings.width, 3), np.uint8)
                for v in settings.views
            }


class _Writer:
    def __init__(self, path: Path, fps: int) -> None:
        self.path = path

    def append_data(self, frame) -> None:
        pass

    def close(self) -> None:
        self.path.write_bytes(b"x")


def _bundle(efforts: float, steps: int = 20) -> InputBundle:
    q = np.zeros((steps + 1, 2))
    return InputBundle(
        spec_bytes=json.dumps({"coordinate_order": ["A", "B"]}).encode(),
        coordinate_order=("A", "B"),
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.full((steps, 2), efforts),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )


def _run(tmp_path: Path, efforts: float, **flags):
    bpath = tmp_path / "b.npz"
    _bundle(efforts).save(bpath)
    backend = _Backend()
    settings = ExportSettings(
        width=32,
        height=32,
        fps=20,
        speeds=(1.0,),
        impact_time_s=0.01,
        shading=ForceColorScale(enabled=True),
        grip=True,
        inset_strip=True,
        **flags,
    )
    results = run_export(
        ExportJob(bpath, tmp_path / "o", "s", "Driver", ("mujoco",)),
        settings,
        lambda e: backend,
        lambda swing, engine: (None, (0.0, 0.0, 0.9)),
        _Writer,
    )
    return results[0], backend


def test_runner_withholds_grip_json_strip_and_legend_on_a_replay(tmp_path) -> None:
    result, backend = _run(tmp_path, 0.0)
    assert not backend.settings.shading_enabled
    assert result.inset_strip == f"inset strip unavailable: {REASON}"
    doc = json.loads(Path(result.grip_json).read_text(encoding="utf-8"))
    assert doc == {"unavailable": REASON}


def test_runner_keeps_everything_when_forces_are_requested(tmp_path) -> None:
    result, backend = _run(tmp_path, 0.0, vectors_requested=True)
    assert backend.settings.shading_enabled
    assert result.grip_json is None  # stub feed has no grip analyses; no record
    assert result.inset_strip != f"inset strip unavailable: {REASON}"


def test_runner_is_unchanged_for_a_bundle_with_efforts(tmp_path) -> None:
    result, backend = _run(tmp_path, 1.0)
    assert backend.settings.shading_enabled and result.grip_json is None
    assert REASON not in str(result.inset_strip)
