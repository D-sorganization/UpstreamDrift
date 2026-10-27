"""OpenSim integration tests for lift video rendering (OSL-8, #13045)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.msk_lift_video import (
    generate_contact_sheet,
    lift_video_schedule,
    render_opensim_lift_video,
    solve_full_lift_motion_and_muscles,
)

_REPO = Path(__file__).resolve().parents[3]
IK_ID_RECEIPTS_PATH = (
    _REPO / "docs/development/lifting/evidence/opensim_ik_id/receipts.json"
)

pytestmark = pytest.mark.unit


def _real_opensim() -> None:
    from tests.helpers.opensim_support import require_opensim

    osim = require_opensim()
    if not hasattr(osim, "__file__"):
        pytest.skip("opensim in sys.modules is a test double, not the real package")


def _require_body() -> None:
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        resolve_base_model,
    )

    try:
        resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(f"muscle-bearing body not available: {exc}")


def test_bench_press_unavailable_status(tmp_path: Path) -> None:
    """Bench press video export reports unavailable without invoking OpenSim."""
    result = render_opensim_lift_video(
        lift="bench_press",
        out_path=tmp_path / "bench.mp4",
    )
    assert result["status"] == "unavailable"
    assert "supine" in result["reason"]


def test_generate_contact_sheet_dimensions(tmp_path: Path) -> None:
    """Contact sheet arranges 6 frames into a 2x3 grid with correct dimensions."""
    frames = [np.full((100, 150, 3), i * 40, dtype=np.uint8) for i in range(6)]
    out_sheet = tmp_path / "sheet.png"
    result_path = generate_contact_sheet(frames, out_sheet, count=6)
    assert result_path.is_file()
    import imageio.v2 as imageio

    img = imageio.imread(str(result_path))
    assert img.shape == (200, 450, 3)


@pytest.mark.requires_opensim
def test_marker_set_cleared_for_showcase() -> None:
    """Experimental and tracking markers are cleared so floating red spheres are not rendered."""
    _real_opensim()
    _require_body()
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        resolve_base_model,
    )
    from src.shared.python.engine_core.opensim_runtime import import_opensim

    osim = import_opensim()
    base_model = resolve_base_model()
    model = osim.Model(str(base_model))
    initial_markers = model.getMarkerSet().getSize()
    assert initial_markers > 0

    model.updMarkerSet().clearAndDestroy()
    assert model.getMarkerSet().getSize() == 0


@pytest.mark.requires_opensim
@pytest.mark.headless_safe
@pytest.mark.timeout(180)
def test_headless_lift_video_render_smoke(tmp_path: Path) -> None:
    """Renders a 2-frame smoke clip headless and verifies model occupies frame."""
    _real_opensim()
    _require_body()

    out_mp4 = tmp_path / "squat_smoke.mp4"
    result = render_opensim_lift_video(
        lift="squat",
        out_path=out_mp4,
        view="side",
        fps=60,
        speed=1.0,
        overlay="forces",
        frames=2,
    )
    assert out_mp4.is_file()
    assert result["frames"] == 2
    assert result["lift"] == "squat"
    assert result["view"] == "side"
    assert result["status"] == "rendered"


@pytest.mark.requires_opensim
@pytest.mark.headless_safe
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "lift",
    [
        "deadlift",
        "clean_and_jerk",
        pytest.param(
            "snatch",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Measured max bar error 60.5 mm exceeds OSL-3 short-window receipt "
                    "57.9 mm at window end during full-motion retargeting (#13045)"
                ),
            ),
        ),
        pytest.param(
            "squat",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Measured max bar error 13.9 mm exceeds OSL-3 short-window receipt "
                    "1.9 mm during dynamic squatting (#13045)"
                ),
            ),
        ),
    ],
)
def test_bar_body_origin_matches_bundle(lift: str) -> None:
    """Compare barbell body origin against bundle bar position for >= 3 frames."""
    import json

    from src.shared.python.lifting.bundle_targets import IK_LANDMARKS, lift_targets
    from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES

    _real_opensim()
    _require_body()

    receipt = json.loads(IK_ID_RECEIPTS_PATH.read_text(encoding="utf-8"))
    receipt_err = float(
        receipt["lifts"][lift]["ik_id"]["ik"]["marker_error"]["per_marker_max_m"]["bar"]
    )

    bundle_path = SAME_INPUT_LIFT_BUNDLES[lift]
    window_targets = lift_targets(lift, bundle_path, stride=5)

    full_targets, build_full, full_ik, _, _, _, window_s, _ = (
        solve_full_lift_motion_and_muscles(lift, stride=5)
    )

    sched, q_interp, _ = lift_video_schedule(
        full_targets.time_s, full_ik.q, fps=60, speed=1.0
    )

    model = build_full.model
    model.updMarkerSet().clearAndDestroy()
    coords = model.getCoordinateSet()
    state = model.initSystem()
    bar_body = model.getBodySet().get("barbell")

    t_start, t_end = window_s
    t_samples = [t_start, (t_start + t_end) / 2.0, t_end]
    measured_errors: list[float] = []

    for t_test in t_samples:
        f_idx = int(np.argmin(np.abs(sched.sample_times_s - t_test)))
        t_actual = float(sched.sample_times_s[f_idx])
        for name, val in zip(full_ik.coordinates, q_interp[f_idx], strict=True):
            coords.get(name).setValue(state, float(val), False)
        model.realizePosition(state)
        sim_bar = np.array(
            [bar_body.getPositionInGround(state).get(i) for i in range(3)]
        )

        t_rel = float(np.clip(t_actual - t_start, 0.0, window_targets.time_s[-1]))
        w_idx = int(np.argmin(np.abs(window_targets.time_s - t_rel)))
        bundle_bar = window_targets.positions_m[w_idx, IK_LANDMARKS.index("bar")]
        err = float(np.linalg.norm(sim_bar - bundle_bar))
        measured_errors.append(err)

    assert len(measured_errors) >= 3
    max_err = max(measured_errors)
    assert max_err <= receipt_err + 1e-4, (
        f"{lift}: max measured bar error {max_err * 1000:.2f} mm > receipt {receipt_err * 1000:.2f} mm"
    )


def test_render_worker_pythonpath_excludes_parent_stdlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#13896: worker PYTHONPATH holds only repo roots, never stdlib/lib-dynload."""
    import os
    import subprocess
    import sysconfig
    from types import SimpleNamespace

    from src.engines.physics_engines.opensim.python import msk_lift_video

    captured: dict[str, dict[str, str]] = {}

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured["env"] = dict(kwargs["env"])  # type: ignore[call-overload]
        raise RuntimeError("stop before result.json")

    monkeypatch.setattr(msk_lift_video.subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("UPSTREAMDRIFT_MSK_BASE_MODEL", "unused.osim")

    with pytest.raises(RuntimeError, match="stop before result.json"):
        render_opensim_lift_video(lift="squat", out_path=tmp_path / "squat.mp4")

    env = captured["env"]
    entries = [p for p in env["PYTHONPATH"].split(os.pathsep) if p]
    paths = sysconfig.get_paths()
    banned = {Path(paths["stdlib"]), Path(paths["platstdlib"])}
    for entry in entries:
        assert Path(entry) not in banned
        assert Path(entry).name != "lib-dynload"
    assert str(_REPO) in entries
    assert env["NATIVE_VIEWER_XVFB"] == "1"
    assert env["UPSTREAMDRIFT_MSK_BASE_MODEL"] == "unused.osim"


def test_worker_env_restores_interpreter_libdir_after_opensim_clobber() -> None:
    """#13896: opensim's Linux ``__init__`` sets ``LD_LIBRARY_PATH = PATH``.

    A worker spawned after ``import opensim`` must still load its own
    ``libpython`` (not the system one), so the interpreter LIBDIR leads.
    """
    from src.tools.native_viewer_export.backends._subprocess import (
        with_interpreter_libdir,
    )

    shared = {"LIBDIR": "/py/lib", "Py_ENABLE_SHARED": 1}
    env = with_interpreter_libdir(
        {"LD_LIBRARY_PATH": "/usr/bin:/bin"}, platform="linux", config=shared.get
    )
    assert env["LD_LIBRARY_PATH"] == "/py/lib:/usr/bin:/bin"

    unset = with_interpreter_libdir({}, platform="linux", config=shared.get)
    assert unset["LD_LIBRARY_PATH"] == "/py/lib"

    present = {"LD_LIBRARY_PATH": "/py/lib:/x"}
    assert (
        with_interpreter_libdir(present, platform="linux", config=shared.get) == present
    )

    static = {"LIBDIR": "/py/lib", "Py_ENABLE_SHARED": 0}
    assert with_interpreter_libdir({}, platform="linux", config=static.get) == {}
    assert with_interpreter_libdir({}, platform="win32", config=shared.get) == {}


def test_worker_env_applies_interpreter_libdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """#13896: ``worker_env`` routes through the LIBDIR guard."""
    from src.tools.native_viewer_export.backends import _subprocess

    seen: list[dict[str, str]] = []

    def spy(env: dict[str, str], **_kw: object) -> dict[str, str]:
        seen.append(env)
        return {**env, "SPY": "1"}

    monkeypatch.setattr(_subprocess, "with_interpreter_libdir", spy)
    assert _subprocess.worker_env()["SPY"] == "1"
    assert len(seen) == 1
