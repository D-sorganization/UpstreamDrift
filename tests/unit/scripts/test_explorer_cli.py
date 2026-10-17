"""Contract tests for the model-explorer container CLI (UD #12878 workstream D).

``scripts/docker/explorer_cli.py`` is the one code path behind the
``explorer-headless`` / ``explorer-gui`` compose services: GL preflight,
engine report, headless MP4 export and the offscreen app screenshot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")
np = pytest.importorskip("numpy")

from scripts.docker import explorer_cli as cli  # noqa: E402

pytestmark = pytest.mark.unit

_GL_ERROR = cli.probe_gl()
requires_render = pytest.mark.skipif(
    _GL_ERROR is not None, reason=f"no offscreen GL context: {_GL_ERROR}"
)

_PENDULUM_XML = """
<mujoco model="pendulum">
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(autouse=True)
def _headless_gl_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the Linux MUJOCO_GL precondition without overriding a real one."""
    if not os.environ.get("MUJOCO_GL"):
        monkeypatch.setenv("MUJOCO_GL", "egl")


@pytest.fixture(name="model_path")
def _model_path(tmp_path: Path) -> Path:
    path = tmp_path / "pendulum.xml"
    path.write_text(_PENDULUM_XML, encoding="utf-8")
    return path


# ---- GL preflight (DbC) -----------------------------------------------------


@pytest.mark.parametrize("backend", ["egl", "osmesa"])
def test_require_gl_backend_accepts_headless_backends(backend: str) -> None:
    assert cli.require_gl_backend({"MUJOCO_GL": backend}) == backend


@pytest.mark.parametrize("env", [{}, {"MUJOCO_GL": ""}])
def test_require_gl_backend_rejects_unset(env: dict[str, str]) -> None:
    with pytest.raises(cli.ExplorerError, match="MUJOCO_GL is not set"):
        cli.require_gl_backend(env)


def test_require_gl_backend_rejects_unknown_backend() -> None:
    with pytest.raises(cli.ExplorerError, match="egl or osmesa"):
        cli.require_gl_backend({"MUJOCO_GL": "glfw"})


# ---- trajectory loading (DbC) -----------------------------------------------


def test_load_trajectory_reads_npz_qpos(tmp_path: Path) -> None:
    path = tmp_path / "r.npz"
    np.savez(path, qpos=np.linspace(0, 1, 10).reshape(10, 1))

    qpos = cli.load_trajectory(path, nq=1)

    assert qpos.shape == (10, 1)


def test_load_trajectory_reads_csv(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    np.savetxt(path, np.zeros((4, 2)), delimiter=",")

    assert cli.load_trajectory(path, nq=2).shape == (4, 2)


def test_load_trajectory_rejects_wrong_width(tmp_path: Path) -> None:
    path = tmp_path / "r.npy"
    np.save(path, np.zeros((5, 3)))

    with pytest.raises(cli.ExplorerError, match="nq=1"):
        cli.load_trajectory(path, nq=1)


def test_load_trajectory_rejects_non_finite(tmp_path: Path) -> None:
    path = tmp_path / "r.npy"
    np.save(path, np.array([[0.0], [np.nan]]))

    with pytest.raises(cli.ExplorerError, match="non-finite"):
        cli.load_trajectory(path, nq=1)


def test_load_trajectory_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(cli.ExplorerError, match="not found"):
        cli.load_trajectory(tmp_path / "nope.npy", nq=1)


def test_load_trajectory_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "r.txt"
    path.write_text("1 2 3")

    with pytest.raises(cli.ExplorerError, match="unsupported results format"):
        cli.load_trajectory(path, nq=1)


# ---- model loading ------------------------------------------------------------


def test_load_model_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(cli.ExplorerError, match="model file not found"):
        cli.load_mujoco_model(str(tmp_path / "missing.xml"))


def test_load_model_rejects_unparseable_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<mujoco><oops/></mujoco>", encoding="utf-8")

    with pytest.raises(cli.ExplorerError, match="could not compile"):
        cli.load_mujoco_model(str(bad))


def test_load_model_resolves_builtin_canonical_humanoid() -> None:
    model = cli.load_mujoco_model(cli.CANONICAL_MODEL_NAME)

    assert model.nq > 20


# ---- export ---------------------------------------------------------------------


def test_simulate_trajectory_has_requested_frame_count(model_path: Path) -> None:
    model = cli.load_mujoco_model(str(model_path))

    qpos = cli.simulate_trajectory(model, duration_s=0.5, fps=10)

    assert qpos.shape == (5, model.nq)
    assert np.isfinite(qpos).all()


def test_simulate_trajectory_fails_fast_when_mujoco_resets_unstable_state(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MuJoCo silently resets an unstable state (time returns to 0); a naive
    ``while data.time < target`` loop then never terminates (seen in the
    ControlTower container on the anthro driver model)."""
    import mujoco

    model = cli.load_mujoco_model(str(model_path))
    real_step = mujoco.mj_step
    calls = {"n": 0}

    def flaky_step(m: object, d: object) -> None:
        calls["n"] += 1
        real_step(m, d)
        if calls["n"] % 5 == 0:
            d.time = 0.0

    monkeypatch.setattr(mujoco, "mj_step", flaky_step)

    with pytest.raises(cli.ExplorerError, match="unstable"):
        cli.simulate_trajectory(model, duration_s=0.5, fps=10)


@pytest.mark.parametrize(("duration", "fps"), [(0.0, 10), (-1.0, 10), (1.0, 0)])
def test_simulate_trajectory_rejects_bad_timing(
    model_path: Path, duration: float, fps: int
) -> None:
    model = cli.load_mujoco_model(str(model_path))

    with pytest.raises(cli.ExplorerError):
        cli.simulate_trajectory(model, duration_s=duration, fps=fps)


@requires_render
def test_export_writes_mp4_with_expected_frames(
    model_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out" / "pendulum.mp4"

    summary = cli.export_video(
        model_arg=str(model_path),
        out=out,
        results=None,
        duration_s=0.5,
        fps=10,
        width=160,
        height=120,
    )

    assert out.is_file()
    assert summary["frames"] == 5
    assert summary["verified_frames"] == 5
    assert (summary["width"], summary["height"]) == (160, 120)
    assert summary["bytes"] > 0


@requires_render
def test_export_replays_loaded_results(model_path: Path, tmp_path: Path) -> None:
    results = tmp_path / "r.npy"
    np.save(results, np.linspace(0.0, 1.0, 7).reshape(7, 1))

    summary = cli.export_video(
        model_arg=str(model_path),
        out=tmp_path / "replay.mp4",
        results=results,
        duration_s=1.0,
        fps=7,
        width=160,
        height=120,
    )

    assert summary["frames"] == 7
    assert summary["source"] == "results"


@requires_render
def test_export_frames_are_not_blank(model_path: Path, tmp_path: Path) -> None:
    model = cli.load_mujoco_model(str(model_path))
    qpos = cli.simulate_trajectory(model, duration_s=0.2, fps=10)

    frames = list(cli.iter_frames(model, qpos, width=160, height=120))

    assert len(frames) == 2
    assert frames[0].shape == (120, 160, 3)
    assert frames[0].std() > 1.0, "rendered frame is a flat colour"


@pytest.mark.parametrize(
    ("width", "height"), [(0, 120), (160, 0), (161, 120), (160, 121)]
)
def test_export_rejects_bad_dimensions(
    model_path: Path, tmp_path: Path, width: int, height: int
) -> None:
    with pytest.raises(cli.ExplorerError, match="even"):
        cli.export_video(
            model_arg=str(model_path),
            out=tmp_path / "x.mp4",
            results=None,
            duration_s=0.5,
            fps=10,
            width=width,
            height=height,
        )


def test_export_rejects_non_mp4_output(model_path: Path, tmp_path: Path) -> None:
    with pytest.raises(cli.ExplorerError, match=r"\.mp4"):
        cli.export_video(
            model_arg=str(model_path),
            out=tmp_path / "x.gif",
            results=None,
            duration_s=0.5,
            fps=10,
            width=160,
            height=120,
        )


# ---- engine report ----------------------------------------------------------------


def test_engine_report_covers_every_engine_without_raising() -> None:
    report = cli.engine_report()

    assert set(report) >= {"mujoco", "pinocchio", "drake", "opensim", "pyqt6"}
    for entry in report.values():
        assert entry["status"] in {"ok", "missing", "broken"}
        assert "detail" in entry
    assert report["mujoco"]["status"] == "ok"


def test_engine_report_marks_missing_engine_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(cli.ENGINE_PROBES, "pinocchio", ("no_such_module_xyz", None))

    report = cli.engine_report()

    assert report["pinocchio"]["status"] == "missing"
    assert "no_such_module_xyz" in report["pinocchio"]["detail"]


# ---- CLI -----------------------------------------------------------------------------


def test_main_engines_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["engines"]) == 0

    assert "mujoco" in json.loads(capsys.readouterr().out)


def test_main_export_reports_error_with_usage_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["export", "--model", str(tmp_path / "nope.xml"), "--out", "x.mp4"])

    assert code == cli.EXIT_BAD_INPUT
    assert "ERROR:" in capsys.readouterr().err


@requires_render
def test_main_export_end_to_end(
    model_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "e2e.mp4"

    code = cli.main(
        [
            "export",
            "--model",
            str(model_path),
            "--out",
            str(out),
            "--duration",
            "0.5",
            "--fps",
            "10",
            "--width",
            "160",
            "--height",
            "120",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 5
