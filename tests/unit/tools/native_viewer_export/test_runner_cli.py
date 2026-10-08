"""Runner, receipt loading, registry and CLI of the native viewer export tool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.backends.registry import (
    BACKEND_FACTORIES,
    make_backend,
)
from src.tools.native_viewer_export.core import (
    ENGINES,
    BackendUnavailable,
    ExportSettings,
    load_receipt_rollout,
)
from src.tools.native_viewer_export.runner import ExportJob, run_export

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COORDS = ("A", "B")


def _bundle(steps: int = 100) -> InputBundle:
    q = np.zeros((steps + 1, 2))
    return InputBundle(
        spec_bytes=json.dumps({"coordinate_order": list(COORDS)}).encode(),
        coordinate_order=COORDS,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((steps, 2)),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )


def _receipt(tmp_path: Path, bundle: InputBundle, name: str = "r") -> Path:
    q = np.ones_like(bundle.reference_q)
    np.savez(tmp_path / f"{name}.npz", q=q)
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "schema": "same-input-closed-loop/v1",
                "engine": "pinocchio",
                "bundle": {"spec_sha256": bundle.spec_sha256},
            }
        ),
        encoding="utf-8",
    )
    return path


class _Backend:
    def __init__(self, engine: str, reason: str | None = None) -> None:
        self.engine, self.reason, self.swings = engine, reason, []

    def unavailable_reason(self) -> str | None:
        return self.reason

    def render(self, swing, settings, indices, overlay):
        self.swings.append((swing, settings, overlay))
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


def test_receipt_rollout_roundtrip(tmp_path: Path) -> None:
    b = _bundle()
    q, engine = load_receipt_rollout(_receipt(tmp_path, b), b)
    assert engine == "pinocchio" and q.shape == b.reference_q.shape


def test_receipt_for_other_spec_is_rejected(tmp_path: Path) -> None:
    b = _bundle()
    path = _receipt(tmp_path, b)
    doc = json.loads(path.read_text())
    doc["bundle"]["spec_sha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="different specification"):
        load_receipt_rollout(path, b)


def test_receipt_schema_is_checked(tmp_path: Path) -> None:
    b = _bundle()
    path = _receipt(tmp_path, b)
    path.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ValueError, match="not a same-input receipt"):
        load_receipt_rollout(path, b)


def test_run_export_skips_unavailable_and_renders_rest(tmp_path: Path) -> None:
    b = _bundle()
    bpath = tmp_path / "b.npz"
    b.save(bpath)
    backends = {
        "drake": _Backend("drake", "no pydrake"),
        "pinocchio": _Backend("pinocchio"),
    }
    seen = []

    def overlay(swing, engine):
        seen.append(engine)
        return None, (0.5, 0.0, 0.9)

    job = ExportJob(
        bpath,
        tmp_path / "out",
        "driver",
        "Driver",
        ("drake", "pinocchio"),
        {"pinocchio": _receipt(tmp_path, b)},
    )
    results = run_export(
        job,
        ExportSettings(width=32, height=32, fps=20, speeds=(1.0,)),
        backends.__getitem__,
        overlay,
        _Writer,
    )
    assert [r.engine for r in results] == ["drake", "pinocchio"]
    assert results[0].skipped and not results[1].skipped
    swing, settings, _ = backends["pinocchio"].swings[0]
    assert swing.rollout_engine == "pinocchio" and float(swing.q[0, 0]) == 1.0
    assert settings.lookat_m == (0.5, 0.0, 0.9)
    assert seen == ["pinocchio"]
    assert (tmp_path / "out" / "driver_pinocchio_face_on_1x.mp4").exists()
    assert (tmp_path / "out" / "driver_pinocchio_2x2_1x.mp4").exists()


def test_overlay_unavailable_still_renders(tmp_path: Path) -> None:
    b = _bundle()
    bpath = tmp_path / "b.npz"
    b.save(bpath)

    def overlay(swing, engine):
        raise BackendUnavailable("no mujoco")

    backend = _Backend("drake")
    results = run_export(
        ExportJob(bpath, tmp_path / "o", "d", "Driver", ("drake",)),
        ExportSettings(width=32, height=32, fps=20, speeds=(1.0,)),
        lambda e: backend,
        overlay,
        _Writer,
    )
    assert not results[0].skipped and backend.swings[0][2] is None


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    job = ExportJob(tmp_path / "b.npz", tmp_path, "d", "Driver", ("nope",))
    with pytest.raises(ValueError, match="unknown engine"):
        run_export(job, ExportSettings())


def test_registry_covers_all_engines_and_validates() -> None:
    assert set(BACKEND_FACTORIES) == set(ENGINES)
    with pytest.raises(ValueError, match="unknown engine"):
        make_backend("gazebo")
    for engine in ENGINES:
        assert make_backend(engine).engine == engine


@pytest.mark.parametrize("text,expected", [("640x544", (640, 544)), ("8X9", (8, 9))])
def test_parse_size(text: str, expected: tuple[int, int]) -> None:
    assert cli.parse_size(text) == expected


def test_parse_size_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="must look like"):
        cli.parse_size("big")


def test_cli_reports_bad_input(tmp_path: Path, capsys) -> None:
    rc = cli.main(
        [
            "--bundle",
            str(tmp_path / "x.npz"),
            "--out",
            str(tmp_path),
            "--swing",
            "d",
            "--size",
            "oops",
        ]
    )
    assert rc == 2 and "error:" in capsys.readouterr().err


def test_backend_unavailable_is_an_exception() -> None:
    assert issubclass(BackendUnavailable, Exception)
