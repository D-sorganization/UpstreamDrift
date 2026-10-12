"""Optional shape requests bind opacity, model provenance and guarded downloads."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.workspace.test_necromatcher_video_jobs import (
    _fake_export,
    _freeze_execution_stamp,
    _wait,
)

pytestmark = pytest.mark.unit


def _shape_worker(path: Path, budget: float, cancelled: Any) -> dict[str, Any]:
    from src.shared.python.workspace.necromatcher_shape_overlay import (
        shape_overlay_provenance,
    )
    from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions
    from src.shared.python.workspace import NecromatcherLibrary

    _fake_export(path, budget, cancelled)
    request = json.loads(path.read_text(encoding="utf-8"))
    library = NecromatcherLibrary(request["library_root"])
    fit = library.load_fit(request["source_fit_id"])
    target = path.parent / "overlay/manifest.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    manifest["shape_overlay"] = shape_overlay_provenance(
        fit["provenance"]["native_definition"],
        fit["model_hash"],
        ShapeOverlayOptions.from_record(request["shape_overlay"]),
    )
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return {"manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def test_shape_option_is_hashed_persisted_and_guarded(
    native_fit_case: Any, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions

    library, source, _ = native_fit_case
    library.add_fit("source", "practice", source)
    _freeze_execution_stamp(jobs, monkeypatch)
    monkeypatch.setattr(jobs, "_execute_worker", _shape_worker)
    options = ShapeOverlayOptions(0.6)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source", shape_overlay=options)["run_id"]
        result = _wait(session, run)
        assert result["status"] == "succeeded", result
        root = library.root / "video-runs" / run
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        assert request["shape_overlay"] == {"opacity": 0.6}
        manifest_run = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest_run["hashes"]["controller_hash"] == jobs._digest(
            {
                "selected_frames": request["selected_frames"],
                "shape_overlay": {"opacity": 0.6},
            }
        )
        assert result["shape_overlay"] == {"opacity": 0.6}
        assert session.download(run).is_file()
        target = root / "overlay/manifest.json"
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["shape_overlay"]["options"]["opacity"] = 0.7
        target.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="shape|Shape"):
            jobs._outputs(root, request)
        with pytest.raises(ValueError, match="verified|hash"):
            session.download(run)
    finally:
        session.close()


def test_worker_passes_shape_only_when_enabled(
    native_fit_case: Any, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.workspace import necromatcher_video_worker as worker
    from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions

    library, source, _ = native_fit_case
    library.add_fit("source", "practice", source)
    stamp = jobs.fit_execution_stamp()
    monkeypatch.setattr(jobs, "fit_execution_stamp", lambda: stamp)
    monkeypatch.setattr(worker, "fit_execution_stamp", lambda: stamp)
    monkeypatch.setattr(jobs, "_execute_worker", _shape_worker)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source", shape_overlay=ShapeOverlayOptions(0.4))["run_id"]
        assert _wait(session, run)["status"] == "succeeded"
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            worker, "export_fit_video", lambda *args, **kwargs: calls.append(kwargs)
        )
        request_path = library.root / "video-runs" / run / "request.json"
        worker.execute(request_path)
        assert calls == [
            {"selected_frames": (0, 2), "shape_overlay": ShapeOverlayOptions(0.4)}
        ]
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["shape_overlay"]["opacity"] = True
        request_path.write_text(json.dumps(request), encoding="utf-8")
        with pytest.raises((ValueError, TypeError), match="opacity|real"):
            worker.execute(request_path)
        assert len(calls) == 1
    finally:
        session.close()


@pytest.mark.parametrize("value", [True, "0.5", -1, 2, float("nan"), float("inf")])
def test_invalid_shape_request_creates_no_owned_job(fit_case: Any, value: Any) -> None:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    session = jobs.NativeVideoSession(library)
    try:
        with pytest.raises((ValueError, TypeError)):
            session.submit(
                "source",
                shape_overlay=SimpleNamespace(to_record=lambda: {"opacity": value}),
            )
        assert not (library.root / "video-runs").exists()
    finally:
        session.close()


def test_disabled_shape_preserves_request_hash_and_rejects_unrequested_geometry(
    fit_case: Any, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    _freeze_execution_stamp(jobs, monkeypatch)
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source", shape_overlay=None)["run_id"]
        assert _wait(session, run)["status"] == "succeeded"
        root = library.root / "video-runs" / run
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        assert "shape_overlay" not in request
        assert "shape_overlay" not in session.view(run)
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["hashes"]["controller_hash"] == jobs._digest(
            request["selected_frames"]
        )
        target = root / "overlay/manifest.json"
        output = json.loads(target.read_text(encoding="utf-8"))
        output["shape_overlay"] = {"options": {"opacity": 0.35}}
        target.write_text(json.dumps(output), encoding="utf-8")
        with pytest.raises(ValueError, match="Disabled shape"):
            jobs._outputs(root, request)
    finally:
        session.close()


@pytest.mark.parametrize("kind", ["dictionary", "duck_typed"])
def test_public_shape_contract_rejects_untyped_objects_before_any_activity(
    fit_case: Any, monkeypatch: Any, kind: str
) -> None:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    activity: list[str] = []

    def record() -> dict[str, float]:
        activity.append("to_record")
        return {"opacity": 0.35}

    options = (
        {"opacity": 0.35} if kind == "dictionary" else SimpleNamespace(to_record=record)
    )
    monkeypatch.setattr(
        jobs.MatchingJobService,
        "start",
        lambda *args, **kwargs: activity.append("schedule"),
    )
    session = jobs.NativeVideoSession(library)
    try:
        with pytest.raises(TypeError, match="ShapeOverlayOptions"):
            session.submit("source", shape_overlay=options)
        assert activity == []
        assert not (library.root / "video-runs").exists()
    finally:
        session.close()
