"""Video provenance binds reviewed scope without altering unscoped recipes."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from tests.unit.workspace.test_necromatcher_source_scope import make_scope
from tests.unit.workspace.test_necromatcher_video_jobs import (
    _fake_export,
    _freeze_execution_stamp,
    _wait,
)
from src.shared.python.workspace import necromatcher_video_jobs as jobs
from src.shared.python.workspace import necromatcher_video as video

pytestmark = pytest.mark.unit


def scope_fields(tmp_path: Path) -> dict:
    identity, scope = make_scope(tmp_path)
    from src.shared.python.workspace.necromatcher_fit import scope_binding_record
    from src.shared.python.workspace.necromatcher_source_scope import (
        bind_source_fit_scope,
    )

    bound = bind_source_fit_scope(identity, scope)
    return {
        "source_fit_scope": scope.to_record(),
        "source_fit_scope_binding": scope_binding_record(bound, (0, 2)),
    }


def test_snapshot_is_detached_and_legacy_absence_exact(tmp_path: Path) -> None:
    fields = scope_fields(tmp_path)
    fit = {"provenance": deepcopy(fields)}
    result = video.video_scope_provenance(fit)
    assert result == fields
    result["source_fit_scope_binding"]["frame_indices"].append(3)
    assert fit["provenance"] == fields
    assert video.video_scope_provenance({"provenance": {}}) == {}
    with pytest.raises(ValueError, match="scope"):
        video.video_scope_provenance({"provenance": {"source_fit_scope": None}})


def test_manifest_carries_scope_without_body_or_clock_changes(
    tmp_path: Path, monkeypatch
) -> None:
    fields = scope_fields(tmp_path)
    binding = SimpleNamespace(
        fit_id="fit",
        fit_hash="hash",
        model_id="model",
        model_hash="modelhash",
        fit={
            "capture_id": "capture",
            "capture_hash": "capturehash",
            "provenance": fields,
        },
    )
    library = SimpleNamespace(load_asset=lambda _: SimpleNamespace(metadata={}))
    monkeypatch.setattr(video, "_rigid_segments", lambda _: [])
    media = {"schema": "necromatcher/source-overlay-video/1", "image_size": [320, 240]}
    scoped = video._export_manifest(binding, library, [], [], {}, [], None, media)
    assert all(scoped[key] == value for key, value in fields.items())
    binding.fit["provenance"] = {}
    legacy = video._export_manifest(binding, library, [], [], {}, [], None, media)
    assert {key: value for key, value in scoped.items() if key not in fields} == legacy


def test_job_snapshots_scope_and_receipt_tamper_disables_download(
    fit_case, tmp_path: Path, monkeypatch
) -> None:
    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    fit = library.load_fit("source")
    fields = scope_fields(tmp_path / "review")
    fit["provenance"].update(deepcopy(fields))
    monkeypatch.setattr(library, "load_fit", lambda _: deepcopy(fit))
    asset_id = fields["source_fit_scope"]["review"]["artifact"]["artifact_id"]
    review_file = library.root / "assets" / "review.json"
    review_file.write_bytes(b"authenticated review boundary double")
    assets = library.assets
    monkeypatch.setattr(
        library,
        "assets",
        lambda swing: [
            *assets(swing),
            SimpleNamespace(
                dataset_id=asset_id,
                path=str(review_file),
                kind="scope_review",
                metadata={"hash": "sha256:" + "a" * 64},
            ),
        ],
    )
    _freeze_execution_stamp(jobs, monkeypatch)

    def worker(path, budget, cancelled):
        result = _fake_export(path, budget, cancelled)
        request = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.parent / "overlay" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({key: request[key] for key in fields})
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        from src.shared.python.workspace.artifact_handoff import compute_file_sha256

        result["manifest_sha256"] = compute_file_sha256(manifest_path).removeprefix(
            "sha256:"
        )
        return result

    monkeypatch.setattr(jobs, "_execute_worker", worker)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        view = _wait(session, run)
        assert view["status"] == "succeeded"
        assert view["download_available"] is True
        assert all(view[key] == value for key, value in fields.items())
        root = library.root / "video-runs" / run
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        assert all(request[key] == value for key, value in fields.items())
        fit["provenance"]["source_fit_scope_binding"]["last_pts"] = [99, 1]
        assert (
            session.view(run)["source_fit_scope_binding"]
            == fields["source_fit_scope_binding"]
        )
        with pytest.raises(ValueError, match="scope"):
            session.view_for_fit("source", run)
        review_file.write_bytes(b"changed review receipt")
        assert session.view(run)["download_available"] is False
        manifest = json.loads(
            (root / "overlay" / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["source_fit_scope_binding"]["first_pts"] = [1, 1]
        (root / "overlay" / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="scope"):
            jobs._outputs(root, request)
    finally:
        session.close()


def test_disabled_scope_manifest_rejects_unrequested_scope(
    fit_case, tmp_path: Path
) -> None:
    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    fit = library.load_fit("source")
    request = {
        "source_fit_id": "source",
        "source_fit_hash": library.load_asset("source").metadata["hash"],
        **{
            key: fit[key]
            for key in ("model_id", "model_hash", "capture_id", "capture_hash")
        },
        "selected_frames": [0, 2],
    }
    root = tmp_path / "run"
    root.mkdir()
    path = root / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    _fake_export(path, 1, lambda: False)
    manifest_path = root / "overlay" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(scope_fields(tmp_path / "review"))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="scope"):
        jobs._outputs(root, request)


def test_outside_fit_shaft_rejected_before_run_or_scheduler(
    fit_case, monkeypatch
) -> None:
    from dataclasses import replace
    from tests.unit.motion_matching.test_shaft_observations import (
        evidence,
        frame_identity,
    )
    from src.shared.python.workspace.necromatcher_shaft_evidence import (
        BoundShaftEvidence,
    )

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    fit = library.load_fit("source")
    value = evidence()
    value = replace(
        value,
        capture_id=fit["capture_id"],
        capture_sha256=fit["capture_hash"],
        frames=(replace(value.frames[0], frame_index=3, frame=frame_identity(3)),),
    )
    monkeypatch.setattr(
        jobs,
        "bind_fit_shaft_evidence",
        lambda *args: BoundShaftEvidence(value, "sha256:" + "e" * 64, 1, 1),
    )
    _freeze_execution_stamp(jobs, monkeypatch)
    session = jobs.NativeVideoSession(library)
    calls = []

    def scheduler(*args, **kwargs):
        calls.append(args)
        raise AssertionError("scheduler invoked for out-of-domain evidence")

    monkeypatch.setattr(session._service, "start", scheduler)
    try:
        with pytest.raises(ValueError, match="Shaft review frames"):
            session.submit("source", value)
        assert calls == []
        assert not (library.root / "video-runs").exists()
    finally:
        session.close()
