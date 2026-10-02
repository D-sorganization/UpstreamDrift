"""Persisted overlay recall stays fit-scoped and never schedules work."""

from __future__ import annotations

import json

import pytest

from src.shared.python.workspace.necromatcher_video_jobs import NativeVideoSession

pytestmark = pytest.mark.unit


def _record(library, fit_id, run_id, **changes):
    fit = library.load_fit(fit_id)
    request = {
        "kind": "necromatcher/video-job/1",
        "run_id": run_id,
        "source_fit_id": fit_id,
        "source_fit_hash": library.load_asset(fit_id).metadata["hash"],
        "model_id": fit["model_id"],
        "model_hash": fit["model_hash"],
        "capture_id": fit["capture_id"],
        "capture_hash": fit["capture_hash"],
        "execution_started": False,
    }
    request.update(changes)
    directory = library.root / "video-runs" / run_id
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")


def test_saved_overlay_listing_is_scoped_without_job_submission(fit_case):
    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    library.add_fit("other", "practice", source)
    _record(library, "source", "a" * 32)
    _record(library, "other", "b" * 32)
    session = NativeVideoSession(library)
    try:
        runs = session.stored_runs("source")
        assert [run["run_id"] for run in runs] == ["a" * 32]
        assert runs[0]["status"] == "pending"
        assert not runs[0]["control_available"]
        assert not runs[0]["download_available"]
        assert session.view_for_fit("source", "a" * 32) == runs[0]
        with pytest.raises(ValueError, match="fit"):
            session.view_for_fit("source", "b" * 32)
    finally:
        session.close()


def test_saved_overlay_listing_empty_requires_an_actual_fit(fit_case):
    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    session = NativeVideoSession(library)
    try:
        assert session.stored_runs("source") == []
        with pytest.raises(KeyError):
            session.stored_runs("missing")
    finally:
        session.close()


def test_unpublished_foreign_directory_does_not_hide_owned_run(fit_case, caplog):
    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    _record(library, "source", "a" * 32)
    (library.root / "video-runs" / ("b" * 32)).mkdir()
    session = NativeVideoSession(library)
    try:
        assert [run["run_id"] for run in session.stored_runs("source")] == ["a" * 32]
        assert "unreadable stored overlay" in caplog.text
    finally:
        session.close()


def test_saved_overlay_recall_rejects_manifest_symlink(fit_case, monkeypatch):
    from pathlib import Path

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    _record(library, "source", "a" * 32)
    previous = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name == "run_manifest.json" or previous(path),
    )
    session = NativeVideoSession(library)
    try:
        with pytest.raises(ValueError, match="manifest.*symlink|regular"):
            session.view_for_fit("source", "a" * 32)
    finally:
        session.close()


@pytest.mark.parametrize("field", ["source_fit_hash", "capture_hash", "model_hash"])
def test_saved_overlay_recall_rejects_changed_parent_binding(fit_case, field):
    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    _record(library, "source", "a" * 32, **{field: "sha256:" + "f" * 64})
    session = NativeVideoSession(library)
    try:
        with pytest.raises(ValueError, match="hash|identity"):
            session.stored_runs("source")
        with pytest.raises(ValueError, match="hash|identity"):
            session.view_for_fit("source", "a" * 32)
    finally:
        session.close()
