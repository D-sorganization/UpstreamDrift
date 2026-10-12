"""Reviewed-window dependencies use the existing immutable library spine."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.shared.python.workspace import NecromatcherLibrary
from src.shared.python.workspace.necromatcher_capture_identity import capture_identity
from src.shared.python.workspace.necromatcher_source_scope import bind_source_fit_scope
from tests.unit.workspace.hypothesis_fixture import imported_capture
from tests.unit.workspace.test_scope_fixtures import fixture_review_artifact

pytestmark = pytest.mark.unit


@pytest.fixture
def scope_asset_case(tmp_path: Path) -> tuple:
    library = NecromatcherLibrary.create(tmp_path / "library")
    library.add_player("hogan", "Ben Hogan")
    library.add_swing("practice", "hogan", "Practice")
    imported_capture(library, tmp_path / "capture-source")
    identity = capture_identity(library, "exact-capture")
    reference = fixture_review_artifact(tmp_path, identity, 0, 2)
    return library, Path(reference.path), identity


def test_review_receipt_survives_restart_without_external_file(
    scope_asset_case: tuple,
) -> None:
    library, source, identity = scope_asset_case
    raw = source.read_bytes()
    saved = library.add_source_scope_review("review-v1", "practice", source)
    source.unlink()
    fresh = NecromatcherLibrary(library.root)
    scope = fresh.load_source_scope_review("review-v1")
    assert saved.kind == "scope_review"
    assert scope.review.artifact.path == "assets/review-v1.json"
    assert Path(saved.path).read_bytes() == raw
    assert bind_source_fit_scope(identity, scope, fresh.root).scope == scope
    assert scope.first_frame == 0 and scope.end_exclusive_frame == 2
    assert scope.review.contact_calibrated is False


def test_review_cannot_bind_another_swing(scope_asset_case: tuple) -> None:
    library, source, _ = scope_asset_case
    library.add_swing("other", "hogan", "Other")
    with pytest.raises(ValueError, match="swing|session"):
        library.add_source_scope_review("wrong", "other", source)
    assert not (library.root / "assets/wrong.json").exists()
    assert library.assets("other") == []


def test_review_rejects_declared_boundary_tamper(scope_asset_case: tuple) -> None:
    library, source, _ = scope_asset_case
    record = json.loads(source.read_bytes())
    record["excluded_identity"]["pts_ticks"] += 1
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary|frame|identity"):
        library.add_source_scope_review("invalid", "practice", source)
    assert not (library.root / "assets/invalid.json").exists()


def test_export_carries_exact_registered_review_bytes_and_relative_identity(
    scope_asset_case: tuple,
    tmp_path: Path,
) -> None:
    library, source, _ = scope_asset_case
    raw = source.read_bytes()
    library.add_source_scope_review("review-v1", "practice", source)
    destination = tmp_path / "swing.zip"
    library.export_swing("practice", destination)
    with ZipFile(destination) as archive:
        assert archive.read("assets/review-v1.json") == raw
        manifest = json.loads(archive.read("manifest.json"))
        review = next(a for a in manifest["assets"] if a["kind"] == "scope_review")
        assert review["path"] == "assets/review-v1.json"
        assert review["metadata"]["qualification"] == "authored_uncalibrated"


def test_changed_stored_review_prevents_recall_and_export(
    scope_asset_case: tuple,
    tmp_path: Path,
) -> None:
    library, source, _ = scope_asset_case
    saved = library.add_source_scope_review("review-v1", "practice", source)
    Path(saved.path).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        library.load_source_scope_review("review-v1")
    destination = tmp_path / "bad.zip"
    with pytest.raises(ValueError, match="hash"):
        library.export_swing("practice", destination)
    assert not destination.exists()


def test_review_asset_identity_is_immutable(scope_asset_case: tuple) -> None:
    from src.shared.python.core.contracts.exceptions import StateError

    library, source, _ = scope_asset_case
    saved = library.add_source_scope_review("review-v1", "practice", source)
    with pytest.raises(StateError, match="already exists"):
        library.add_source_scope_review("review-v1", "practice", source)
    assert library.load_asset("review-v1") == saved


def test_registered_review_can_travel_through_workspace_handoff(
    scope_asset_case: tuple,
) -> None:
    from src.shared.python.workspace.artifact_handoff import WorkspaceHandoff

    library, source, _ = scope_asset_case
    library.add_source_scope_review("review-v1", "practice", source)
    scope = library.load_source_scope_review("review-v1")
    handoff = WorkspaceHandoff(
        handoff_id="review-transfer",
        project_id="necromatcher",
        session_id="practice",
        subject_id="hogan",
        engine="mujoco",
        model_id="research-model",
        club={},
        units={"length": "m"},
        frame="world",
        timebase={"physical_time_qualified": False},
        parameters={},
        inputs=(scope.review.artifact,),
    )
    handoff.inputs[0].verify_on_disk(library.root)
    assert handoff.inputs[0].path == "assets/review-v1.json"


def test_registered_scope_admits_only_canonical_selected_frames(
    scope_asset_case: tuple,
) -> None:
    from src.shared.python.workspace.necromatcher_fit import (
        admit_refit_scope,
        scope_binding_record,
    )

    library, source, identity = scope_asset_case
    library.add_source_scope_review("review-v1", "practice", source)
    scope = library.load_source_scope_review("review-v1")
    parent = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {},
    }
    bound = admit_refit_scope(library, parent, (0, 1), requested=scope)
    binding = scope_binding_record(bound, (0, 1))
    first = identity.frames[0].presentation_time
    last = identity.frames[1].presentation_time
    assert binding["first_pts"] == [first.numerator, first.denominator]
    assert binding["last_pts"] == [last.numerator, last.denominator]
    with pytest.raises(ValueError, match="outside source scope"):
        admit_refit_scope(library, parent, (0, 2), requested=scope)
