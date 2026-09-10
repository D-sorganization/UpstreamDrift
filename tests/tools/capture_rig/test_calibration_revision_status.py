"""A recalibrated capture cannot inherit completion from an older reconstruction."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.motion_capture.rig.documents import write_document
from src.motion_capture.rig.edits import SessionEdits, save_edits
from src.tools.capture_rig.goal_catalog import load_catalog
from src.tools.capture_rig.goal_planner import resolve
from src.tools.capture_rig.wizard_evidence import (
    CalibrationReview,
    inspect_capture,
    reconstruction_invalidation,
)
from tests.tools.capture_rig.test_wizard_evidence import _capture, _reviewed_calibration

pytestmark = pytest.mark.unit


def test_matching_calibration_keeps_reconstruction_current(tmp_path: Path) -> None:
    root, _, media = _capture(tmp_path)
    review, _ = _reviewed_calibration(root, tmp_path)
    media = replace(
        media, reconstruction={"camera_source_sha256": review.calibration_sha256}
    )
    assert reconstruction_invalidation(media, review) == {}


@pytest.mark.parametrize("recorded", [None, "0" * 64])
def test_changed_or_unverified_calibration_requires_reconstruction_again(
    tmp_path: Path, recorded: str | None
) -> None:
    root, _, media = _capture(tmp_path)
    review, _ = _reviewed_calibration(root, tmp_path)
    old = {"camera_source_sha256": recorded}
    media = replace(media, reconstruction=old)
    invalid = reconstruction_invalidation(media, review)
    assert "reconstruct again" in invalid["reconstruct"].lower()
    assert media.reconstruction == old


def test_new_capture_has_no_stale_reconstruction(tmp_path: Path) -> None:
    root, _, media = _capture(tmp_path)
    review, _ = _reviewed_calibration(root, tmp_path)
    assert reconstruction_invalidation(media, review) == {}


def test_wizard_refresh_blocks_old_reconstruction_and_downstream_model(
    tmp_path: Path,
) -> None:
    root, library, media = _capture(tmp_path)
    review, path = _reviewed_calibration(root, tmp_path)
    edits = SessionEdits()
    save_edits(root, edits)
    folder = root / "observations"
    folder.mkdir()
    write_document(
        folder / "observations.json",
        {"provenance": {"edits": edits.model_dump(mode="json")}},
    )
    views = []
    for view in media.views:
        output = folder / f"{view.view}.json"
        write_document(output, {})
        views.append(replace(view, observations=output))
    media = replace(
        media,
        views=tuple(views),
        reliability={},
        reconstruction={"camera_source_sha256": review.calibration_sha256},
    )
    route = resolve(load_catalog(), ["fit_model"])
    current = inspect_capture(
        route, media, library.root, review=review, start_file=path
    )
    assert current.states["step.reconstruct"].status == "done"
    path.write_bytes(path.read_bytes() + b"\n")
    updated_review = CalibrationReview.confirmed(root, path)
    refreshed = inspect_capture(
        route, media, library.root, review=updated_review, start_file=path
    )
    assert refreshed.states["step.reconstruct"].status == "blocked"
    assert "Reconstruct again" in refreshed.states["step.reconstruct"].reason
    assert "step.reconstruct" in refreshed.states["step.fit_model"].prerequisites
