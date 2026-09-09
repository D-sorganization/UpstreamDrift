"""Swing selection preserves source pixels, clocks and prior analyses (#9860)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.rig.bundle import load_bundle
from src.motion_capture.rig.edits import (
    EDITS_FILE,
    CropRect,
    SessionEdits,
    ViewEdit,
    load_edits,
    save_edits,
)
from src.motion_capture.rig.ingest import FramePose, ingest_bundle, ingest_view
from src.motion_capture.reconstruct.bundle import observations_from_views

from .test_ingest import FakeEstimator, _bundle
from .test_ingest_alignment import _timing, _with_timing

pytestmark = pytest.mark.unit


class CentreEstimator(FakeEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def estimate(self, image: np.ndarray, timestamp_ms: int) -> FramePose:
        self.timestamps.append(timestamp_ms)
        self.shapes.append(image.shape)
        return FramePose(np.array([[0.5, 0.5], [0.0, 0.0]]), np.ones(2))


def test_trim_crop_restore_original_camera_pixels_and_frame_clock(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    entry = load_bundle(root)[1].recordings[0]
    source = root / entry.file
    original = source.read_bytes()
    edit = ViewEdit(first=2, last=4, crop=CropRect(x=8, y=6, width=32, height=24))
    estimator = CentreEstimator()
    result = ingest_view(entry, source, estimator, edit=edit)
    assert estimator.timestamps == [67, 100, 133]
    assert estimator.shapes == [(24, 32, 3)] * 3
    assert result.frames_total == 6  # source timeline, not number of inference calls
    assert result.provenance["frames_processed"] == 3
    assert result.frames[0]["time_s"] == pytest.approx(2 / 30)
    assert result.frames[0]["keypoints_px"] == [[24.0, 18.0], [8.0, 6.0]]
    assert (result.width, result.height) == (64, 48)
    assert result.provenance["edit"] == edit.model_dump(mode="json")
    assert source.read_bytes() == original
    payload = result.model_dump(mode="json")
    stacked = observations_from_views({"a": payload, "b": payload}, ["a", "b"])
    assert np.isfinite(stacked.pixels[:, 4]).all()
    assert not np.isfinite(stacked.pixels[:, :2]).any()


def test_bundle_recipe_applies_per_view_and_keeps_reference_timing(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    _with_timing(root, _timing())
    recipe = SessionEdits(
        views={"a": ViewEdit(first=1, last=2), "b": ViewEdit(first=3, last=4)}
    )
    save_edits(root, recipe)
    out = root / "observations"
    ingest_bundle(root, out, CentreEstimator)
    b = json.loads((out / "b.json").read_text(encoding="utf-8"))
    assert [r["time_s"] for r in b["frames"]] == pytest.approx([0.1, 4 / 30])
    assert b["frames"][0]["time_ref_s"] == pytest.approx(0.0)
    assert load_edits(root) == recipe
    with pytest.raises(ValueError, match="editable copy"):
        save_edits(root, SessionEdits())
    assert load_edits(root) == recipe


@pytest.mark.parametrize(
    "payload",
    [
        {"first": 4, "last": 2},
        {"first": -1},
        {"first": 1.5},
        {"crop": {"x": 0, "y": 0, "width": 0, "height": 2}},
    ],
)
def test_invalid_recipe_fields_fail(payload: dict) -> None:
    with pytest.raises(ValueError):
        ViewEdit.model_validate(payload)


@pytest.mark.parametrize(
    "edit",
    [
        ViewEdit(first=6),
        ViewEdit(last=6),
        ViewEdit(crop=CropRect(x=50, y=0, width=30, height=20)),
    ],
)
def test_out_of_media_bounds_never_saves(tmp_path: Path, edit: ViewEdit) -> None:
    root = _bundle(tmp_path)
    with pytest.raises(ValueError):
        save_edits(root, SessionEdits(views={"a": edit}))
    assert not (root / EDITS_FILE).exists()


def test_missing_and_corrupt_recipe_are_distinct(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    assert load_edits(root) == SessionEdits()
    (root / EDITS_FILE).write_text('{"schema_version":"future"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_edits(root)


def test_limit_counts_selected_frames_and_decode_shortfall_fails(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path)
    entry = load_bundle(root)[1].recordings[0]
    estimator = CentreEstimator()
    result = ingest_view(
        entry, root / entry.file, estimator, edit=ViewEdit(first=3), max_frames=2
    )
    assert estimator.timestamps == [100, 133]
    assert result.frames_total == 6
    assert result.provenance["frames_processed"] == 2
    with pytest.raises(ValueError, match="ended"):
        ingest_view(
            entry.model_copy(update={"frames": 10}),
            root / entry.file,
            CentreEstimator(),
            edit=ViewEdit(first=8, last=9),
        )
