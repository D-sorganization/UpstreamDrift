"""Caption recipes retain scheduling ownership, hashing and guarded manifests."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from tests.unit.workspace.test_necromatcher_video_jobs import (
    _fake_export,
    _freeze_execution_stamp,
    _wait,
)

from tests.unit.workspace.test_necromatcher_video import video_case as _video_case

video_case = _video_case

pytestmark = pytest.mark.unit


def _caption_worker(path: Path, budget: float, cancelled: Any) -> dict[str, Any]:
    from src.shared.python.workspace import NecromatcherLibrary
    from src.shared.python.workspace.necromatcher_caption import (
        CaptionFrame,
        CaptionOverlayOptions,
        caption_layout,
        caption_provenance,
    )

    _fake_export(path, budget, cancelled)
    request = json.loads(path.read_text(encoding="utf-8"))
    target = path.parent / "overlay/manifest.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    manifest["caption_overlay"] = caption_provenance(
        CaptionOverlayOptions.from_record(request["caption_overlay"])
    )
    fit = NecromatcherLibrary(request["library_root"]).load_fit(
        request["source_fit_id"]
    )
    manifest["image_size"] = [320, 240]
    manifest["frames"] = []
    for index, identity in zip(fit["frame_indices"], fit["frames"], strict=True):
        pts = Fraction(
            identity["pts_ticks"] * identity["timebase_numerator"],
            identity["timebase_denominator"],
        )
        frame = CaptionFrame(index, pts, None, 0)
        manifest["frames"].append(
            {
                "frame_index": index,
                "frame": identity,
                "matched_rms_pixels": None,
                "matched_marker_count": 0,
                "caption_overlay": caption_layout(
                    (320, 240), frame, CaptionOverlayOptions()
                ).to_record(),
            }
        )
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return {"manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def test_captured_caption_option_is_hashed_recalled_and_tamper_guarded(
    video_case: Any, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.workspace.necromatcher_caption import CaptionOverlayOptions

    library, _, _ = video_case
    _freeze_execution_stamp(jobs, monkeypatch)
    monkeypatch.setattr(jobs, "_execute_worker", _caption_worker)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("video-fit", caption_overlay=CaptionOverlayOptions())[
            "run_id"
        ]
        result = _wait(session, run)
        assert result["status"] == "succeeded", result["message"]
        root = library.root / "video-runs" / run
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        assert request["caption_overlay"] == {"style": "compact_research_v1"}
        hashes = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))[
            "hashes"
        ]
        assert hashes["controller_hash"] == jobs._digest(
            {
                "selected_frames": request["selected_frames"],
                "caption_overlay": request["caption_overlay"],
            }
        )
        assert result["caption_overlay"] == request["caption_overlay"]
        assert (
            session.view_for_fit("video-fit", run)["caption_overlay"]
            == request["caption_overlay"]
        )
        assert session.download(run).is_file()
        target = root / "overlay/manifest.json"
        record = json.loads(target.read_text(encoding="utf-8"))
        record["caption_overlay"]["physical_time_qualified"] = True
        target.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="caption|Caption"):
            jobs._outputs(root, request)
    finally:
        session.close()


@pytest.mark.parametrize("bad", [{"style": "compact_research_v1"}, object()])
def test_wrong_caption_type_rejects_before_scheduling(
    native_fit_case: Any, bad: Any
) -> None:
    from src.shared.python.workspace.necromatcher_video_jobs import NativeVideoSession

    library, _, _ = native_fit_case
    session = NativeVideoSession(library)
    try:
        with pytest.raises(TypeError, match="Caption"):
            session.submit("absent", caption_overlay=bad)
        assert not (library.root / "video-runs").exists()
    finally:
        session.close()


def test_self_consistent_caption_cannot_authorize_changed_source_size(
    video_case: Any, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.workspace.necromatcher_caption import (
        CaptionFrame,
        CaptionOverlayOptions,
        caption_layout,
    )

    library, _, _ = video_case
    _freeze_execution_stamp(jobs, monkeypatch)
    monkeypatch.setattr(jobs, "_execute_worker", _caption_worker)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("video-fit", caption_overlay=CaptionOverlayOptions())[
            "run_id"
        ]
        assert _wait(session, run)["status"] == "succeeded"
        root = library.root / "video-runs" / run
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        target = root / "overlay/manifest.json"
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["image_size"] = [640, 480]
        for row in manifest["frames"]:
            identity = row["frame"]
            pts = Fraction(
                identity["pts_ticks"] * identity["timebase_numerator"],
                identity["timebase_denominator"],
            )
            row["caption_overlay"] = caption_layout(
                (640, 480),
                CaptionFrame(row["frame_index"], pts, None, 0),
                CaptionOverlayOptions(),
            ).to_record()
        target.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="source dimensions"):
            jobs._outputs(root, request)
    finally:
        session.close()
