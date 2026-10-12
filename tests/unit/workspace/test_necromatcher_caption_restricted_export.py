"""Canonical tiny restricted seeds use one composer in still and codec exports."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest

from tests.unit.workspace.test_necromatcher_video import video_case as _video_case
from test_scope_fixtures import fixture_review_artifact

video_case = _video_case
pytestmark = pytest.mark.unit


def canonical_capture(case: Any, tmp_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Upgrade only this synthetic fixture through the public capture importer."""
    from src.shared.python.shadow_tracker.source_records import SourceAsset
    from src.shared.python.shadow_tracker.historical_capture import observation_record
    from src.shared.python.pose_estimation.interface import PoseEstimationResult
    from src.shared.python.shadow_tracker.ingestion import compute_frame_hash
    import cv2

    library, old, payload = case
    directory = tmp_path / "canonical-source"
    directory.mkdir()
    with ZipFile(old.path) as archive:
        for name in ("frame-0.png", "frame-1.png", "observations.jsonl"):
            (directory / name).write_bytes(archive.read(name))
    rows = [
        json.loads(line)
        for line in (directory / "observations.jsonl").read_text().splitlines()
    ]
    for row in rows:
        old_path = directory / row["image"]
        row["image"] = row["frame"]["frame_id"] + ".png"
        old_path.rename(directory / row["image"])
        image = cv2.imread(str(directory / row["image"]))
        row["frame"]["frame_sha256"] = compute_frame_hash(
            image.tobytes(), decoder_name=row["frame"]["decoder_name"]
        )
        row["observation"] = observation_record(
            PoseEstimationResult(
                {}, 1.0, 0, raw_keypoints={"origin": np.array([0.5, 0.5])}
            )
        )
    observations = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (directory / "observations.jsonl").write_bytes(observations)
    source = SourceAsset(
        schema_version="shadow-tracker/source/1.0.0",
        asset_id="source-" + "a" * 64,
        source_uri="https://example.invalid/synthetic.mp4",
        content_sha256="a" * 64,
        width_px=320,
        height_px=240,
        rights_status="unknown",
        rights_note="Synthetic fixture",
    )
    receipt = {
        "schema_version": "historical-capture/1.0.0",
        "subject_id": "hogan",
        "qualification": "image_observations_only",
        "physical_time_verified": False,
        "source": source.to_dict(),
        "window_presentation_s": [0, 2 / 30],
        "observations_sha256": hashlib.sha256(observations).hexdigest(),
        "frame_count": 2,
        "detected_count": 2,
    }
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    capture = library.add_capture("caption-source", "practice", directory)
    payload = deepcopy(payload)
    payload.update(
        capture_id=capture.dataset_id,
        capture_hash=capture.metadata["hash"],
        frames=[row["frame"] for row in rows],
    )
    return library, capture, payload


def restriction_options(config: Any, order: tuple[str, ...]) -> dict[str, Any]:
    return {
        "frame_indices": [0, 1],
        "knot_count": 2,
        "coordinate_scales": [1.0] * len(order),
        "config": asdict(config),
        "unknown_visibility_weight": 0.5,
        "budget_wall_s": 300.0,
        "operation": "restrict_initialization",
        "initialization_source": "restricted_spline",
    }


def register_restricted_seed(case: Any, tmp_path: Path) -> tuple[Any, str]:
    from src.shared.python.estimation import CubicHermiteSplineTrajectory
    from src.shared.python.motion_matching.historical_fit import (
        ImageFitConfig,
        ImageFitResult,
        ImageSplineStart,
        restrict_image_spline_interval,
    )
    from src.shared.python.workspace import necromatcher_fit as fit
    from src.shared.python.workspace.necromatcher_capture_identity import (
        capture_identity,
    )
    from src.shared.python.workspace.necromatcher_fit_records import (
        build_native_fit_payload,
    )

    library, capture, payload = canonical_capture(case, tmp_path)
    identity = capture_identity(library, capture.dataset_id)
    review = fixture_review_artifact(tmp_path, identity, 0, 2)
    library.add_source_scope_review("caption-review", "practice", Path(review.path))
    scope = library.load_source_scope_review("caption-review")
    parent = deepcopy(payload)
    order = tuple(parent["coordinate_order"])
    times = np.array([float(f.presentation_time) for f in identity.frames])
    trajectory = CubicHermiteSplineTrajectory(times, len(order))
    model_sha = hashlib.sha256(
        json.dumps(parent["provenance"]["native_definition"], allow_nan=False).encode()
    ).hexdigest()
    start = ImageSplineStart.from_coefficients(
        times,
        trajectory.pack(np.zeros((2, len(order))), np.zeros((2, len(order)))),
        order,
        order,
        model_sha,
    )
    config = ImageFitConfig()
    parent["evidence"]["original_fit"].update(
        {
            **start.to_record(),
            "spline_start": start.to_record(),
            "config": asdict(config),
            "frame_indices": [0, 1],
            "source_times": times.tolist(),
            "q": parent["q"],
        }
    )
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    library.add_fit("caption-parent", "practice", path)
    parent = library.load_fit("caption-parent")
    bound = fit.admit_refit_scope(library, parent, (0, 1), config, requested=scope)
    receipt = restrict_image_spline_interval(start, times[0], times[1])
    result = ImageFitResult(
        times,
        np.asarray(parent["q"]),
        2.0,
        2.0,
        np.zeros((2, 1)),
        2,
        model_sha,
        order,
        False,
        "Synthetic unoptimized restriction",
        times,
        np.asarray(start.spline_coefficients),
        order,
        optimizer_ran=False,
        initial_spline=receipt.restricted_start,
    )
    options = restriction_options(config, order)
    request = {
        "source_fit_id": "caption-parent",
        "source_fit_hash": library.load_asset("caption-parent").metadata["hash"],
        "options": options,
        "source_scope": scope.to_record(),
        "source_scope_binding": fit.scope_binding_record(bound, (0, 1)),
        "spline_interval_restriction": receipt.to_record(),
        "spline_restriction_prior": {
            "policy": "selected_first_parent_pose",
            "frame_index": 0,
        },
        "execution_stamp": {},
    }
    record = build_native_fit_payload(
        request,
        parent,
        result,
        ((0, 1), parent["frames"], result.q),
        dict.fromkeys(
            ("started_at_utc", "source_sha256", "runtime_sha256"), "synthetic"
        ),
        0.0,
    )
    path = tmp_path / "restricted.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    library.add_fit("caption-restricted", "practice", path)
    return library, "caption-restricted"


@pytest.mark.parametrize("kind", ["stills", "video"])
def test_public_restricted_exports_label_seed_without_changing_legacy(
    video_case: Any, tmp_path: Path, kind: str
) -> None:
    from src.shared.python.workspace import (
        CaptionOverlayOptions,
        export_fit_stills,
        export_fit_video,
    )
    from src.shared.python.workspace.necromatcher_caption import (
        validate_caption_manifest,
    )

    library, fit_id = register_restricted_seed(video_case, tmp_path)
    export = export_fit_stills if kind == "stills" else export_fit_video
    compact = export(
        library,
        fit_id,
        tmp_path / "compact",
        selected_frames=(0, 1),
        caption_overlay=CaptionOverlayOptions(),
    )
    assert compact["restricted_initialization_seed"] is True
    assert "authored_initialization_seed" not in compact
    for row in compact["frames"]:
        assert (
            row["caption_overlay"]["lines"][0]["text"]
            == "UNOPTIMIZED RESTRICTED RESEARCH SEED"
        )
    fit = library.load_fit(fit_id)
    validate_caption_manifest(compact, CaptionOverlayOptions(), fit)
    wrong = deepcopy(compact)
    del wrong["restricted_initialization_seed"]
    with pytest.raises(ValueError):
        validate_caption_manifest(wrong, CaptionOverlayOptions(), fit)
    legacy = export(library, fit_id, tmp_path / "legacy", selected_frames=(0, 1))
    explicit = export(
        library,
        fit_id,
        tmp_path / "explicit",
        selected_frames=(0, 1),
        caption_overlay=None,
    )
    assert legacy == explicit
    assert "restricted_initialization_seed" not in legacy
    for index in (0, 1):
        name = f"frame-{index:06d}.png"
        assert (tmp_path / "legacy" / name).read_bytes() == (
            tmp_path / "explicit" / name
        ).read_bytes()


@pytest.mark.parametrize("kind", ["stills", "video"])
def test_public_export_rejects_changed_restriction_parent_before_output(
    video_case: Any, tmp_path: Path, kind: str
) -> None:
    from src.shared.python.workspace import (
        CaptionOverlayOptions,
        export_fit_stills,
        export_fit_video,
    )

    library, fit_id = register_restricted_seed(video_case, tmp_path)
    parent = Path(library.load_asset("caption-parent").path)
    parent.write_bytes(parent.read_bytes() + b" ")
    output = tmp_path / "forged-export"
    export = export_fit_stills if kind == "stills" else export_fit_video
    with pytest.raises(ValueError):
        export(
            library,
            fit_id,
            output,
            selected_frames=(0, 1),
            caption_overlay=CaptionOverlayOptions(),
        )
    assert not output.exists()
