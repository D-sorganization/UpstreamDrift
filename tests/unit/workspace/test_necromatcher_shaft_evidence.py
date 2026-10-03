"""Shaft bearing annotations bind to verified original capture bytes."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.shared.python.motion_matching.historical_fit.shaft_observations import (
    ShaftAxisEvidence,
)
from src.shared.python.workspace.necromatcher_shaft_evidence import (
    BoundShaftEvidence,
    bind_shaft_axis_evidence,
)
from tests.unit.motion_matching.test_shaft_observations import evidence

pytestmark = pytest.mark.unit


@pytest.fixture
def binding_inputs(monkeypatch):
    module = importlib.import_module(
        "src.shared.python.workspace.necromatcher_shaft_evidence"
    )
    png = b"owned synthetic image bytes"
    value = evidence()
    value = replace(
        value,
        frames=(
            replace(
                value.frames[0], png_sha256="sha256:" + hashlib.sha256(png).hexdigest()
            ),
        ),
    )
    row = {
        "frame": value.frames[0].frame.to_dict(),
        "image_width": 100,
        "image_height": 80,
    }
    asset = SimpleNamespace(
        kind="image_capture",
        metadata={"hash": value.capture_sha256, "source_sha256": "a" * 64},
    )
    library = SimpleNamespace(load_asset=lambda identity: asset)

    class Review:
        capture_id = "capture"
        frame_count = 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def frame(self, index):
            return deepcopy(row)

        def image(self, index):
            return png

    monkeypatch.setattr(module, "CaptureReview", lambda lib, identity: Review())
    return library, value, row, asset


def test_binding_preserves_distinct_hashes_and_clock(binding_inputs):
    library, value, _, _ = binding_inputs
    bound = bind_shaft_axis_evidence(library, value)
    assert bound.evidence == value and bound.reviewed_frame_count == 1
    assert bound.observed_segment_count == 1
    assert value.frames[0].png_sha256 != "sha256:" + value.frames[0].frame.frame_sha256
    assert bound.source_clock_sha256.startswith("sha256:")
    assert bind_shaft_axis_evidence(library, value) == bound


@pytest.mark.parametrize(
    "fault",
    ["capture", "source", "png", "pixel", "pts", "timebase", "size", "index", "shot"],
)
def test_each_source_identity_tamper_rejected(binding_inputs, fault):
    library, value, row, asset = binding_inputs
    if fault == "capture":
        asset.metadata["hash"] = "sha256:" + "f" * 64
    elif fault == "source":
        asset.metadata["source_sha256"] = "f" * 64
    elif fault == "png":
        value = replace(
            value, frames=(replace(value.frames[0], png_sha256="sha256:" + "f" * 64),)
        )
    elif fault == "pixel":
        row["frame"]["frame_sha256"] = "f" * 64
    elif fault == "pts":
        row["frame"]["pts_ticks"] += 1
    elif fault == "timebase":
        row["frame"]["timebase_denominator"] = 31
    elif fault == "size":
        row["image_width"] = 99
    elif fault == "index":
        value = replace(value, frames=(replace(value.frames[0], frame_index=1),))
    else:
        row["frame"]["shot_id"] = "other"
    with pytest.raises(ValueError):
        bind_shaft_axis_evidence(library, value)


def test_sparse_abstention_remains_explicit(binding_inputs):
    library, value, _, _ = binding_inputs
    old = value.frames[0]
    segment = replace(
        old.segment,
        status="ambiguous",
        points_px=None,
        confidence=None,
        visibility=None,
        sigma_px=None,
    )
    value = replace(value, frames=(replace(old, segment=segment),))
    assert bind_shaft_axis_evidence(library, value).observed_segment_count == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"evidence": None},
        {"source_clock_sha256": "bad"},
        {"reviewed_frame_count": -1},
        {"observed_segment_count": True},
        {"observed_segment_count": 0},
    ],
)
def test_bound_receipt_rejects_forged_shape_or_counts(binding_inputs, updates):
    library, value, _, _ = binding_inputs
    bound = bind_shaft_axis_evidence(library, value)
    with pytest.raises(ValueError):
        replace(bound, **updates)


@pytest.mark.parametrize("first", ["historical_fit", "workspace"])
def test_curated_imports_are_sdk_free_in_both_orders(first):
    other = (
        "workspace" if first == "historical_fit" else "motion_matching.historical_fit"
    )
    first = "motion_matching.historical_fit" if first == "historical_fit" else first
    code = f"import src.shared.python.{first}; import src.shared.python.{other}; import sys; assert not set(('mujoco','pydrake','pinocchio','opensim')) & set(sys.modules)"
    root = Path(__file__).parents[3]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root), str(root / "src")))}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_real_public_archive_binding_preserves_original_bytes(tmp_path):
    import numpy as np
    from src.shared.python.pose_estimation.interface import PoseEstimationResult
    from src.shared.python.shadow_tracker.historical_capture import (
        CaptureWindow,
        export_capture,
    )
    from src.shared.python.shadow_tracker.source_records import FrameIdentity
    from src.shared.python.motion_matching.historical_fit.shaft_observations import (
        ShaftAxisSegment,
        SourceBoundShaftFrame,
    )
    from src.shared.python.workspace import NecromatcherLibrary, CaptureReview

    av = pytest.importorskip("av")
    pytest.importorskip("cv2")
    video = tmp_path / "owned.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(
                    np.zeros((32, 32, 3), dtype=np.uint8), format="bgr24"
                )
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    class MissingEstimator:
        def estimate_from_image(self, image, timestamp_ms):
            return PoseEstimationResult({}, 0, 0)

    capture = tmp_path / "capture"
    receipt = export_capture(
        video,
        capture,
        CaptureWindow(start_s=0, end_s=0.3),
        subject_id="test",
        estimator=MissingEstimator(),
        detector_identity={"name": "synthetic_test_only"},
    )
    library = NecromatcherLibrary.create(tmp_path / "library")
    library.add_player("test", "Synthetic")
    library.add_swing("swing", "test", "Synthetic")
    asset = library.add_capture("capture", "swing", capture)
    before = Path(asset.path).read_bytes()
    with CaptureReview(library, "capture") as review:
        item = SourceBoundShaftFrame(
            1,
            FrameIdentity.from_dict(review.frame(1)["frame"]),
            "sha256:" + hashlib.sha256(review.image(1)).hexdigest(),
            ShaftAxisSegment(
                "unreviewed", None, "test", "No annotation", None, None, None
            ),
        )
    value = ShaftAxisEvidence(
        "capture",
        asset.metadata["hash"],
        "sha256:" + receipt["source"]["content_sha256"],
        (32, 32),
        (item,),
    )
    bound = bind_shaft_axis_evidence(library, value)
    assert bound.reviewed_frame_count == 1 and bound.observed_segment_count == 0
    assert Path(asset.path).read_bytes() == before
