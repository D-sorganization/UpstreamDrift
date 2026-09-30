"""Unit tests for the Shadow Tracker segmentation adapter, provenance, and benchmark (MMR-13, #11099).

The adapter must never report successful model inference without the pinned
checkpoint having been verified, loaded, and executed against real decoded
pixels. The benchmark must score against independently recorded gold
artifacts loaded from an evidence directory, and must emit a typed blocked
result (no numeric metrics) whenever that evidence is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from shared.python.shadow_tracker._validation import (
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
)
from shared.python.shadow_tracker.contracts import SegmentationRequest
from shared.python.shadow_tracker.mask_records import MaskFrame
from shared.python.shadow_tracker.model_segmentation import (
    PINNED_MODELS,
    RealSegmentationAdapter,
    SegmentationModelCard,
    evaluate_segmentation_benchmark,
    verify_checkpoint,
)
from shared.python.shadow_tracker.segmentation import ManualMaskProvider
from shared.python.shadow_tracker.source_records import FrameIdentity

pytestmark = pytest.mark.unit

EVIDENCE_DIR = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "development"
    / "matched_swing_program"
    / "evidence"
    / "segmentation"
)

_FABRICATED_SAM_VIT_B_PIN = (
    "9f8a3d7b6e5c4a1f2e8b0d9c7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f"
)
_FABRICATED_MOBILESAM_PIN = (
    "4a2c8e1f5b9d3a7e6c0f8d2b4a1e9c7f5d3b1a0e8c6f4d2b9a7e5c3b1a9f0d8e"
)
_INVALID_PIN = "z" * 64
_OPERATOR_PIN = "a" * 64


def _pattern_pixels(high: int, low: int, width: int, height: int) -> np.ndarray:
    """Deterministic non-uniform RGB pattern so differing pixels hash differently."""
    grid = np.indices((height, width))
    ramp = (grid[0] * 7 + grid[1] * 11) % 3
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = high + ramp
    pixels[:, :, 1] = low + ramp
    pixels[:, :, 2] = (high + low) // 2 + ramp
    return pixels


def _segment_request() -> SegmentationRequest:
    return SegmentationRequest(
        shot_id="shot-01",
        frame_ids=("f-01", "f-02"),
        options={
            "asset_id": "asset-01",
            "frames": {
                "f-01": {
                    "pixels": _pattern_pixels(200, 40, 16, 16),
                    "pts_ticks": 7,
                    "timebase_numerator": 1,
                    "timebase_denominator": 2,
                },
                "f-02": {
                    "pixels": _pattern_pixels(90, 60, 16, 16),
                    "pts_ticks": 9,
                    "timebase_numerator": 1,
                    "timebase_denominator": 2,
                    "physical_time_s": 0.045,
                    "physical_time_reason": "container_pts",
                },
            },
        },
    )


def _frame_identity() -> FrameIdentity:
    return FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-tour-01",
        shot_id="shot-01",
        swing_id="swing-01",
        camera_id="cam-face-on",
        frame_id="frame-001",
        pts_ticks=10,
        timebase_numerator=1,
        timebase_denominator=1,
        physical_time_s=0.01,
        physical_time_reason="shutter",
        frame_sha256="e" * 64,
    )


def _mask_bytes(width: int, height: int, body: bool, club: bool = False) -> bytes:
    return bytes((body if i % 2 == 0 else club) for i in range(width * height))


# ---------------------------------------------------------------------------
# Model cards: metadata without fabricated weight pins
# ---------------------------------------------------------------------------


def test_model_cards_carry_metadata_but_no_fabricated_pins() -> None:
    assert set(PINNED_MODELS) == {"sam-vit-b-golf", "mobilesam-golf"}

    for name, card in PINNED_MODELS.items():
        assert isinstance(card, SegmentationModelCard)
        assert card.model_name == name
        assert card.license == "Apache-2.0"
        # The repository ships no trusted weight pins; operators must pin the
        # checkpoints they provision per deployment (verify stays fail-closed).
        assert card.checkpoint_sha256 is None
        assert card.parameter_count_m > 0
        assert "min_ram_gb" in card.hardware_requirements
        assert len(card.known_limitations) >= 2


def test_fabricated_sha256_constants_are_gone_from_module() -> None:
    """The fabricated pins from the PR body must not exist anywhere in code."""
    import shared.python.shadow_tracker.model_segmentation as module

    assert not hasattr(module, "SAM_VIT_B_GOLF_SHA256")
    assert not hasattr(module, "MOBILESAM_GOLF_SHA256")
    model_text = Path(module.__file__).read_text(encoding="utf-8")
    assert _FABRICATED_SAM_VIT_B_PIN not in model_text
    assert _FABRICATED_MOBILESAM_PIN not in model_text


# ---------------------------------------------------------------------------
# Checkpoint verification: fail-closed with operator-supplied pins
# ---------------------------------------------------------------------------


def test_checkpoint_verification_zero_hidden_downloads(tmp_path: Path) -> None:
    nonexistent = tmp_path / "missing_model.pth"
    with pytest.raises(
        FileNotFoundError, match="Hidden network downloads are disallowed"
    ):
        verify_checkpoint("sam-vit-b-golf", nonexistent, expected_sha256=_OPERATOR_PIN)


def test_checkpoint_verification_rejects_malformed_pins(tmp_path: Path) -> None:
    path = tmp_path / "weights.onnx"
    path.write_bytes(b"weights-payload-v1")

    with pytest.raises(ValueError, match="64 lowercase hex"):
        verify_checkpoint("sam-vit-b-golf", path, expected_sha256=_INVALID_PIN)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        verify_checkpoint("sam-vit-b-golf", path, expected_sha256="a" * 63)
    with pytest.raises(ValueError, match="Unknown model_name"):
        verify_checkpoint("not-registered", path, expected_sha256=_OPERATOR_PIN)


def test_checkpoint_verification_corrupt_or_arbitrary_weights_fail_closed(
    tmp_path: Path,
) -> None:
    corrupt_file = tmp_path / "corrupt_weights.pth"
    corrupt_file.write_bytes(b"arbitrary invalid bytes that do not match pinned SHA")

    with pytest.raises(
        RuntimeError, match="not a valid model checkpoint: hash mismatch"
    ):
        verify_checkpoint("sam-vit-b-golf", corrupt_file, expected_sha256=_OPERATOR_PIN)


def test_checkpoint_verification_passes_with_matching_operator_pin(
    tmp_path: Path,
) -> None:
    content = b"operator-provisioned-checkpoint-payload"
    path = tmp_path / "weights.bin"
    path.write_bytes(content)
    pin = hashlib.sha256(content).hexdigest()

    card = verify_checkpoint("mobilesam-golf", path, expected_sha256=pin)
    assert card.model_name == "mobilesam-golf"
    # The pin is deployment configuration; the card stays pin-free.
    assert card.checkpoint_sha256 is None


# ---------------------------------------------------------------------------
# Adapter: typed unavailability, never synthetic success
# ---------------------------------------------------------------------------


def test_adapter_fails_closed_when_checkpoint_missing(tmp_path: Path) -> None:
    from shared.python.shadow_tracker.model_segmentation import (
        SegmentationUnavailableError,
    )

    with pytest.raises(SegmentationUnavailableError, match="checkpoint"):
        RealSegmentationAdapter(
            "sam-vit-b-golf",
            tmp_path / "missing_weights.onnx",
            expected_sha256=_OPERATOR_PIN,
        )


def test_adapter_never_reports_success_without_checkpoint_run(
    tmp_path: Path,
) -> None:
    """Pinned-but-unloadable weights fail typed; no masks, no success count."""
    from shared.python.shadow_tracker.model_segmentation import (
        SegmentationUnavailableError,
        build_frame_identity,
    )

    content = b"operator-pinned payload that no runtime in this environment runs"
    path = tmp_path / "golf-sam.onnx"
    path.write_bytes(content)
    manual_provider = ManualMaskProvider()

    adapter = RealSegmentationAdapter(
        "sam-vit-b-golf",
        path,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        manual_provider=manual_provider,
    )

    # No torch/onnxruntime is installed in this environment; the adapter must
    # refuse with the typed unavailability instead of synthesizing geometry.
    real_pixels = _pattern_pixels(180, 30, 16, 16)
    identity = build_frame_identity(_pixel_meta(5), real_pixels)
    with pytest.raises(SegmentationUnavailableError):
        adapter.infer_frame(identity, real_pixels)

    with pytest.raises(SegmentationUnavailableError):
        adapter.segment(_segment_request())

    assert manual_provider._masks == {}
    assert adapter.manual_provider is manual_provider


def test_adapter_fails_closed_when_pinned_artifact_changes(
    tmp_path: Path,
) -> None:
    """Weights swapped after pinning must fail before any load, typed."""
    from shared.python.shadow_tracker.model_segmentation import (
        SegmentationUnavailableError,
        build_frame_identity,
    )

    content = b"pinned payload v1"
    path = tmp_path / "golf-sam.onnx"
    path.write_bytes(content)
    adapter = RealSegmentationAdapter(
        "sam-vit-b-golf",
        path,
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )
    pixels = _pattern_pixels(180, 30, 16, 16)
    identity = build_frame_identity(_pixel_meta(5), pixels)

    # Operator artifact is swapped/replaced after the pin was recorded.
    path.write_bytes(b"tampered payload v2")
    with pytest.raises(SegmentationUnavailableError, match="hash mismatch"):
        adapter.infer_frame(identity, pixels)

    # Removing the artifact entirely also fails typed at the call boundary.
    path.unlink()
    with pytest.raises(SegmentationUnavailableError, match="not found"):
        adapter.infer_frame(identity, pixels)


def test_segment_requires_real_pixels_and_frame_metadata(tmp_path: Path) -> None:
    """Dimensions-only synthesis is forbidden: pixels and PTS are mandatory."""
    from shared.python.shadow_tracker.model_segmentation import (
        SegmentationUnavailableError,
    )

    content = b"some operator payload"
    path = tmp_path / "weights.bin"
    path.write_bytes(content)
    adapter = RealSegmentationAdapter(
        "sam-vit-b-golf",
        path,
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )

    with pytest.raises(SegmentationUnavailableError, match="pixels"):
        adapter.segment(
            SegmentationRequest(
                shot_id="shot-01",
                frame_ids=("f-01",),
                options={"asset_id": "asset-01"},
            )
        )

    with pytest.raises(SegmentationUnavailableError, match="f-99"):
        adapter.segment(
            SegmentationRequest(
                shot_id="shot-01",
                frame_ids=("f-99",),
                options={"asset_id": "asset-01", "frames": {}},
            )
        )


# ---------------------------------------------------------------------------
# Frame provenance: pixels hash + honest timing
# ---------------------------------------------------------------------------


def _pixel_meta(pts: int, **extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "frame_id": "frame-001",
        "pts_ticks": pts,
        "timebase_numerator": 1,
        "timebase_denominator": 2,
    }
    meta.update(extra)
    return meta


def test_frame_identity_hashes_decoded_pixels_not_identifiers() -> None:
    from shared.python.shadow_tracker.model_segmentation import (
        build_frame_identity,
    )

    pixels_a = _pattern_pixels(200, 40, 16, 16)
    pixels_b = _pattern_pixels(90, 60, 16, 16)

    identity_a = build_frame_identity(_pixel_meta(5), pixels_a)
    identity_again = build_frame_identity(_pixel_meta(5), pixels_a)
    identity_b = build_frame_identity(_pixel_meta(5), pixels_b)

    # Content identity comes from the decoded pixels, not the frame ID string.
    assert identity_a.frame_sha256 != identity_b.frame_sha256
    assert identity_a.frame_sha256 == identity_again.frame_sha256

    expected = hashlib.sha256(
        b"unspecified:rgb24:" + np.ascontiguousarray(pixels_a).tobytes()
    ).hexdigest()
    assert identity_a.frame_sha256 == expected

    # Same pixels under different decoder provenance bind to different hashes.
    identity_c = build_frame_identity(_pixel_meta(5), pixels_a, decoder_name="opencv")
    assert identity_c.frame_sha256 != identity_a.frame_sha256

    # Caller-supplied timing survives verbatim.
    identity_d = build_frame_identity(
        _pixel_meta(5, physical_time_s=2.5, physical_time_reason="container_pts"),
        pixels_a,
    )
    assert identity_d.physical_time_s == 2.5
    assert identity_d.physical_time_reason == "container_pts"


def test_frame_identity_records_unknown_timing_honestly() -> None:
    from shared.python.shadow_tracker.model_segmentation import (
        build_frame_identity,
    )

    pixels = _pattern_pixels(200, 40, 16, 16)

    identity = build_frame_identity(_pixel_meta(5), pixels)
    assert identity.physical_time_s is None
    assert identity.physical_time_reason
    assert identity.pts_ticks == 5

    # PTS is required: no fabricated tick sequence may replace it.
    meta = _pixel_meta(5)
    del meta["pts_ticks"]
    with pytest.raises(ValueError, match="pts_ticks"):
        build_frame_identity(meta, pixels)

    # A supplied finite physical time requires its own recorded reason.
    with pytest.raises(ValueError, match="physical_time_reason"):
        build_frame_identity(_pixel_meta(5, physical_time_s=1.25), pixels)


# ---------------------------------------------------------------------------
# Revision IDs: content + config sensitive (no collisions for one frame)
# ---------------------------------------------------------------------------


def test_revision_id_diverges_on_content_and_config() -> None:
    from shared.python.shadow_tracker.model_segmentation import derive_revision_id

    frame = _frame_identity()

    id_a = derive_revision_id(
        "sam-vit-b-golf",
        expected_sha256="a" * 64,
        frame=frame,
        body=_mask_bytes(16, 16, True),
        club=_mask_bytes(16, 16, False),
        valid=_mask_bytes(16, 16, True),
        config={"width_px": 16, "height_px": 16, "adverse_conditions": []},
    )
    id_b = derive_revision_id(
        "sam-vit-b-golf",
        expected_sha256="a" * 64,
        frame=frame,
        body=_mask_bytes(16, 16, True),
        club=_mask_bytes(16, 16, False),
        valid=_mask_bytes(16, 16, True),
        config={"width_px": 32, "height_px": 32, "adverse_conditions": ["blur"]},
    )
    identity = derive_revision_id(
        "sam-vit-b-golf",
        expected_sha256="a" * 64,
        frame=frame,
        body=_mask_bytes(16, 16, True),
        club=_mask_bytes(16, 16, False),
        valid=_mask_bytes(16, 16, True),
        config={"width_px": 16, "height_px": 16, "adverse_conditions": []},
    )

    # Same frame/content/config → stable idempotent identity.
    assert id_a == identity
    # Same frame, different config → distinct revision (never colliding).
    assert id_a != id_b
    assert id_a.startswith("rev-sam-vit-b-golf")


def test_manual_mask_provider_accepts_divergent_revisions_for_one_frame() -> None:
    from shared.python.shadow_tracker.model_segmentation import derive_revision_id

    frame = _frame_identity()
    provider = ManualMaskProvider()
    config = {"width_px": 16, "height_px": 16, "adverse_conditions": []}
    ids = []

    for body_true in (True, False):
        revision_id = derive_revision_id(
            "sam-vit-b-golf",
            expected_sha256="c" * 64,
            frame=frame,
            body=_mask_bytes(16, 16, body_true),
            club=_mask_bytes(16, 16, not body_true),
            valid=_mask_bytes(16, 16, True),
            config=config,
        )
        provider.register_mask(
            MaskFrame(
                schema_version=MASK_SCHEMA_VERSION,
                frame=frame,
                width_px=16,
                height_px=16,
                body=_mask_bytes(16, 16, body_true),
                club=_mask_bytes(16, 16, not body_true),
                valid=_mask_bytes(16, 16, True),
                revision_id=revision_id,
                parent_revision_id=None,
                producer_id="model:sam-vit-b-golf",
                correction_note="dev-only checkpoint-validation harness draft",
            )
        )
        ids.append(revision_id)

    assert ids[0] != ids[1]
    history = provider.get_revision_history("frame-001")
    assert len(history) == 2
    assert history[0].revision_id == ids[0]
    assert history[1].revision_id == ids[1]


# ---------------------------------------------------------------------------
# Benchmark: independent gold only; typed blocked results
# ---------------------------------------------------------------------------


class _ThresholdStubAdapter:
    """Software-only stub segmenter for benchmark plumbing tests.

    It is explicitly NOT the real model: it derives masks from pixel
    thresholds so the benchmark metric pipeline can be verified without a
    neural runtime. Its masks are never treated as gold.
    """

    model_card = SimpleNamespace(
        model_name="stub-threshold",
        checkpoint_sha256=None,
    )

    def __init__(self) -> None:
        self.manual_provider = ManualMaskProvider()

    def infer_frame(
        self,
        frame: FrameIdentity,
        pixels: np.ndarray,
        *,
        adverse_conditions: list[str] | None = None,
    ) -> MaskFrame:
        height, width = pixels.shape[:2]
        body = np.zeros((height, width), dtype=np.uint8)
        body[:, : width // 2] = 1  # left half: deliberately wider than gold
        club = np.zeros((height, width), dtype=np.uint8)
        club[height - 1, width - 4 :] = 1  # club corner far from gold band
        valid = np.ones(width * height, dtype=np.uint8)
        revision_id = f"stub-{frame.frame_id}-{int(body.sum())}"
        return MaskFrame(
            schema_version=MASK_SCHEMA_VERSION,
            frame=frame,
            width_px=width,
            height_px=height,
            body=bytes(body.reshape(-1)),
            club=bytes(club.reshape(-1)),
            valid=bytes(valid),
            revision_id=revision_id,
            parent_revision_id=None,
            producer_id="stub:threshold",
            correction_note="software stub for benchmark plumbing tests",
        )


def _stub_frame_source(width: int, height: int) -> Any:
    def source(index: int) -> tuple[np.ndarray, dict[str, Any]]:
        pixels = _pattern_pixels(120 + index * 30, 10 + index, width, height)
        meta = _pixel_meta((index + 1) * 3)
        meta["frame_id"] = f"f-{index + 1:03d}"
        return pixels, meta

    return source


def _write_gold(evidence_dir: Path, label: np.ndarray) -> list[dict[str, Any]]:
    gold_dir = evidence_dir / "gold_masks"
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_path = gold_dir / "f-001.npy"
    np.save(gold_path, label.astype(np.uint8))
    return [
        {
            "frame_id": "f-001",
            "file": str(gold_path.relative_to(evidence_dir)),
            "sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            "pts_ticks": 3,
            "timebase_numerator": 1,
            "timebase_denominator": 1,
        }
    ]


def _write_manifest(evidence_dir: Path, gold_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "schema_version": 1,
        "benchmark": "shadow_tracker_silhouette_segmentation",
        "clips": [
            {
                "clip_id": "clip-01",
                "clip_type": "modern_high_speed",
                "description": "synthetic evidence clip for plumbing tests",
                "width_px": 16,
                "height_px": 16,
                "adverse_conditions": [],
                "gold_masks": gold_rows,
            }
        ],
    }
    (evidence_dir / "clip_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_benchmark_refuses_without_independent_evidence(tmp_path: Path) -> None:
    """No evidence dir ⇒ typed blocked result with zero fabricated numbers."""
    content = b"operator payload, absent evidence"
    path = tmp_path / "golf-sam.onnx"
    path.write_bytes(content)
    adapter = RealSegmentationAdapter(
        "sam-vit-b-golf",
        path,
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )

    report = evaluate_segmentation_benchmark(adapter, evidence_dir=tmp_path)

    assert report["status"] == "blocked"
    assert report["reason_code"] == "missing_evidence"
    assert report["metrics_reported"] is False
    assert report["clips"] == []
    assert "pinned_checkpoint_weights" in report["missing_evidence"]

    body = json.dumps(report, allow_nan=False)
    for forbidden in (
        "body_iou",
        "club_recall",
        "boundary_f1",
        "latency_ms_per_frame",
        "peak_memory_mb",
    ):
        assert forbidden not in body


def test_benchmark_blocked_when_runtime_cannot_run_checkpoint(
    tmp_path: Path,
) -> None:
    content = b"pinned payload, no neural runtime installed"
    path = tmp_path / "golf-sam.onnx"
    path.write_bytes(content)
    manual_provider = ManualMaskProvider()
    adapter = RealSegmentationAdapter(
        "sam-vit-b-golf",
        path,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        manual_provider=manual_provider,
    )

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    label = np.zeros((16, 16), dtype=np.uint8)
    label[0:2, 0:2] = 1  # body label region
    gold_rows = _write_gold(evidence_dir, label)
    _write_manifest(evidence_dir, gold_rows)

    report = evaluate_segmentation_benchmark(
        adapter, evidence_dir=evidence_dir, frame_source=_stub_frame_source(16, 16)
    )

    assert report["status"] == "blocked"
    assert report["reason_code"] == "model_unavailable"
    assert report["metrics_reported"] is False
    assert report["clips"][0]["clip_id"] == "clip-01"
    assert report["clips"][0]["status"] == "model_unavailable"
    assert not set(report["clips"][0]) & {
        "body_iou",
        "club_recall",
        "boundary_f1",
        "latency_ms_per_frame",
        "peak_memory_mb",
    }
    assert "runnable_model_runtime" in report["missing_evidence"]
    # The recorded gold was never replaced by adapter output.
    assert manual_provider._masks == {}


def test_benchmark_scores_against_independent_gold_masks(tmp_path: Path) -> None:
    stub = _ThresholdStubAdapter()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    # Independent gold: known body/club label image, deliberately different
    # from the stub's threshold output so a self-echo could not score 1.0.
    label = np.zeros((16, 16), dtype=np.uint8)
    label[0:2, 0:10] = 1  # body band
    label[10, 0:2] = 2  # club pixels
    gold_rows = _write_gold(evidence_dir, label)
    _write_manifest(evidence_dir, gold_rows)

    report = evaluate_segmentation_benchmark(
        stub, evidence_dir=evidence_dir, frame_source=_stub_frame_source(16, 16)
    )

    assert report["status"] == "ok"
    assert report["metrics_reported"] is True
    metrics = next(c for c in report["clips"] if c["status"] == "evaluated")
    # Metrics came from the recorded gold file, not the adapter under test.
    assert 0.0 <= metrics["body_iou"] < 1.0
    assert 0.0 <= metrics["club_recall"] < 1.0
    assert metrics["correction_effort_edits"] > 0
    assert metrics["latency_ms_per_frame"] >= 0.0
    assert metrics["peak_memory_mb"] >= 0.0


def test_benchmark_gold_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    stub = _ThresholdStubAdapter()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    label = np.zeros((16, 16), dtype=np.uint8)
    label[0:2, 0:10] = 1
    gold_rows = _write_gold(evidence_dir, label)
    # Tamper the manifest with a wrong sha256; the artifact must fail closed.
    gold_rows[0]["sha256"] = "a" * 64
    _write_manifest(evidence_dir, gold_rows)

    report = evaluate_segmentation_benchmark(
        stub, evidence_dir=evidence_dir, frame_source=_stub_frame_source(16, 16)
    )

    assert report["status"] == "blocked"
    assert report["metrics_reported"] is False
    assert report["clips"][0]["status"] == "invalid_evidence"


# ---------------------------------------------------------------------------
# Public façade registration
# ---------------------------------------------------------------------------


def test_facade_exports_real_segmentation_adapter_lazily() -> None:
    import shared.python.shadow_tracker as facade

    assert "RealSegmentationAdapter" in facade.__all__
    assert facade.RealSegmentationAdapter is RealSegmentationAdapter
    assert facade.SegmentationUnavailableError is not None


# ---------------------------------------------------------------------------
# Committed evidence package honesty
# ---------------------------------------------------------------------------


def test_committed_report_is_typed_blocked_without_evidence() -> None:
    model_card_path = EVIDENCE_DIR / "MODEL_CARD.md"
    report_path = EVIDENCE_DIR / "benchmark_report.json"
    readme_path = EVIDENCE_DIR / "README.md"

    assert model_card_path.is_file(), f"Missing {model_card_path}"
    assert report_path.is_file(), f"Missing {report_path}"
    assert readme_path.is_file(), f"Missing {readme_path}"

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["status"] == "blocked"
    assert data["metrics_reported"] is False
    assert "pinned_checkpoint_weights" in data["missing_evidence"]

    serialized = json.dumps(data)
    for forbidden in (
        "body_iou",
        "club_recall",
        "boundary_f1",
        "latency_ms_per_frame",
        "peak_memory_mb",
    ):
        assert forbidden not in serialized

    # No fabricated pins survive in the committed package.
    assert _FABRICATED_SAM_VIT_B_PIN not in model_card_path.read_text(encoding="utf-8")


def test_committed_readbook_reflects_reality() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    assert "blocked" in readme
    assert "gold_masks" in readme
    assert _FABRICATED_SAM_VIT_B_PIN not in readme
