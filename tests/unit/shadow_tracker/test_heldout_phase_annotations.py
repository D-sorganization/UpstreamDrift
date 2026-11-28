"""Unit tests for the frozen independent held-out phase annotation set (#15837 WP-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.shadow_tracker import heldout_phase_annotations as ho
from src.shared.python.shadow_tracker import markerless_scorecard as ms

pytestmark = pytest.mark.unit

PINNED = {
    "epic_issue": 15814,
    "scorecard_issue": 15837,
    "video_id": "9i7ok4rVkVM",
    "source_url": "https://www.youtube.com/watch?v=9i7ok4rVkVM",
    "source_sha256": "6" * 64,
    "frame_map_sha256": "7" * 64,
    "dimensions_px": [1280, 720],
    "timebase": [1, 30000],
    "candidate_window_source_indices": [1080, 1500],
    "timebase_status": "unverified",
    "physical_time_verified": False,
    "qualification_floor": "image_observations_only",
}


def frame(
    index: int = 1100,
    phase: str = "top",
    x: float = 640.0,
    y: float = 300.0,
    angle: float = 12.5,
    digest: str = "a" * 64,
) -> ho.HeldoutFrameAnnotation:
    """Build one valid held-out annotation record."""
    return ho.HeldoutFrameAnnotation(
        frame_index=index,
        phase=phase,
        frame_sha256=digest,
        keypoints_px={"left_wrist": (x, y), "club_head": (x + 40.0, y + 120.0)},
        shaft_angle_deg=angle,
    )


def annotation_set(
    frames: tuple[ho.HeldoutFrameAnnotation, ...],
    *,
    status: str = "reviewed",
    floor_status: str = "measured",
    pinned: dict[str, object] | None = None,
) -> ho.HeldoutPhaseAnnotations:
    """Build a held-out annotation envelope for tests."""
    measured = floor_status == "measured"
    return ho.HeldoutPhaseAnnotations(
        schema_version=ho.HELDOUT_SCHEMA_VERSION,
        annotation_set_id="tiger-2000-heldout-test",
        status=status,  # type: ignore[arg-type]
        annotator="test-annotator",
        pinned_reference=dict(pinned or PINNED),
        required_phases=ms.load_scorecard_gates().required_phases,
        frames=frames,
        independence=ho.IndependenceDeclaration("human_reviewed", False, False),
        noise_floor=ho.NoiseFloor(
            floor_status,  # type: ignore[arg-type]
            3.2 if measured else None,
            1.1 if measured else None,
            8 if measured else 0,
        ),
        review=ho.AnnotationReview(
            "independent_human_annotation_on_pinned_decoded_frames",
            True,
            "test-annotator",
            "2026-11-28",
        ),
    )


def test_shipped_heldout_envelope_is_frozen_and_pin_aligned() -> None:
    """The committed artifact loads, is versioned, and shares the WP-1 pin set."""
    loaded = ho.load_heldout_annotations()
    gates = ms.load_scorecard_gates()
    assert loaded.schema_version == ho.HELDOUT_SCHEMA_VERSION
    assert loaded.required_phases == gates.required_phases
    assert loaded.status == "pending"
    assert loaded.frames == ()
    ho.assert_pins_match(gates.pinned_reference, loaded.pinned_reference)


def test_pending_heldout_set_is_not_scoreable() -> None:
    """A pending annotation set reports blockers instead of scoring."""
    blockers = ho.load_heldout_annotations().scoreability_blockers()
    assert "heldout_annotation_pass_pending" in blockers
    assert "annotator_noise_floor_unmeasured" in blockers


def test_reviewed_measured_independent_set_is_scoreable() -> None:
    """A reviewed set with a measured floor on an independent pass has no blocker."""
    reviewed = annotation_set((frame(), frame(1101, "impact", 610.0, 400.0, -55.0)))
    assert reviewed.scoreability_blockers() == ()


def test_frame_digest_and_window_are_verified_against_pinned_map() -> None:
    """Frame digests must appear in the pinned frame map inside the window."""
    frame_map = {1100: "a" * 64, 1101: "b" * 64}
    reviewed = annotation_set((frame(),))
    reviewed.verify_frame_map(frame_map)
    with pytest.raises(ValueError, match="frame_sha256"):
        annotation_set((frame(digest="f" * 64),)).verify_frame_map(frame_map)
    with pytest.raises(ValueError, match="candidate window"):
        annotation_set((frame(index=2000),)).verify_frame_map(frame_map)


def test_out_of_frame_keypoints_are_rejected() -> None:
    """Image coordinates must lie inside the pinned frame dimensions."""
    with pytest.raises(ValueError, match="pinned frame"):
        annotation_set((frame(x=1400.0),))
    with pytest.raises(ValueError, match="pinned frame"):
        annotation_set((frame(y=-3.0),))


def test_measured_noise_floor_requires_at_least_two_repeat_frames() -> None:
    """A single repeat cannot establish a noise floor."""
    with pytest.raises(ValueError, match="two repeat frames"):
        ho.NoiseFloor(
            status="measured",
            keypoint_rmse_px=6.0,
            shaft_angle_deg=4.0,
            repeat_pair_count=1,
        )


def test_content_sha256_is_deterministic_and_tamper_evident(tmp_path: Path) -> None:
    """Serialization is deterministic and a pinned hash mismatch fails closed."""
    reviewed = annotation_set((frame(),))
    again = annotation_set((frame(),))
    assert reviewed.content_sha256 == again.content_sha256
    payload = reviewed.to_json()
    path = tmp_path / "heldout.json"
    path.write_text(payload, encoding="utf-8")
    assert ho.load_heldout_annotations(path).content_sha256 == reviewed.content_sha256
    tampered = payload.replace('"annotator": "test-annotator"', '"annotator": "x"')
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        ho.load_heldout_annotations(path)


def test_pin_drift_against_gate_config_is_rejected() -> None:
    """A held-out set pinned to a different rendition cannot be loaded."""
    drifted = annotation_set((frame(),), pinned={**PINNED, "source_sha256": "9" * 64})
    with pytest.raises(ValueError, match="source_sha256"):
        ho.assert_pins_match(
            ms.load_scorecard_gates().pinned_reference, drifted.pinned_reference
        )
