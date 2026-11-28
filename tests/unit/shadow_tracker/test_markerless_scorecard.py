"""Unit tests for Tiger 2000 markerless scorecard and null baseline (#15837)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.shared.python.shadow_tracker import heldout_phase_annotations as ho
from src.shared.python.shadow_tracker import markerless_scorecard as ms

pytestmark = pytest.mark.unit


def _heldout(
    anns: list[ms.FramePoseRecord],
    *,
    status: str = "reviewed",
    floor: str = "measured",
    pinned: dict[str, object] | None = None,
) -> ho.HeldoutPhaseAnnotations:
    """Wrap synthetic annotations in a pinned held-out envelope."""
    gates = ms.load_scorecard_gates()
    return ho.HeldoutPhaseAnnotations(
        schema_version=ho.HELDOUT_SCHEMA_VERSION,
        annotation_set_id="tiger-2000-heldout-fixture",
        status=status,  # type: ignore[arg-type]
        annotator="fixture-annotator",
        pinned_reference=dict(pinned or gates.pinned_reference),
        required_phases=gates.required_phases,
        frames=tuple(
            ho.HeldoutFrameAnnotation(
                frame_index=a.frame_index,
                phase=a.phase,
                frame_sha256=f"{a.frame_index:064x}",
                keypoints_px=a.keypoints_px,
                shaft_angle_deg=a.shaft_angle_deg,
            )
            for a in anns
        ),
        independence=ho.IndependenceDeclaration(
            annotator_kind="human_reviewed",
            derived_from_null_template=False,
            derived_from_candidate_tracker=False,
        ),
        noise_floor=ho.NoiseFloor(
            status=floor,  # type: ignore[arg-type]
            keypoint_rmse_px=3.0 if floor == "measured" else None,
            shaft_angle_deg=1.0 if floor == "measured" else None,
            repeat_pair_count=6 if floor == "measured" else 0,
        ),
        review=ho.AnnotationReview(
            method="independent_human_annotation_on_pinned_decoded_frames",
            second_pass_required=True,
            reviewer="fixture-reviewer",
            completed_at="2026-11-28",
        ),
    )


def _sample(
    phases: tuple[str, ...] = ("address", "top", "downswing", "impact", "finish"),
    per_phase: int = 2,
    kp_e: float = 0.0,
    ang_e: float = 0.0,
) -> tuple[list[ms.PhaseAnnotation], list[ms.CandidateFramePrediction]]:
    """Build synthetic annotations and predictions offset from the null template."""
    tpl = ms.build_phase_aligned_null_template()
    anns: list[ms.PhaseAnnotation] = []
    preds: list[ms.CandidateFramePrediction] = []
    idx = 1080
    for phase in phases:
        pose = tpl.phase_poses[phase]
        kp = {k: (v[0] + 10.0, v[1]) for k, v in pose.keypoints_px.items()}
        pkp = {k: (v[0] + kp_e, v[1]) for k, v in kp.items()}
        ang = pose.shaft_angle_deg + 5.0
        for _ in range(per_phase):
            anns.append(ms.PhaseAnnotation(idx, phase, kp, ang))
            preds.append(ms.CandidateFramePrediction(idx, phase, pkp, ang + ang_e))
            idx += 10
    return anns, preds


def _assert_unverified(sc: ms.MarkerlessScorecard) -> None:
    """Assert that a scorecard retains unverified physical timebase flags."""
    assert sc.timebase == "unverified"
    assert sc.physical_time_verified is False
    assert sc.qualification == "image_observations_only"


def test_load_scorecard_gates_and_pinned_metadata() -> None:
    """Gate config loads Tiger 2000 thresholds and #15814/9i7ok4rVkVM metadata."""
    g = ms.load_scorecard_gates()
    assert g.schema_version == ms.SCORECARD_GATES_SCHEMA_VERSION
    assert g.required_phases == ("address", "top", "downswing", "impact", "finish")
    assert g.noise_floor_multiplier == pytest.approx(1.5)
    ref = g.pinned_reference
    assert ref["epic_issue"] == 15814 and ref["video_id"] == "9i7ok4rVkVM"
    assert ref["timebase_status"] == "unverified"
    assert ref["physical_time_verified"] is False
    assert ref["qualification_floor"] == "image_observations_only"


def test_known_offsets_fail_1_5x_noise_floor_ceiling() -> None:
    """Offsets exceeding 1.5x annotator noise floor fail closed."""
    for kp_e, ang_e, frag in ((13.0, 1.0, "keypoint"), (1.0, 7.0, "shaft_angle")):
        anns, preds = _sample(kp_e=kp_e, ang_e=ang_e)
        sc = ms.evaluate_markerless_scorecard(
            "c-n", anns, preds, heldout=_heldout(anns)
        )
        assert sc.status == "below_gate" and sc.match_classification == "below_gate"
        assert any(frag in r for r in sc.failure_reasons)
        assert sc.failing_phases == ("address", "top", "downswing", "impact", "finish")


def test_missing_and_low_coverage_phases_fail_closed() -> None:
    """Missing impact and sub-threshold finish coverage fail closed and name both."""
    anns, preds = _sample(("address", "top", "downswing", "finish"), per_phase=4)
    sc = ms.evaluate_markerless_scorecard(
        "c-cov", anns, preds[:-2], heldout=_heldout(anns)
    )
    assert sc.status == "below_gate"
    assert {"impact", "finish"}.issubset(sc.failing_phases)
    assert all(any(p in r for r in sc.failure_reasons) for p in ("impact", "finish"))


def test_candidate_must_beat_phase_aligned_null_template_by_margin() -> None:
    """Candidate within noise ceiling still fails without beating null margin."""
    anns, preds = _sample(kp_e=9.0, ang_e=4.2)
    g = ms.load_scorecard_gates()
    tpl = ms.build_phase_aligned_null_template(g)
    sc = ms.evaluate_markerless_scorecard(
        "c-m", anns, preds, gates=g, null_template=tpl, heldout=_heldout(anns)
    )
    assert sc.status == "below_gate"
    assert any("null_baseline" in r for r in sc.failure_reasons)


def test_passing_candidate_emits_2d_or_3d_plausible_and_unverified_timebase() -> None:
    """Passing candidate emits 2D match or 3D plausible with unverified timebase."""
    anns, preds = _sample(kp_e=1.5, ang_e=0.5)
    held = _heldout(anns)
    sc2 = ms.evaluate_markerless_scorecard("c-2d", anns, preds, heldout=held)
    assert sc2.schema_version == ms.SCORECARD_SCHEMA_VERSION and sc2.status == "passed"
    assert sc2.match_classification == "kinematic_2d_match"
    _assert_unverified(sc2)
    part = ms.ModelSpaceChecks(True, True, True, True, False)
    sc_p = ms.evaluate_markerless_scorecard(
        "c-p", anns, preds, model_space_checks=part, heldout=held
    )
    assert sc_p.match_classification == "kinematic_2d_match"
    full = ms.ModelSpaceChecks(True, True, True, True, True)
    sc3 = ms.evaluate_markerless_scorecard(
        "c-3d", anns, preds, model_space_checks=full, heldout=held
    )
    assert sc3.status == "passed"
    assert sc3.match_classification == "kinematic_3d_plausible"
    _assert_unverified(sc3)


def test_scorecard_serialization_and_deterministic_sha256_hash(tmp_path: Path) -> None:
    """Scorecard JSON serialization and SHA-256 content hash are deterministic."""
    anns, preds = _sample(kp_e=1.0, ang_e=0.5)
    held = _heldout(anns)
    sc_a = ms.evaluate_markerless_scorecard("c-h", anns, preds, heldout=held)
    sc_b = ms.evaluate_markerless_scorecard("c-h", anns, preds, heldout=held)
    assert len(sc_a.content_sha256) == 64 and sc_a.content_sha256 == sc_b.content_sha256
    assert sc_a.to_json() == sc_b.to_json()
    pl = sc_a.to_payload_dict()
    raw = json.dumps(pl, sort_keys=True, allow_nan=False, separators=(",", ":"))
    assert sc_a.content_sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    out = tmp_path / "scorecard.json"
    out.write_text(sc_a.to_json(), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["content_sha256"] == sc_a.content_sha256
    assert loaded["timebase"] == "unverified"
    assert loaded["physical_time_verified"] is False
    assert loaded["qualification"] == "image_observations_only"


def test_unpinned_evaluation_cannot_report_passed() -> None:
    """Without a frozen held-out set the gate refuses any pass verdict."""
    anns, preds = _sample(kp_e=1.0, ang_e=0.5)
    sc = ms.evaluate_markerless_scorecard("c-unpinned", anns, preds)
    assert sc.status == "below_gate"
    assert sc.heldout_set_sha256 == ""
    assert "heldout_annotations_missing" in sc.failure_reasons


def test_pending_or_assumed_heldout_set_blocks_pass_verdict() -> None:
    """An unreviewed pass or an assumed noise floor cannot yield a pass."""
    anns, preds = _sample(kp_e=1.0, ang_e=0.5)
    sc = ms.evaluate_markerless_scorecard(
        "c-pending",
        anns,
        preds,
        heldout=_heldout(anns, status="pending", floor="assumed"),
    )
    assert sc.status == "below_gate"
    assert {
        "heldout_annotation_pass_pending",
        "annotator_noise_floor_unmeasured",
    }.issubset(sc.failure_reasons)


def test_heldout_pin_must_match_gate_config() -> None:
    """A held-out set pinned to another rendition is rejected before scoring."""
    anns, preds = _sample()
    drifted = _heldout(
        anns,
        pinned={
            **ms.load_scorecard_gates().pinned_reference,
            "source_sha256": "9" * 64,
        },
    )
    with pytest.raises(ValueError, match="pinned"):
        ms.evaluate_markerless_scorecard("c-drift", anns, preds, heldout=drifted)
