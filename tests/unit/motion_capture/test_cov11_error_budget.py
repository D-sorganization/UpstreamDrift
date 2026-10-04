"""Unit tests for COV-11 error-budget receipt and public summary (#11279).

Strict TDD tests asserting:
- Deterministic budget generation from synthetic COV-7/COV-9 summaries (byte-identical rerun).
- Cell with n_swings = 0 is impossible; missing data appears in not_measured.
- L1 cell can never be labelled trustworthy for a per-frame quantity (contract rule).
- Frozen-rule guard: changing a threshold without bumping schema version fails.
- Public summary generator privacy invariants (no frame indices, no absolute paths, no private file names).
"""

from __future__ import annotations

import re
from typing import Any
import pytest

pytestmark = pytest.mark.unit

from src.motion_capture.reference.error_budget import (
    ERROR_BUDGET_SCHEMA_VERSION,
    ErrorBudget,
    ErrorBudgetCell,
    GuidanceItem,
    GuidanceReport,
    GuidanceRule,
    NotMeasuredRecord,
    build_error_budget,
    derive_guidance,
    generate_public_summary,
    get_default_guidance_rules,
)


def _make_synthetic_cov7_summary(
    *,
    backend: str = "mediapipe",
    video_swing_id: str = "cov-01-s1",
    view: str = "dtl",
    grade: str = "A",
    level: str = "L2",
    joint_name: str = "lead_wrist",
    p50: float = 12.5,
    p95: float = 24.0,
    worst: float = 38.0,
    camera_spread: float = 3.2,
    resolvable: bool = True,
    receipt_hash: str = "a" * 64,
) -> dict[str, Any]:
    """Create a synthetic COV-7 2D comparison summary payload."""
    return {
        "summary_type": "cov-7-2d",
        "backend": backend,
        "video_swing_id": video_swing_id,
        "view": view,
        "grade": grade,
        "level": level,
        "joint_or_landmark": joint_name,
        "phase_bin": "all",
        "metric": "residual_norm",
        "unit": "norm",
        "n_swings": 1,
        "n_frames": 120,
        "p50": p50,
        "p95": p95,
        "worst": worst,
        "camera_uncertainty_spread": camera_spread,
        "resolvable": resolvable,
        "receipt_hash": receipt_hash,
    }


def _make_synthetic_cov9_summary(
    *,
    backend: str = "necromatcher_anchored",
    video_swing_id: str = "cov-01-s1",
    view: str = "dtl",
    grade: str = "A",
    level: str = "L3",
    joint_name: str = "hand_path_depth",
    p50: float = 18.0,
    p95: float = 32.0,
    worst: float = 45.0,
    receipt_hash: str = "b" * 64,
) -> dict[str, Any]:
    """Create a synthetic COV-9 3D comparison summary payload."""
    return {
        "summary_type": "cov-9-3d",
        "backend": backend,
        "video_swing_id": video_swing_id,
        "view": view,
        "grade": grade,
        "level": level,
        "joint_or_landmark": joint_name,
        "phase_bin": "all",
        "metric": "depth_error",
        "unit": "mm",
        "n_swings": 1,
        "n_frames": 120,
        "p50": p50,
        "p95": p95,
        "worst": worst,
        "camera_uncertainty_spread": None,
        "resolvable": True,
        "receipt_hash": receipt_hash,
    }


def test_synthetic_error_budget_building_is_deterministic() -> None:
    """Building the budget from synthetic COV-7/COV-9 summaries is deterministic (byte-identical output on rerun)."""
    summaries = [
        _make_synthetic_cov7_summary(joint_name="lead_wrist", p95=22.0),
        _make_synthetic_cov7_summary(joint_name="trail_wrist", p95=28.0),
        _make_synthetic_cov9_summary(joint_name="hand_path_depth", p95=35.0),
        _make_synthetic_cov9_summary(joint_name="pelvis_rotation_top", p95=4.5),
    ]

    budget1 = build_error_budget(summaries)
    budget2 = build_error_budget(summaries)

    json1 = budget1.to_json()
    json2 = budget2.to_json()

    assert json1 == json2, "Budget JSON serialization must be byte-identical on rerun"
    assert budget1.receipt_digest == budget2.receipt_digest
    assert len(budget1.cells) == 4
    assert all(c.n_swings > 0 for c in budget1.cells)


def test_synthetic_cell_with_zero_swings_is_impossible_and_appears_in_not_measured() -> (
    None
):
    """A cell with n swings = 0 is impossible; missing data appears as an absent cell plus an explicit not_measured list."""
    with pytest.raises(ValueError, match="n_swings must be positive"):
        ErrorBudgetCell(
            source="mediapipe",
            joint_or_landmark="lead_wrist",
            phase_bin="all",
            view="dtl",
            grade="A",
            comparison_level="L2",
            metric="residual_norm",
            unit="norm",
            n_swings=0,
            n_frames=0,
            p50=0.0,
            p95=0.0,
            worst=0.0,
            camera_uncertainty_spread=None,
            resolvable=False,
            source_receipt_hashes=("a" * 64,),
        )

    # Ingesting summaries with missing / zero-observation quantities
    summaries = [
        _make_synthetic_cov7_summary(joint_name="lead_wrist", p95=20.0),
        {
            "summary_type": "cov-7-2d",
            "backend": "mediapipe",
            "video_swing_id": "cov-02-s1",
            "view": "dtl",
            "grade": "C",
            "level": "L2",
            "joint_or_landmark": "club_head",
            "phase_bin": "downswing",
            "metric": "residual_norm",
            "unit": "norm",
            "n_swings": 0,
            "n_frames": 0,
            "reason": "club_head occluded throughout downswing in Grade C clip",
            "receipt_hash": "c" * 64,
        },
    ]

    budget = build_error_budget(summaries)

    # Assert no cell has n_swings == 0
    for cell in budget.cells:
        assert cell.n_swings > 0
        assert cell.joint_or_landmark != "club_head"

    # Missing data must be explicitly captured in not_measured
    assert len(budget.not_measured) == 1
    unmeasured = budget.not_measured[0]
    assert isinstance(unmeasured, NotMeasuredRecord)
    assert unmeasured.joint_or_landmark == "club_head"
    assert "club_head occluded" in unmeasured.reason


def test_synthetic_l1_cell_cannot_be_labelled_trustworthy_for_per_frame_quantity() -> (
    None
):
    """An L1 cell can never be labelled trustworthy for a per-frame quantity (contract rule raises ValueError)."""
    # Create an L1 cell with an artificially tiny error for a per-frame trajectory metric
    l1_cell = ErrorBudgetCell(
        source="mediapipe",
        joint_or_landmark="hand_path_depth",
        phase_bin="all",
        view="dtl",
        grade="A",
        comparison_level="L1",  # L1 = envelope only, never per-frame
        metric="depth_error",
        unit="mm",
        n_swings=5,
        n_frames=600,
        p50=2.0,
        p95=5.0,  # Below threshold X (e.g. 40.0 mm), would look trustworthy if evaluated naively
        worst=8.0,
        camera_uncertainty_spread=1.0,
        resolvable=True,
        source_receipt_hashes=("d" * 64,),
    )

    budget = ErrorBudget(
        schema_version=ERROR_BUDGET_SCHEMA_VERSION,
        cells=(l1_cell,),
        not_measured=(),
        created_utc="2026-10-04T00:00:00Z",
        receipt_digest="e" * 64,
    )

    # Contract rule: L1 comparison level cannot certify per-frame quantities as trustworthy
    with pytest.raises(
        ValueError,
        match="L1 cell cannot be labelled trustworthy for per-frame quantity",
    ):
        derive_guidance(budget)


def test_synthetic_guidance_threshold_change_without_schema_bump_fails_frozen_rule_guard() -> (
    None
):
    """Changing a guidance threshold without bumping the schema version -> test failure (frozen-rule guard)."""
    default_rules = get_default_guidance_rules()
    assert "hand_path_depth" in default_rules

    # Tampering with a threshold under the current schema version
    tampered_rules = dict(default_rules)
    tampered_rules["hand_path_depth"] = GuidanceRule(
        quantity_name="hand_path_depth",
        target_metric="depth_error",
        max_trustworthy_p95=999.0,  # Loosened threshold!
        max_indicative_p95=1200.0,
        unit="mm",
        requires_level="L3",
        is_per_frame=True,
    )

    budget = build_error_budget([_make_synthetic_cov9_summary()])

    # Frozen-rule guard: modified thresholds with current schema version must raise ValueError
    with pytest.raises(ValueError, match="frozen in schema version"):
        derive_guidance(
            budget,
            rules=tampered_rules,
            schema_version=ERROR_BUDGET_SCHEMA_VERSION,
        )

    # When schema version is legitimately bumped, modified rules are accepted
    bumped_report = derive_guidance(
        budget,
        rules=tampered_rules,
        schema_version="error-budget/2.0.0",
    )
    assert bumped_report.schema_version == "error-budget/2.0.0"


def test_synthetic_public_summary_generator_privacy_invariants() -> None:
    """Public summary generator: output contains no cov-NN frame indices, no absolute paths, and no private store file names."""
    summary_input = _make_synthetic_cov9_summary(
        p95=25.0,
        receipt_hash="f" * 64,
    )
    # Add mock private traces in source summaries metadata
    summary_input["metadata"] = {
        "private_path": "C:\\Users\\diete\\AppData\\Local\\capture-O-video\\originals\\VID_20261002.MP4",
        "frame_index": "cov-01_frame_000123",
        "c3d_file": "private_capture_O.c3d",
    }

    budget = build_error_budget([summary_input])
    guidance = derive_guidance(budget)

    # Owner-approved summary generation
    public_summary = generate_public_summary(budget, guidance, owner_approved=True)

    # 1. Privacy invariant: No frame indices (e.g. cov-01_frame_123 or cov-01:123)
    frame_index_pattern = re.compile(r"cov-\d+.*frame|frame[ _-]?\d+", re.IGNORECASE)
    assert not frame_index_pattern.search(public_summary), (
        "Public summary must not leak frame indices"
    )

    # 2. Privacy invariant: No absolute paths (Windows or POSIX)
    path_pattern = re.compile(
        r"[A-Za-z]:[\\/]|/(?:home|Users|var|tmp|AppData)/", re.IGNORECASE
    )
    assert not path_pattern.search(public_summary), (
        "Public summary must not leak absolute filesystem paths"
    )

    # 3. Privacy invariant: No private file names or extensions
    private_file_pattern = re.compile(
        r"\bVID_\d+\b|\.mp4\b|\.c3d\b|\.mov\b|originals[\\/]", re.IGNORECASE
    )
    assert not private_file_pattern.search(public_summary), (
        "Public summary must not leak private store file names"
    )

    # 4. Mandatory content: Neutral IDs and aggregate body-height-normalized numbers
    assert "capture-O" in public_summary or "subject-O" in public_summary
    assert "Error Budget" in public_summary
    assert "Guidance" in public_summary

    # 5. Fail-closed without owner approval
    with pytest.raises(ValueError, match="owner approval"):
        generate_public_summary(budget, guidance, owner_approved=False)


def test_synthetic_unresolvable_due_to_camera_spread_marked_not_recoverable() -> None:
    """When camera uncertainty spread makes metric unresolvable, guidance marks it not recoverable."""
    summary = _make_synthetic_cov9_summary(
        joint_name="hand_path_depth",
        p95=10.0,  # Below threshold X (40.0)
    )
    summary["resolvable"] = False
    summary["camera_uncertainty_spread"] = 25.0

    budget = build_error_budget([summary])
    guidance = derive_guidance(budget)

    assert len(guidance.items) == 1
    item = guidance.items[0]
    assert item.classification == "not recoverable from single view"
    assert "not resolvable" in item.rationale


def test_synthetic_error_budget_cell_validation() -> None:
    """ErrorBudgetCell enforces finite numbers, valid SHA-256 hashes, and valid frames."""
    base_kwargs = {
        "source": "hmr2",
        "joint_or_landmark": "pelvis",
        "phase_bin": "top",
        "view": "dtl",
        "grade": "A",
        "comparison_level": "L3",
        "metric": "mpjpe",
        "unit": "mm",
        "n_swings": 3,
        "n_frames": 100,
        "p50": 15.0,
        "p95": 25.0,
        "worst": 35.0,
        "source_receipt_hashes": ("e" * 64,),
    }

    # Invalid hash (not 64 hex chars)
    with pytest.raises(ValueError, match="SHA-256 digest"):
        ErrorBudgetCell(**{**base_kwargs, "source_receipt_hashes": ("invalid_hash",)})

    # Non-finite number
    with pytest.raises(ValueError, match="finite number"):
        ErrorBudgetCell(**{**base_kwargs, "p95": float("nan")})

    # Negative camera spread
    with pytest.raises(ValueError, match="non-negative"):
        ErrorBudgetCell(**{**base_kwargs, "camera_uncertainty_spread": -5.0})


def test_synthetic_guidance_classifications_match_profile_tiers() -> None:
    """Verifies trustworthy, indicative, and not recoverable guidance classifications."""
    summaries = [
        # Trustworthy: p95 = 20.0 < 40.0
        _make_synthetic_cov9_summary(joint_name="hand_path_depth", p95=20.0),
        # Indicative: 40.0 <= p95 = 55.0 < 80.0
        _make_synthetic_cov9_summary(joint_name="hand_path_depth", p95=55.0),
        # Not recoverable: p95 = 95.0 >= 80.0
        _make_synthetic_cov9_summary(joint_name="hand_path_depth", p95=95.0),
    ]

    budget = build_error_budget(summaries)
    guidance = derive_guidance(budget)

    classifications = [item.classification for item in guidance.items]
    assert classifications == [
        "trustworthy at p95 < 40.0",
        "indicative",
        "not recoverable from single view",
    ]
