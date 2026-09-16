"""Tests for full-body showpiece design decisions record (HO-7 #10161)."""

from __future__ import annotations

from pathlib import Path
import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
DOC_PATH = (
    REPO_ROOT / "docs" / "development" / "full_body_models" / "DESIGN_DECISIONS.md"
)


def test_design_decisions_file_exists() -> None:
    """Verify that DESIGN_DECISIONS.md exists."""
    assert DOC_PATH.is_file(), f"Expected {DOC_PATH} to exist."


def test_design_decisions_parser_and_order() -> None:
    """Verify all 14 decisions exist in order, each with what, why, receipt, rejected."""
    from src.shared.python.motion_matching.pipeline.design_decisions import (
        parse_design_decisions,
        validate_design_decisions,
    )

    content = DOC_PATH.read_text(encoding="utf-8")
    decisions = parse_design_decisions(content)
    assert len(decisions) == 14, f"Expected 14 decisions, got {len(decisions)}"

    # Validate decision contents and verify all referenced receipt links exist on disk
    validate_design_decisions(decisions, repo_root=REPO_ROOT)


def test_design_decisions_rejects_bad_input() -> None:
    """Verify DbC validation error handling on invalid inputs."""
    from src.shared.python.motion_matching.pipeline.design_decisions import (
        parse_design_decisions,
        validate_design_decisions,
    )

    with pytest.raises(TypeError, match="must be a str"):
        parse_design_decisions(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot be empty"):
        parse_design_decisions("   ")

    with pytest.raises(ValueError, match="repo_root must be a valid directory"):
        validate_design_decisions([], repo_root=Path("non_existent_path_xyz"))

    with pytest.raises(ValueError, match="decisions sequence cannot be empty"):
        validate_design_decisions([], repo_root=REPO_ROOT)


def test_design_decisions_rejects_broken_link(tmp_path: Path) -> None:
    """Verify that a broken link target raises FileNotFoundError."""
    from src.shared.python.motion_matching.pipeline.design_decisions import (
        DesignDecision,
        validate_design_decisions,
    )

    # Create dummy decisions list where one decision has a broken link
    fake_decisions = [
        DesignDecision(
            number=i + 1,
            title=title,
            what="What",
            why="Why",
            receipt="[`receipt.json`](evidence/non_existent_file.json)"
            if i == 0
            else "[`r`](evidence/ground_support/anthro_driver/receipt.json)",
            rejected="Rejected",
            review_sections="[REVIEW](evidence/anthropometry/REVIEW.md#1)",
            raw_markdown="",
        )
        for i, title in enumerate(
            [
                "Anthropometric Geometry From de Leva",
                "Arms Forward at Zero Pose",
                "Scapula Rz",
                "One Static-Trial Round",
                "Marker-Driven Elbow Pits",
                "Anatomical Wrist Axes and Neutral-Grip Turn",
                "Fitted Hand-to-Club Rotation (`GRIP_ROTATION_DEG`)",
                "Human Ranges in the Matching Only, Wrists Bounded by Default",
                "Clubs From `club_models`",
                "Compliant 50 kN/m Sole",
                "12 Hz Tracked Reference",
                "Reference Zero-Moment-Point Diagnostic",
                "Rejected: Grip-Roll Scan, Closure Fit From the Address, Cart-Table Filter, Fixed-Point and Iterative-Learning Shooting Fits",
                "MJX Differentiable Optimisation (Windowed)",
            ]
        )
    ]

    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        validate_design_decisions(fake_decisions, repo_root=REPO_ROOT)
