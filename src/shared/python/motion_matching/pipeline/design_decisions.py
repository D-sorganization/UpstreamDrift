"""Design decision record parser and validator for the full-body showpiece (HO-7 #10161)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DECISION_HEADING_PATTERN = re.compile(r"^##\s+(\d+)\.\s+(.+)$", re.MULTILINE)

REQUIRED_FIELDS = ("what", "why", "receipt", "rejected")

EXPECTED_DECISION_TITLES = [
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


@dataclass(frozen=True)
class DesignDecision:
    """A single consolidated design decision entry."""

    number: int
    title: str
    what: str
    why: str
    receipt: str
    rejected: str
    review_sections: str
    raw_markdown: str


def parse_design_decisions(markdown_text: str) -> list[DesignDecision]:
    """Parse DESIGN_DECISIONS.md content into structured DesignDecision records.

    DbC Preconditions:
    - markdown_text must be non-empty string.

    DbC Postconditions:
    - Returns a list of parsed DesignDecision instances.
    """
    if not isinstance(markdown_text, str):
        raise TypeError(f"markdown_text must be a str, got {type(markdown_text)}")
    if not markdown_text.strip():
        raise ValueError("markdown_text cannot be empty")

    matches = list(DECISION_HEADING_PATTERN.finditer(markdown_text))
    if not matches:
        return []

    decisions: list[DesignDecision] = []
    for i, match in enumerate(matches):
        number = int(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        block = markdown_text[start:end].strip()

        # Extract subsections / bullet fields
        what = _extract_field(block, "What")
        why = _extract_field(block, "Why")
        receipt = _extract_field(block, "Evidence Receipt")
        rejected = _extract_field(block, "What Was Tried and Rejected")
        review_sections = _extract_field(block, "REVIEW.md Sections")

        decisions.append(
            DesignDecision(
                number=number,
                title=title,
                what=what,
                why=why,
                receipt=receipt,
                rejected=rejected,
                review_sections=review_sections,
                raw_markdown=block,
            )
        )

    return decisions


def _extract_field(block: str, field_name: str) -> str:
    """Extract field text under ### Field Name or - **Field Name**:"""
    # Try ### Field Name
    pattern_h3 = re.compile(
        rf"^###\s+{re.escape(field_name)}\s*\n(.*?)(?=^###|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match_h3 = pattern_h3.search(block)
    if match_h3:
        return match_h3.group(1).strip()

    # Try - **Field Name**:
    pattern_bullet = re.compile(
        rf"^\s*-\s*\*\*{re.escape(field_name)}\*\*:\s*(.*?)(?=^\s*-\s*\*\*|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match_bullet = pattern_bullet.search(block)
    if match_bullet:
        return match_bullet.group(1).strip()

    return ""


def validate_design_decisions(
    decisions: Sequence[DesignDecision], repo_root: Path
) -> None:
    """Validate completeness, order, and physical existence of receipt links.

    DbC Preconditions:
    - decisions must be non-empty sequence of DesignDecision.
    - repo_root must be a valid directory Path.
    """
    if not isinstance(repo_root, Path) or not repo_root.is_dir():
        raise ValueError(f"repo_root must be a valid directory path, got {repo_root}")
    if not decisions:
        raise ValueError("decisions sequence cannot be empty")

    if len(decisions) != len(EXPECTED_DECISION_TITLES):
        raise ValueError(
            f"Expected {len(EXPECTED_DECISION_TITLES)} decisions, found {len(decisions)}"
        )

    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    for idx, (dec, expected_title) in enumerate(
        zip(decisions, EXPECTED_DECISION_TITLES, strict=True)
    ):
        if dec.number != idx + 1:
            raise ValueError(
                f"Decision {dec.title} expected number {idx + 1}, got {dec.number}"
            )
        if dec.title != expected_title:
            raise ValueError(
                f"Decision {idx + 1} title mismatch: expected '{expected_title}', got '{dec.title}'"
            )

        if not dec.what:
            raise ValueError(f"Decision {dec.number} missing 'What'")
        if not dec.why:
            raise ValueError(f"Decision {dec.number} missing 'Why'")
        if not dec.receipt:
            raise ValueError(f"Decision {dec.number} missing 'Evidence Receipt'")
        if not dec.rejected:
            raise ValueError(
                f"Decision {dec.number} missing 'What Was Tried and Rejected'"
            )

        # Verify all markdown links in the receipt and review_sections fields exist
        full_field_text = f"{dec.receipt}\n{dec.review_sections}"
        for match in link_pattern.finditer(full_field_text):
            link_target = match.group(2)
            if link_target.startswith(("http://", "https://")):
                continue

            # Strip anchor if present
            file_part = link_target.split("#")[0]
            if not file_part:
                continue

            # Resolve relative to docs/development/full_body_models/
            doc_dir = repo_root / "docs" / "development" / "full_body_models"
            target_path = (doc_dir / file_part).resolve()
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Decision {dec.number} link target '{link_target}' resolves to "
                    f"'{target_path}', which does not exist on disk."
                )
