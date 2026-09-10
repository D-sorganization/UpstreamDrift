"""Mermaid C4 Architecture Map Contract Validator.

Validates docs/architecture/C4.md against the maintainable architecture-map contract:
- Non-empty C4Context and C4Container views
- Feature Map table mapping capabilities to components and test evidence
- Architecture Change Log baseline table
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


class ArchitectureMapContractError(ValueError):
    """Raised when an architecture map violates the contract."""


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of architecture map validation."""

    is_valid: bool
    context_views: int
    container_views: int
    feature_map_entries: tuple[dict[str, str], ...]
    changelog_entries: tuple[dict[str, str], ...]


def _check_mermaid_views(text: str) -> tuple[int, int]:
    """Validate and return count of C4Context and C4Container mermaid views."""
    context_matches = re.findall(
        r"```mermaid\s*\n\s*(?:C4Context|graph|flowchart)?[\s\S]*?C4Context[\s\S]*?```",
        text,
        re.IGNORECASE,
    )
    if not context_matches:
        blocks = re.findall(r"```mermaid\s*\n([\s\S]*?)```", text)
        context_matches = [b for b in blocks if "C4Context" in b]

    if not context_matches:
        raise ArchitectureMapContractError(
            "Architecture map missing required 'C4Context' view block in mermaid format."
        )

    blocks = re.findall(r"```mermaid\s*\n([\s\S]*?)```", text)
    container_matches = [b for b in blocks if "C4Container" in b]
    if not container_matches:
        raise ArchitectureMapContractError(
            "Architecture map missing required 'C4Container' view block in mermaid format."
        )

    return len(context_matches), len(container_matches)


def _parse_feature_map(text: str) -> list[dict[str, str]]:
    """Extract and validate Feature Map table entries."""
    if "Feature Map" not in text:
        raise ArchitectureMapContractError(
            "Architecture map missing required 'Feature Map' section."
        )

    feature_entries: list[dict[str, str]] = []
    in_feature_map = False
    for line in text.splitlines():
        if "## Feature Map" in line:
            in_feature_map = True
            continue
        if in_feature_map:
            if line.startswith("## "):
                in_feature_map = False
                continue
            if (
                line.startswith("|")
                and not line.startswith("| ---")
                and not line.startswith("| Capability")
            ):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 4:
                    feature_entries.append(
                        {
                            "capability": parts[0],
                            "component": parts[1],
                            "interface": parts[2],
                            "evidence": parts[3],
                        }
                    )

    if not feature_entries:
        raise ArchitectureMapContractError(
            "Feature Map table is empty or missing required columns (Capability, Component, Interface, Evidence)."
        )
    return feature_entries


def _parse_change_log(text: str) -> list[dict[str, str]]:
    """Extract and validate Architecture Change Log table entries."""
    if "Change Log" not in text:
        raise ArchitectureMapContractError(
            "Architecture map missing required 'Architecture Change Log' section."
        )

    changelog_entries: list[dict[str, str]] = []
    in_changelog = False
    for line in text.splitlines():
        if "Change Log" in line and line.startswith("#"):
            in_changelog = True
            continue
        if in_changelog:
            if line.startswith("## ") and "Change Log" not in line:
                in_changelog = False
                continue
            if (
                line.startswith("|")
                and not line.startswith("| ---")
                and not line.startswith("| Date")
            ):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 3:
                    changelog_entries.append(
                        {
                            "date": parts[0],
                            "pr_or_issue": parts[1],
                            "description": parts[2],
                        }
                    )

    if not changelog_entries:
        raise ArchitectureMapContractError(
            "Architecture Change Log table is empty or missing required columns."
        )
    return changelog_entries


def validate_architecture_map(path: Path) -> ValidationResult:
    """Validate a docs/architecture/C4.md file against the contract."""
    if not path.is_file():
        raise ArchitectureMapContractError(
            f"Architecture map file does not exist: {path}"
        )

    text = path.read_text(encoding="utf-8")
    num_context, num_container = _check_mermaid_views(text)
    feature_entries = _parse_feature_map(text)
    changelog_entries = _parse_change_log(text)

    return ValidationResult(
        is_valid=True,
        context_views=num_context,
        container_views=num_container,
        feature_map_entries=tuple(feature_entries),
        changelog_entries=tuple(changelog_entries),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for architecture map validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("docs/architecture/C4.md"),
        help="Path to C4.md architecture map file",
    )
    args = parser.parse_args(argv)

    try:
        res = validate_architecture_map(args.path)
        print(
            f"PASS: {args.path} is valid ({res.context_views} context, "
            f"{res.container_views} container, {len(res.feature_map_entries)} features, "
            f"{len(res.changelog_entries)} changelog rows)"
        )
        return 0
    except ArchitectureMapContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
