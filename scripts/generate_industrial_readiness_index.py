"""Generate the industrial readiness index from industrial_readiness.json.

Epic #9539 is the repository-level execution index for the 2026-09-04
industrial readiness review. The committed markdown index must always match
the machine-readable ledger; the freshness test in
tests/config/industrial_readiness/test_readiness_index_freshness.py
regenerates the doc and compares it byte-for-byte.

Usage:
    python -m scripts.generate_industrial_readiness_index [--check]

``--check`` exits non-zero (without writing) when the committed doc is stale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.industrial_readiness_loader import (  # noqa: E402
    AcceptanceCriterion,
    IndustrialReadinessLedger,
    ReadinessItem,
)

INDEX_PATH = REPO_ROOT / "docs" / "operations" / "industrial-readiness-index.md"

_ISSUE_URL = "https://github.com/D-sorganization/UpstreamDrift/issues"

_QUEUE_BADGES = {
    "merged": "✅ merged",
    "open": "🔴 open",
}

_ACCEPTANCE_BADGES = {
    "met": "✅ Met",
    "partial": "🟡 Partial",
    "unmet": "🔴 Unmet",
}

_RELEASE_BADGES = {
    "blocked": "🔴 blocked",
    "ready": "✅ ready",
}


def _issue_link(issue: int) -> str:
    """Render a GitHub issue number as a markdown link."""
    return f"[#{issue}]({_ISSUE_URL}/{issue})"


def _issue_links(issues: tuple[int, ...]) -> str:
    """Render a tuple of issue numbers, or an em dash when empty."""
    return ", ".join(_issue_link(issue) for issue in issues) if issues else "—"


def _paths(values: tuple[str, ...]) -> str:
    """Render repo-relative paths as inline code, or an em dash when empty."""
    return " · ".join(f"`{value}`" for value in values) if values else "—"


def _render_item(item: ReadinessItem) -> list[str]:
    """Render one queue entry as a markdown subsection."""
    lines = [
        f"### {item.key} — {item.title}",
        "",
        f"{_QUEUE_BADGES[item.status]} · {item.priority} · {_issue_link(item.issue)}",
        "",
    ]
    if item.is_open:
        lines += [
            f"- **Owner:** {item.owner}",
            f"- **Depends on:** {_issue_links(item.depends_on)}",
            f"- **Source:** {_paths(item.implementation)}",
            f"- **Existing tests:** {_paths(item.tests)}",
            "",
            f"**Narrow PR plan.** {item.plan}",
            "",
        ]
        return lines

    shas = " · ".join(f"`{sha}`" for sha in item.merge_shas)
    lines += [
        f"- **Merge SHA:** {shas}",
        f"- **Implementation:** {_paths(item.implementation)}",
        f"- **Tests:** {_paths(item.tests)}",
        "",
        f"**Acceptance evidence.** {item.acceptance_evidence}",
        "",
    ]
    return lines


def _render_acceptance(criterion: AcceptanceCriterion) -> list[str]:
    """Render one acceptance criterion as a markdown subsection."""
    return [
        f"### {_ACCEPTANCE_BADGES[criterion.status]} — {criterion.criterion}",
        "",
        f"- **Blockers:** {_issue_links(criterion.blockers)}",
        "",
        criterion.evidence,
        "",
    ]


def render_index(ledger: IndustrialReadinessLedger) -> str:
    """Render the readiness ledger as a markdown index.

    Args:
        ledger: Loaded industrial-readiness ledger

    Returns:
        Full markdown document content (deterministic for a given ledger)
    """
    open_items = ledger.open_items
    merged_items = ledger.merged_items
    lines = [
        "# Industrial Readiness Index",
        "",
        "<!-- AUTO-GENERATED — do not edit by hand. -->",
        "<!-- Regenerate with: python3 -m scripts.generate_industrial_readiness_index -->",
        "",
        "Generated from [`src/config/industrial_readiness.json`]"
        "(../../src/config/industrial_readiness.json)"
        f" (ledger v{ledger.version}).",
        "",
        f"Execution index for epic {_issue_link(ledger.epic)}, the 2026-09-04",
        "industrial readiness review. The priority children hold the code changes;",
        "this record says which of them landed, what proves it, and which remain",
        "open. It is a software-correctness record only — scientific and human",
        "qualification are recorded separately, through the design-manual",
        "governance pathway.",
        "",
        f"- **Release status:** {_RELEASE_BADGES[ledger.release_status]}",
        f"- **Reconciled against:** `{ledger.reconciled_against}`"
        f" on {ledger.reconciled_on}",
        f"- **Audit snapshot:** `{ledger.audit_snapshot}`"
        " (context, not current branch identity)",
        f"- **Queue:** {len(merged_items)} merged · {len(open_items)} open",
        "",
        "## Priority Implementation Queue",
        "",
    ]
    # A markdown table would be reformatted by prettier (which pads cells to a
    # common display width) and then fail the byte-for-byte freshness gate on
    # the very next run. A list renders identically under both.
    for item in ledger.queue:
        summary = (
            f"- **{item.key}** · {_issue_link(item.issue)} · {item.priority}"
            f" · {_QUEUE_BADGES[item.status]}"
        )
        if item.is_open:
            summary += (
                f" · owner {item.owner}, depends on {_issue_links(item.depends_on)}"
            )
        lines.append(summary)
    lines.append("")

    for item in ledger.queue:
        lines += _render_item(item)

    lines += [
        "## Acceptance Criteria",
        "",
    ]
    for criterion in ledger.acceptance:
        lines += _render_acceptance(criterion)

    lines += [
        "## Keeping This Record Honest",
        "",
        "`src/config/industrial_readiness_loader.py` refuses a ledger that claims",
        "more than the tree supports. A merged entry must carry a 40-character",
        "merge SHA, at least one test path, and user-visible acceptance evidence;",
        "an open entry must carry no merge SHA, an owner, a plan, and a dependency",
        "when no owner is named. Every referenced path must exist. Every open",
        "issue must appear in an acceptance blocker list, and the release status",
        "cannot read `ready` while any entry is open or any criterion is unmet.",
        "",
        "An issue closure, a mock-only success, a changed golden file or a raised",
        "tolerance is not evidence of correctness and must not be recorded here.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success / fresh, 1 when ``--check`` finds a stale doc.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed index matches the ledger; do not write.",
    )
    args = parser.parse_args(argv)

    ledger = IndustrialReadinessLedger.load()
    rendered = render_index(ledger)

    if args.check:
        current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""
        if current != rendered:
            print(  # noqa: T201 - CLI tool output
                f"STALE: {INDEX_PATH} does not match industrial_readiness.json. "
                "Run: python3 -m scripts.generate_industrial_readiness_index"
            )
            return 1
        print(f"OK: {INDEX_PATH} is up to date.")  # noqa: T201 - CLI tool output
        return 0

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {INDEX_PATH}")  # noqa: T201 - CLI tool output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
