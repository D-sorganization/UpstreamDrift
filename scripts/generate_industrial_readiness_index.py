"""Generate the readiness indexes from the machine-readable ledgers.

Epic #9539 is the repository-level execution index for the 2026-09-04
industrial readiness review; epic #9546 is the same review's Impact Zone and
Impact Explorer product program. Each committed markdown index must always
match its ledger; the freshness test in
tests/config/industrial_readiness/test_readiness_index_freshness.py
regenerates every doc and compares it byte-for-byte.

Usage:
    python -m scripts.generate_industrial_readiness_index [--check]

``--check`` exits non-zero (without writing) when any committed doc is stale.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.industrial_readiness_loader import (  # noqa: E402
    IMPACT_ZONE_LEDGER_PATH,
    LEDGER_PATH,
    AcceptanceCriterion,
    IndustrialReadinessLedger,
    ReadinessItem,
)

_DOCS_DIR = REPO_ROOT / "docs" / "operations"


@dataclass(frozen=True)
class IndexSpec:
    """One ledger -> one generated index document.

    Attributes:
        ledger_path: Machine-readable ledger the doc is generated from
        index_path: Committed markdown document
        title: Top-level heading
        scope: Prose lines describing what the epic's queue covers
    """

    ledger_path: Path
    index_path: Path
    title: str
    scope: tuple[str, ...]


INDUSTRIAL_INDEX = IndexSpec(
    ledger_path=LEDGER_PATH,
    index_path=_DOCS_DIR / "industrial-readiness-index.md",
    title="Industrial Readiness Index",
    scope=(
        "industrial readiness review. The priority children hold the code changes;",
        "this record says which of them landed, what proves it, and which remain",
        "open. It is a software-correctness record only — scientific and human",
    ),
)

IMPACT_ZONE_INDEX = IndexSpec(
    ledger_path=IMPACT_ZONE_LEDGER_PATH,
    index_path=_DOCS_DIR / "impact-zone-readiness-index.md",
    title="Impact Zone Readiness Index",
    scope=(
        "review's Impact Zone and Impact Explorer product program. Tools owns the",
        "shared impact/flight runtime, so provider fixes land there first and this",
        "record says which reviewed pins UpstreamDrift has consumed, what proves",
        "it, and which product slices remain open. It is a software-correctness",
        "record only — scientific and human",
    ),
)

INDEX_SPECS: tuple[IndexSpec, ...] = (INDUSTRIAL_INDEX, IMPACT_ZONE_INDEX)

#: Kept for callers that predate the second ledger.
INDEX_PATH = INDUSTRIAL_INDEX.index_path

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


def render_index(
    ledger: IndustrialReadinessLedger, spec: IndexSpec = INDUSTRIAL_INDEX
) -> str:
    """Render a readiness ledger as a markdown index.

    Args:
        ledger: Loaded readiness ledger
        spec: Which index document the ledger renders into

    Returns:
        Full markdown document content (deterministic for a given ledger)
    """
    open_items = ledger.open_items
    merged_items = ledger.merged_items
    ledger_rel = spec.ledger_path.relative_to(REPO_ROOT).as_posix()
    lines = [
        f"# {spec.title}",
        "",
        "<!-- AUTO-GENERATED — do not edit by hand. -->",
        "<!-- Regenerate with: python3 -m scripts.generate_industrial_readiness_index -->",
        "",
        f"Generated from [`{ledger_rel}`](../../{ledger_rel})"
        f" (ledger v{ledger.version}).",
        "",
        f"Execution index for epic {_issue_link(ledger.epic)}, the 2026-09-04",
        *spec.scope,
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

    stale = False
    for spec in INDEX_SPECS:
        ledger = IndustrialReadinessLedger.load(spec.ledger_path)
        rendered = render_index(ledger, spec)
        index_path = spec.index_path

        if args.check:
            current = (
                index_path.read_text(encoding="utf-8") if index_path.exists() else ""
            )
            if current != rendered:
                stale = True
                print(  # noqa: T201 - CLI tool output
                    f"STALE: {index_path} does not match {spec.ledger_path.name}. "
                    "Run: python3 -m scripts.generate_industrial_readiness_index"
                )
            else:
                print(f"OK: {index_path} is up to date.")  # noqa: T201 - CLI output
            continue

        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"Wrote {index_path}")  # noqa: T201 - CLI tool output
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
