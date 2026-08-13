"""Build and validate the proximal-to-distal scientific claim audit.

The inventory is deliberately broader than a hand-selected claim list. It
captures every narrative paragraph from the Quarto master and its included
chapters, preserving the canonical source path and line range. Human review
then decides whether a candidate is material and whether it is supported,
contradicted, inconclusive, or untested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "proximal-distal-claim-audit-v1"
INCLUDE_PATTERN = re.compile(r"^\s*\{\{<\s*include\s+([^ >]+)\s*>\}\}\s*$")
CITATION_PATTERN = re.compile(r"(?<!\w)@([A-Za-z0-9_:.\-]+)")
NUMERIC_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?\d+(?:[,.]\d+)*(?:e[-+]?\d+)?", re.IGNORECASE
)
ASSERTION_PATTERN = re.compile(
    r"\b(?:show|shows|shown|demonstrate|demonstrates|support|supports|"
    r"reject|rejects|increase|increases|decrease|decreases|produce|produces|"
    r"predict|predicts|establish|establishes|remain|remains|result|results|"
    r"evidence|correlate|correlates|falsif(?:y|ies|ied))\b",
    re.IGNORECASE,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _source_digest(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _paragraphs(path: Path, *, skip_front_matter: bool) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    paragraphs: list[dict[str, Any]] = []
    buffer: list[str] = []
    start_line = 0
    in_code = False
    in_math = False
    in_front_matter = skip_front_matter and bool(lines) and lines[0].strip() == "---"

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line
        if not buffer:
            return
        text = _normalise_text(" ".join(buffer))
        buffer = []
        if len(text) < 20:
            return
        paragraphs.append(
            {"text": text, "line_start": start_line, "line_end": end_line}
        )

    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if in_front_matter:
            if index > 1 and stripped == "---":
                in_front_matter = False
            continue
        if stripped.startswith("```"):
            flush(index - 1)
            in_code = not in_code
            continue
        if stripped == "$$":
            flush(index - 1)
            in_math = not in_math
            continue
        if in_code or in_math:
            continue
        if not stripped:
            flush(index - 1)
            continue
        if stripped.startswith(("#", "{{<", ":::")) or re.fullmatch(
            r"\|?[\s:|-]+\|?", stripped
        ):
            flush(index - 1)
            continue
        if not buffer:
            start_line = index
        buffer.append(stripped)
    flush(len(lines))
    return paragraphs


def _ordered_sources(master: Path) -> list[Path]:
    sources = [master]
    for line in master.read_text(encoding="utf-8").splitlines():
        match = INCLUDE_PATTERN.match(line)
        if match:
            source = (master.parent / match.group(1)).resolve()
            if not source.is_file():
                raise ValueError(f"Included source does not exist: {source}")
            sources.append(source)
    return sources


def build_candidate_inventory(
    master: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Return a deterministic, source-located inventory of narrative candidates."""
    master = master.resolve()
    root = (repository_root or _repository_root()).resolve()
    if not master.is_file():
        raise ValueError(f"Paper source does not exist: {master}")
    try:
        master.relative_to(root)
    except ValueError as exc:
        raise ValueError("Paper source must be inside repository_root") from exc

    sources = _ordered_sources(master)
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in sources:
        for paragraph in _paragraphs(source, skip_front_matter=source == master):
            text = paragraph["text"]
            relative = source.relative_to(root).as_posix()
            identity = f"{relative}:{paragraph['line_start']}:{text}"
            candidate_id = (
                "PD-CAND-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            )
            if candidate_id in seen_ids:
                raise ValueError(f"Duplicate candidate identifier: {candidate_id}")
            seen_ids.add(candidate_id)
            citations = sorted(set(CITATION_PATTERN.findall(text)))
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source_path": relative,
                    "line_start": paragraph["line_start"],
                    "line_end": paragraph["line_end"],
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "citation_keys": citations,
                    "has_numeric_content": bool(NUMERIC_PATTERN.search(text)),
                    "has_assertive_language": bool(ASSERTION_PATTERN.search(text)),
                    "review_state": "unadjudicated",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_source": master.relative_to(root).as_posix(),
        "source_digest": _source_digest(sources, root),
        "source_count": len(sources),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _require_list(record: dict[str, Any], field: str, claim_id: str) -> None:
    value = record.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{claim_id}: {field} must be a non-empty list")


def validate_registry(
    registry_path: Path,
    *,
    repository_root: Path | None = None,
    check_release_manifest: bool = True,
) -> dict[str, Any]:
    """Validate claim contracts and reconcile the public release claims."""
    root = (repository_root or _repository_root()).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported claim-audit schema: {registry.get('schema_version')!r}"
        )

    claims = registry.get("claims")
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    claim_ids = [record.get("claim_id") for record in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Duplicate claim_id in registry")

    required_text = (
        "statement",
        "classification",
        "published_status",
        "audit_status",
        "model_domain",
        "uncertainty_boundary",
        "falsifier",
        "adjudication",
        "reviewer",
        "last_verified_on",
    )
    required_lists = (
        "source_locations",
        "evidence_artifacts",
        "competing_explanations",
        "negative_controls",
    )
    for record in claims:
        claim_id = record.get("claim_id", "<missing claim_id>")
        for field in required_text:
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{claim_id}: {field} must be a non-empty string")
        for field in required_lists:
            _require_list(record, field, claim_id)

    paper = registry.get("paper", {})
    source = root / paper.get("source", "")
    inventory = build_candidate_inventory(source, repository_root=root)
    if paper.get("source_digest") != inventory["source_digest"]:
        raise ValueError(
            "Paper source_digest is stale; rebuild the candidate inventory"
        )

    release_inventory = registry.get("release_claim_inventory", [])
    release_keys = [item.get("release_claim_key") for item in release_inventory]
    if len(release_keys) != len(set(release_keys)):
        raise ValueError("Duplicate release_claim_key in registry")
    if check_release_manifest:
        manifest_path = root / (
            "docs/research/proximal_distal_energy_transfer/release_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = set(manifest.get("claims", {}))
        actual = set(release_keys)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"Release claim inventory mismatch; missing={missing}, extra={extra}"
            )

    completion = registry.get("audit_scope", {}).get("completion_status")
    adjudicated = {
        candidate for claim in claims for candidate in claim.get("candidate_ids", [])
    }
    candidate_ids = {item["candidate_id"] for item in inventory["candidates"]}
    unadjudicated = candidate_ids - adjudicated
    if completion == "complete" and unadjudicated:
        raise ValueError(
            "Audit cannot be complete while paper candidates remain unadjudicated"
        )
    return {
        "completion_status": completion,
        "candidate_count": len(candidate_ids),
        "unadjudicated_candidate_count": len(unadjudicated),
        "registered_claim_count": len(claims),
        "release_claim_count": len(release_keys),
        "source_digest": inventory["source_digest"],
    }


def _default_paths(root: Path) -> tuple[Path, Path, Path]:
    directory = root / "docs/research/proximal_distal_energy_transfer"
    return (
        directory / "proximal_distal_energy_transfer.qmd",
        directory / "data/claim_candidate_inventory.json",
        directory / "data/claim_audit_registry.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory", help="Write claim candidates")
    inventory_parser.add_argument("--output", type=Path)
    subparsers.add_parser("validate", help="Validate the claim registry")
    args = parser.parse_args()

    root = _repository_root()
    master, default_output, registry = _default_paths(root)
    if args.command == "inventory":
        output = args.output or default_output
        result = build_candidate_inventory(master, repository_root=root)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"output": output.as_posix(), "candidates": result["candidate_count"]}
            )
        )
    else:
        print(json.dumps(validate_registry(registry, repository_root=root), indent=2))


if __name__ == "__main__":
    main()
