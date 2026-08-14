"""Build and validate the proximal-to-distal scientific claim audit.

The inventory is deliberately broader than a hand-selected claim list. It
captures every narrative paragraph from the Quarto master and its included
chapters, preserving the canonical source path and line range. Human review
then decides whether a candidate is material and whether it is supported,
contradicted, inconclusive, or untested.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "proximal-distal-claim-audit-v1"
INCLUDE_PATTERN = re.compile(r"^\s*\{\{<\s*include\s+([^ >]+)\s*>\}\}\s*$")
DISPLAY_MATH_FENCE_PATTERN = re.compile(r"^\$\$(?:\s*\{[^}]*\})?\s*$")
CITATION_PATTERN = re.compile(r"(?<!\w)@([A-Za-z0-9_][A-Za-z0-9_:.\-]*)")
CROSS_REFERENCE_PREFIXES = ("sec-", "fig-", "eq-", "tbl-", "lst-")
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
HIGH_RISK_LANGUAGE_PATTERN = re.compile(
    r"\b(?:caus(?:e|al|ally|ation)|mechanism|optimal|univers(?:al|ally)|"
    r"validat(?:e|es|ed|ion)|prove|proves|best|dominant|necessary|sufficient)\b",
    re.IGNORECASE,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _bibliographic_citations(text: str) -> list[str]:
    """Return bibliography keys while excluding Quarto cross-references."""
    keys: set[str] = set()
    for raw_key in CITATION_PATTERN.findall(text):
        key = raw_key.rstrip(".:")
        if key and not key.startswith(CROSS_REFERENCE_PREFIXES):
            keys.add(key)
    return sorted(keys)


def _triage(text: str, citations: list[str]) -> tuple[int, list[str]]:
    """Rank review urgency without making a scientific adjudication."""
    flags: list[str] = []
    score = 0
    if NUMERIC_PATTERN.search(text):
        flags.append("numeric")
        score += 2
    if ASSERTION_PATTERN.search(text):
        flags.append("assertive_language")
        score += 2
    if citations:
        flags.append("external_citation")
        score += 1
    if HIGH_RISK_LANGUAGE_PATTERN.search(text):
        flags.append("causal_or_generalizing_language")
        score += 3
    return score, flags


def _source_digest(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        canonical_text = (
            path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _front_matter_abstract(path: Path) -> list[dict[str, Any]]:
    """Extract Quarto's literal-block abstract while ignoring other metadata."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    abstract_lines: list[str] = []
    start_line = 0
    end_line = 0
    in_abstract = False
    for index, line in enumerate(lines[1:], start=2):
        if not in_abstract:
            if re.fullmatch(r"abstract:\s*[|>]\s*", line):
                in_abstract = True
            elif line.strip() == "---":
                break
            continue
        if line and not line[0].isspace():
            break
        if not line.strip():
            continue
        if start_line == 0:
            start_line = index
        end_line = index
        abstract_lines.append(line.strip())
    if not abstract_lines:
        return []
    return [
        {
            "text": _normalise_text(" ".join(abstract_lines)),
            "line_start": start_line,
            "line_end": end_line,
        }
    ]


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
        if DISPLAY_MATH_FENCE_PATTERN.fullmatch(stripped):
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
    if skip_front_matter:
        return _front_matter_abstract(path) + paragraphs
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
        text_occurrences: Counter[str] = Counter()
        for paragraph in _paragraphs(source, skip_front_matter=source == master):
            text = paragraph["text"]
            relative = source.relative_to(root).as_posix()
            text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            occurrence = text_occurrences[text_digest]
            text_occurrences[text_digest] += 1
            identity = f"{relative}\0{text_digest}\0{occurrence}"
            candidate_id = (
                "PD-CAND-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            )
            if candidate_id in seen_ids:
                raise ValueError(f"Duplicate candidate identifier: {candidate_id}")
            seen_ids.add(candidate_id)
            citations = _bibliographic_citations(text)
            priority_score, triage_flags = _triage(text, citations)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source_path": relative,
                    "line_start": paragraph["line_start"],
                    "line_end": paragraph["line_end"],
                    "text": text,
                    "text_sha256": text_digest,
                    "citation_keys": citations,
                    "has_numeric_content": bool(NUMERIC_PATTERN.search(text)),
                    "has_assertive_language": bool(ASSERTION_PATTERN.search(text)),
                    "priority_score": priority_score,
                    "triage_flags": triage_flags,
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


def _validate_source_location(location: str, claim_id: str, root: Path) -> None:
    """Resolve a repository-relative ``path:line`` locator or fail closed."""
    if not isinstance(location, str) or not location.strip():
        raise ValueError(f"{claim_id}: source_locations entries must be non-empty")
    path_text, separator, line_text = location.rpartition(":")
    if not separator or not path_text or not line_text.isdigit():
        raise ValueError(
            f"{claim_id}: source location must use path:line: {location!r}"
        )
    source_path = Path(path_text)
    if source_path.is_absolute():
        raise ValueError(
            f"{claim_id}: source location must be repository-relative: {location!r}"
        )
    resolved = (root / source_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"{claim_id}: source location escapes repository root: {location!r}"
        )
    if not resolved.is_file():
        raise ValueError(f"{claim_id}: missing source location file: {location!r}")
    line_number = int(line_text)
    line_count = len(resolved.read_text(encoding="utf-8").splitlines())
    if line_number < 1 or line_number > line_count:
        raise ValueError(
            f"{claim_id}: source location line is out of range: {location!r}"
        )


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
        for location in record["source_locations"]:
            _validate_source_location(location, claim_id, root)
        for artifact in record["evidence_artifacts"]:
            if not isinstance(artifact, str) or not artifact.strip():
                raise ValueError(
                    f"{claim_id}: evidence_artifacts entries must be non-empty strings"
                )
            if artifact.startswith(("https://", "http://")):
                continue
            artifact_path_text = artifact.partition("#")[0]
            artifact_path = Path(artifact_path_text)
            if artifact_path.is_absolute():
                raise ValueError(
                    f"{claim_id}: local evidence artifact must be repository-relative: "
                    f"{artifact!r}"
                )
            resolved_artifact = (root / artifact_path).resolve()
            if not resolved_artifact.is_relative_to(root):
                raise ValueError(
                    f"{claim_id}: local evidence artifact escapes repository root: "
                    f"{artifact!r}"
                )
            if not resolved_artifact.is_file():
                raise ValueError(
                    f"{claim_id}: missing local evidence artifact: {artifact!r}"
                )

    paper = registry.get("paper", {})
    source = root / paper.get("source", "")
    inventory = build_candidate_inventory(source, repository_root=root)
    if paper.get("source_digest") != inventory["source_digest"]:
        raise ValueError(
            "Paper source_digest is stale; rebuild the candidate inventory"
        )

    bibliography = source.parent / "references.bib"
    if bibliography.is_file():
        bibliography_keys = set(
            re.findall(
                r"@\w+\s*\{\s*([^,\s]+)",
                bibliography.read_text(encoding="utf-8"),
            )
        )
        cited_keys = {
            key
            for candidate in inventory["candidates"]
            for key in candidate["citation_keys"]
        }
        missing_citations = sorted(cited_keys - bibliography_keys)
        if missing_citations:
            raise ValueError(
                f"Paper citations missing from references.bib: {missing_citations}"
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
    open_release_states = {"pending", "in_progress"}
    for item in release_inventory:
        release_key = item.get("release_claim_key", "<missing release_claim_key>")
        for field in ("published_status", "audit_state"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"{release_key}: {field} must be a non-empty string")
    open_release_keys = sorted(
        item["release_claim_key"]
        for item in release_inventory
        if item["audit_state"] in open_release_states
    )
    release_review_completion = "complete" if not open_release_keys else "in_progress"
    declared_release_completion = registry.get("audit_scope", {}).get(
        "release_review_completion_status"
    )
    if (
        declared_release_completion is not None
        and declared_release_completion != release_review_completion
    ):
        raise ValueError(
            "Declared release review completion does not match release claim states"
        )

    completion = registry.get("audit_scope", {}).get("completion_status")
    candidate_ids = {item["candidate_id"] for item in inventory["candidates"]}
    claim_id_set = set(claim_ids)
    reviews = registry.get("candidate_reviews")
    if not isinstance(reviews, list):
        raise ValueError("candidate_reviews must be a list")
    review_ids = [review.get("candidate_id") for review in reviews]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Duplicate candidate_id in candidate_reviews")
    unknown_reviews = sorted(set(review_ids) - candidate_ids)
    if unknown_reviews:
        raise ValueError(f"candidate_reviews contains unknown IDs: {unknown_reviews}")

    allowed_dispositions = {
        "material_claims_mapped",
        "non_material",
        "editorial_or_navigation",
        "requires_split",
    }
    review_map: dict[str, set[str]] = {}
    for review in reviews:
        candidate_id = review.get("candidate_id", "<missing candidate_id>")
        disposition = review.get("disposition")
        if disposition not in allowed_dispositions:
            raise ValueError(f"{candidate_id}: unsupported disposition {disposition!r}")
        for field in ("rationale", "reviewer", "last_verified_on"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                raise ValueError(f"{candidate_id}: {field} must be a non-empty string")
        mapped_claims = review.get("claim_ids")
        if not isinstance(mapped_claims, list):
            raise ValueError(f"{candidate_id}: claim_ids must be a list")
        mapped_set = set(mapped_claims)
        if len(mapped_claims) != len(mapped_set):
            raise ValueError(f"{candidate_id}: duplicate claim_ids")
        unknown_claims = sorted(mapped_set - claim_id_set)
        if unknown_claims:
            raise ValueError(
                f"{candidate_id}: unknown mapped claim_ids {unknown_claims}"
            )
        if disposition == "material_claims_mapped" and not mapped_set:
            raise ValueError(f"{candidate_id}: material review must map a claim")
        if disposition in {"non_material", "editorial_or_navigation"} and mapped_set:
            raise ValueError(
                f"{candidate_id}: non-material review cannot map scientific claims"
            )
        review_map[candidate_id] = mapped_set

    for claim in claims:
        claim_id = claim["claim_id"]
        for candidate_id in claim.get("candidate_ids", []):
            if claim_id not in review_map.get(candidate_id, set()):
                raise ValueError(
                    f"{claim_id}: {candidate_id} lacks a reciprocal candidate review"
                )

    unadjudicated = candidate_ids - set(review_ids)
    if completion == "complete" and unadjudicated:
        raise ValueError(
            "Audit cannot be complete while paper candidates remain unadjudicated"
        )
    if completion == "complete" and any(
        review["disposition"] == "requires_split" for review in reviews
    ):
        raise ValueError("Audit cannot be complete while candidates require splitting")
    return {
        "completion_status": completion,
        "candidate_count": len(candidate_ids),
        "unadjudicated_candidate_count": len(unadjudicated),
        "registered_claim_count": len(claims),
        "reviewed_candidate_count": len(review_ids),
        "release_claim_count": len(release_keys),
        "release_review_completion_status": release_review_completion,
        "open_release_claim_count": len(open_release_keys),
        "open_release_claim_keys": open_release_keys,
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
