"""Build reviewer-facing claim outcome and evidence-qualification tables."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
from typing import Any

SUMMARY_SCHEMA_VERSION = "proximal-distal-claim-adjudication-summary-v1"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _evidence_qualifications(claim: dict[str, Any]) -> list[str]:
    """Expose review work still open without changing scientific outcomes."""
    audit = claim["audit_status"].lower()
    published = claim["published_status"].lower()
    classification = claim["classification"].lower()
    artifacts = claim["evidence_artifacts"]
    qualifications: set[str] = set()
    if any(item.startswith(("http://", "https://")) for item in artifacts):
        qualifications.add("external_source_registered")
    if "full_text" in audit or "full article" in audit:
        qualifications.add("original_full_text_checked")
    elif "abstract" in audit:
        qualifications.add("abstract_or_record_only")
    if any(
        item.endswith((".py", ".json", ".npz", ".csv"))
        for item in artifacts
        if not item.startswith(("http://", "https://"))
    ):
        qualifications.add("project_executable_or_data")
    if any(
        token in audit
        for token in (
            "reanalysis_open",
            "reimplementation_open",
            "replication_open",
            "external_validation_open",
            "systematic_review_open",
            "matrix_incomplete",
        )
    ):
        qualifications.add("independent_followup_open")
    if any(
        token in f"{published} {classification} {claim['statement'].lower()}"
        for token in ("human validation", "human data", "participant-held-out")
    ):
        qualifications.add("governed_human_validation_open")
    if any(token in classification for token in ("hypothesis", "prospective")):
        qualifications.add("hypothesis_or_prospective_protocol")
    if not qualifications:
        qualifications.add("declared_internal_review")
    return sorted(qualifications)


def build_summary(registry_path: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    claims = registry["claims"]
    outcome_counts = Counter(claim["adjudication_outcome"] for claim in claims)
    qualification_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for claim in claims:
        qualifications = _evidence_qualifications(claim)
        qualification_counts.update(qualifications)
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "adjudication_outcome": claim["adjudication_outcome"],
                "statement": claim["statement"],
                "classification": claim["classification"],
                "published_status": claim["published_status"],
                "audit_status": claim["audit_status"],
                "evidence_qualifications": qualifications,
                "source_locations": claim["source_locations"],
                "evidence_artifacts": claim["evidence_artifacts"],
                "model_domain": claim["model_domain"],
                "uncertainty_boundary": claim["uncertainty_boundary"],
                "falsifier": claim["falsifier"],
                "adjudication": claim["adjudication"],
                "reviewer": claim["reviewer"],
                "last_verified_on": claim["last_verified_on"],
            }
        )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source_registry": registry_path.name,
        "paper_source_digest": registry["paper"]["source_digest"],
        "claim_count": len(rows),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "evidence_qualification_counts": dict(sorted(qualification_counts.items())),
        "interpretation": (
            "Normalized outcomes apply to declared claim estimands. Evidence "
            "qualifications expose source-review and validation boundaries and do "
            "not promote an outcome."
        ),
        "claims": rows,
    }


def _csv_text(summary: dict[str, Any]) -> str:
    fields = [
        "claim_id",
        "adjudication_outcome",
        "classification",
        "published_status",
        "audit_status",
        "evidence_qualifications",
        "statement",
        "model_domain",
        "uncertainty_boundary",
        "falsifier",
        "adjudication",
        "reviewer",
        "last_verified_on",
        "source_locations",
        "evidence_artifacts",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for claim in summary["claims"]:
        row = dict(claim)
        for field in (
            "evidence_qualifications",
            "source_locations",
            "evidence_artifacts",
        ):
            row[field] = " | ".join(row[field])
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue()


def write_summary(root: Path) -> dict[str, Any]:
    data = root / "docs/research/proximal_distal_energy_transfer/data"
    summary = build_summary(data / "claim_audit_registry.json")
    (data / "claim_adjudication_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (data / "claim_adjudication_summary.csv").write_text(
        _csv_text(summary), encoding="utf-8", newline=""
    )
    return summary


def validate_summary(root: Path) -> dict[str, Any]:
    data = root / "docs/research/proximal_distal_energy_transfer/data"
    expected = build_summary(data / "claim_audit_registry.json")
    actual = json.loads(
        (data / "claim_adjudication_summary.json").read_text(encoding="utf-8")
    )
    if actual != expected:
        raise ValueError("Committed claim adjudication JSON is stale")
    expected_csv = _csv_text(expected)
    actual_csv = (data / "claim_adjudication_summary.csv").read_text(encoding="utf-8")
    if actual_csv != expected_csv:
        raise ValueError("Committed claim adjudication CSV is stale")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "validate"))
    args = parser.parse_args()
    root = _repository_root()
    summary = write_summary(root) if args.command == "write" else validate_summary(root)
    print(
        json.dumps(
            {"claim_count": summary["claim_count"], **summary["outcome_counts"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
