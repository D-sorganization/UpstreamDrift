"""Build reviewer-facing claim outcome and evidence-qualification tables."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
from typing import Any

SUMMARY_SCHEMA_VERSION = "proximal-distal-claim-adjudication-summary-v2"


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


def _external_profiles(review_path: Path) -> dict[str, dict[str, Any]]:
    """Return claim-level profiles from the governed canonical-work review."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    profiles: dict[str, dict[str, Any]] = {}
    for work in review["works"]:
        supported = set(work["supports_claims"])
        for claim_id in work["linked_claims"]:
            profile = profiles.setdefault(
                claim_id, {"linked": [], "supporting": [], "roles": set()}
            )
            profile["linked"].append(work)
            if claim_id in supported and work["evidence_disposition"] == "eligible":
                profile["supporting"].append(work)
                profile["roles"].add(f"external_{work['evidence_role']}")
    return profiles


def _source_independence(profile: dict[str, Any] | None) -> str:
    if profile is None:
        return "project_only"
    supporting = profile["supporting"]
    if not supporting:
        return "external_context_only"
    independent_count = sum(
        work["independence"] == "independent_of_project" for work in supporting
    )
    if independent_count >= 2:
        return "multiple_independent_external_support"
    if independent_count == 1:
        return "single_independent_external_support"
    if any(work["independence"] == "project_author_overlap" for work in supporting):
        return "project_author_overlap_support"
    return "external_independence_unclear"


def _evidence_tiers(claim: dict[str, Any], profile: dict[str, Any] | None) -> list[str]:
    tiers = set(profile["roles"] if profile else ())
    for artifact in claim["evidence_artifacts"]:
        if artifact.startswith(("http://", "https://")):
            continue
        path = artifact.partition("#")[0]
        suffix = Path(path).suffix.lower()
        if suffix in {".py", ".json", ".npz", ".csv"}:
            tiers.add("project_executable_or_generated")
        elif suffix == ".bib":
            tiers.add("bibliographic_record")
        else:
            tiers.add("project_derivation_or_document")
    if claim["adjudication_outcome"] == "untested":
        tiers.add("prospective_or_unexecuted")
    if not tiers:
        tiers.add("declared_internal_review_only")
    return sorted(tiers)


def _model_tiers(claim: dict[str, Any]) -> list[str]:
    """Classify the declared model surface without changing claim outcome."""
    text = " ".join(
        [claim["classification"], claim["model_domain"], *claim["source_locations"]]
    ).lower()
    tiers: set[str] = set()
    mappings = {
        "articulated_spatial": ("articulated", "_ch06ca_"),
        "spatial_reduced": ("spatial", "_ch06c_", "_ch06cb_", "_ch06cc_"),
        "compliant_shaft": ("shaft", "flex", "_ch06_", "_ch06b_"),
        "two_hand_or_bilateral": (
            "two_hand",
            "two-hand",
            "bilateral",
            "distributed_grip",
            "_ch05_",
            "_ch05b_",
            "_ch06bc_",
        ),
        "finite_ground_or_base": ("finite_ground", "ground", "mobile_hub"),
        "planar_reduced": (
            "double-pendulum",
            "planar",
            "_ch02_",
            "_ch03_interaction_",
            "_ch04_",
        ),
        "uncertainty_or_control": ("uncertainty", "_ch06d_"),
        "human_or_external_evidence": (
            "external_",
            "human",
            "participant",
            "_ch03_evidence_",
            "_ch06e_",
        ),
        "cross_tier_synthesis": (
            "cross_tier",
            "model_ladder",
            "_ch01_",
            "_ch07_",
            "_ch08",
        ),
    }
    for tier, tokens in mappings.items():
        if any(token in text for token in tokens):
            tiers.add(tier)
    if not tiers:
        tiers.add("declared_model_tier_unclassified")
    return sorted(tiers)


def _unresolved_replication_classes(claim: dict[str, Any]) -> list[str]:
    text = " ".join(
        (
            claim["audit_status"],
            claim["published_status"],
            claim["statement"],
            claim["uncertainty_boundary"],
        )
    ).lower()
    classes: set[str] = set()
    token_classes = {
        "independent_reimplementation_open": ("reimplementation_open",),
        "statistical_reanalysis_or_replication_open": (
            "reanalysis_open",
            "replication_open",
        ),
        "external_validation_open": ("external_validation_open",),
        "systematic_review_open": ("systematic_review_open", "matrix_incomplete"),
        "full_text_or_original_source_open": (
            "full_text_reanalysis_open",
            "full_reanalysis_open",
            "abstract_only",
        ),
    }
    for class_name, tokens in token_classes.items():
        if any(token in text for token in tokens):
            classes.add(class_name)
    if claim["adjudication_outcome"] == "untested":
        classes.add("prospective_experiment_open")
        if any(token in text for token in ("human", "participant", "bilateral")):
            classes.add("governed_human_data_unavailable")
    if claim["adjudication_outcome"] == "inconclusive":
        classes.add("mixed_or_insufficient_evidence")
    if not classes:
        classes.add("none_declared")
    return sorted(classes)


def _update_list_counts(counter: Counter[str], values: list[str]) -> None:
    counter.update(values)


def _claim_family(claim: dict[str, Any]) -> str:
    source_path = claim["source_locations"][0].rpartition(":")[0]
    return Path(source_path).stem.lstrip("_")


def _family_independence_summary(
    family_counts: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, counts in sorted(family_counts.items()):
        total = sum(counts.values())
        project_only = counts["project_only"]
        single = counts["single_independent_external_support"]
        multiple = counts["multiple_independent_external_support"]
        if project_only == total:
            flag = "project_authored_only"
        elif multiple == 0:
            flag = "external_support_concentrated"
        else:
            flag = "mixed_independence"
        rows.append(
            {
                "claim_family": family,
                "claim_count": total,
                "project_only_count": project_only,
                "single_independent_external_support_count": single,
                "multiple_independent_external_support_count": multiple,
                "concentration_flag": flag,
            }
        )
    return rows


def build_summary(registry_path: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    claims = registry["claims"]
    external_profiles = _external_profiles(
        registry_path.with_name("external_source_review.json")
    )
    outcome_counts = Counter(claim["adjudication_outcome"] for claim in claims)
    qualification_counts: Counter[str] = Counter()
    evidence_tier_counts: Counter[str] = Counter()
    independence_counts: Counter[str] = Counter()
    model_tier_counts: Counter[str] = Counter()
    replication_counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {}
    rows: list[dict[str, Any]] = []
    for claim in claims:
        qualifications = _evidence_qualifications(claim)
        external_profile = external_profiles.get(claim["claim_id"])
        evidence_tiers = _evidence_tiers(claim, external_profile)
        source_independence = _source_independence(external_profile)
        model_tiers = _model_tiers(claim)
        replication_classes = _unresolved_replication_classes(claim)
        claim_family = _claim_family(claim)
        qualification_counts.update(qualifications)
        _update_list_counts(evidence_tier_counts, evidence_tiers)
        independence_counts[source_independence] += 1
        _update_list_counts(model_tier_counts, model_tiers)
        _update_list_counts(replication_counts, replication_classes)
        family_counts.setdefault(claim_family, Counter())[source_independence] += 1
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "adjudication_outcome": claim["adjudication_outcome"],
                "statement": claim["statement"],
                "classification": claim["classification"],
                "published_status": claim["published_status"],
                "audit_status": claim["audit_status"],
                "evidence_qualifications": qualifications,
                "evidence_tiers": evidence_tiers,
                "source_independence": source_independence,
                "model_tiers": model_tiers,
                "unresolved_replication_classes": replication_classes,
                "claim_family": claim_family,
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
        "evidence_tier_counts": dict(sorted(evidence_tier_counts.items())),
        "source_independence_counts": dict(sorted(independence_counts.items())),
        "model_tier_counts": dict(sorted(model_tier_counts.items())),
        "unresolved_replication_counts": dict(sorted(replication_counts.items())),
        "claim_family_source_concentration": _family_independence_summary(
            family_counts
        ),
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
        "evidence_tiers",
        "source_independence",
        "model_tiers",
        "unresolved_replication_classes",
        "claim_family",
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
            "evidence_tiers",
            "model_tiers",
            "unresolved_replication_classes",
            "source_locations",
            "evidence_artifacts",
        ):
            row[field] = " | ".join(row[field])
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue()


def _count_table(
    heading: str, first_column: str, counts: dict[str, int], label: str
) -> list[str]:
    rows = [
        f"### {heading}",
        "",
        f"| {first_column} | Claim Count |",
        "| --- | ---: |",
    ]
    rows.extend(f"| `{name}` | {count} |" for name, count in counts.items())
    rows.extend(("", f": {heading} for the Current Claim Census. {{#{label}}}", ""))
    return rows


def _reviewer_qmd(summary: dict[str, Any]) -> str:
    lines = [
        "<!-- Generated by claim_adjudication_summary.py; do not hand-edit. -->",
        "## Normalized Claim Adjudication",
        "",
        f"The current census contains {summary['claim_count']} material claims. "
        "The normalized outcome applies only to each claim's declared estimand, "
        "model domain, and uncertainty boundary. A `supported` model-conditional "
        "claim is not thereby independently replicated, empirically validated, or "
        "converted into a human strategy recommendation.",
        "",
        "The absence of a `contradicted` row does not mean that every mechanism "
        "survived every adverse test. Claims that accurately report null, mixed, "
        "or adverse model results can themselves be supported. The complete "
        "finding-level reasons, falsifiers, locators, and boundaries are available "
        "in the [reviewer JSON](data/claim_adjudication_summary.json) and "
        "[reviewer CSV](data/claim_adjudication_summary.csv).",
        "",
    ]
    tables = (
        (
            "Outcome Counts",
            "Normalized Outcome",
            summary["outcome_counts"],
            "tbl-claim-outcomes",
        ),
        (
            "Evidence Tier Counts",
            "Evidence Tier",
            summary["evidence_tier_counts"],
            "tbl-claim-evidence-tiers",
        ),
        (
            "Source Independence Counts",
            "Source Independence",
            summary["source_independence_counts"],
            "tbl-claim-source-independence",
        ),
        (
            "Model Tier Counts",
            "Model Tier",
            summary["model_tier_counts"],
            "tbl-claim-model-tiers",
        ),
        (
            "Unresolved Replication Counts",
            "Replication Class",
            summary["unresolved_replication_counts"],
            "tbl-claim-replication",
        ),
    )
    for heading, first_column, counts, label in tables:
        if heading == "Evidence Tier Counts":
            lines.extend(("```{=latex}", "\\newpage", "```", ""))
        lines.extend(_count_table(heading, first_column, counts, label))
    lines.extend(
        (
            "```{=latex}",
            "\\newpage",
            "```",
            "",
            "### Claim-Family Source Concentration",
            "",
            "| Claim Family | Claims | Source-Category Tuple | Flag |",
            "| --- | ---: | ---: | --- |",
        )
    )
    for row in summary["claim_family_source_concentration"]:
        family = row["claim_family"].replace("_", " ")
        flag = row["concentration_flag"].replace("_", " ")
        source_tuple = (
            f"{row['project_only_count']} / "
            f"{row['single_independent_external_support_count']} / "
            f"{row['multiple_independent_external_support_count']}"
        )
        lines.append(f"| {family} | {row['claim_count']} | {source_tuple} | {flag} |")
    lines.extend(
        (
            "",
            ": Claim-Family Source Concentration Flags; the tuple is project "
            "only / one independent work / two or more independent works. "
            "{#tbl-claim-family-concentration}",
            "",
        )
    )
    lines.extend(
        (
            "Evidence-tier, model-tier, and unresolved-replication labels are "
            "nonexclusive, so those totals can exceed the claim count. Source "
            "independence is an exact one-category partition derived from the "
            "canonical-work review. No governed participant outcome is present; "
            "human validation remains an external-data boundary.",
            "",
        )
    )
    return "\n".join(lines)


def write_summary(root: Path) -> dict[str, Any]:
    data = root / "docs/research/proximal_distal_energy_transfer/data"
    summary = build_summary(data / "claim_audit_registry.json")
    (data / "claim_adjudication_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (data / "claim_adjudication_summary.csv").write_text(
        _csv_text(summary), encoding="utf-8", newline=""
    )
    chapter = data.parent / "chapters/_claim_adjudication_summary.qmd"
    chapter.write_text(_reviewer_qmd(summary), encoding="utf-8")
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
    chapter = data.parent / "chapters/_claim_adjudication_summary.qmd"
    if chapter.read_text(encoding="utf-8") != _reviewer_qmd(expected):
        raise ValueError("Committed reviewer-facing adjudication table is stale")
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
