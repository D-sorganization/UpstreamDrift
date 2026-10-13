"""Additive final-report pipeline for the bounded matched-delivery supplement.

This module reads accepted matrix bundles and writes a separate report tree. It
never edits the original matrix plans, cell artifacts, or report inventories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import PATTERN_NAMES
from .matched_lean_control import analyze_explicit_source_pair
from .matched_report import (
    load_matrix_report,
    load_reviewed_common_targets,
    validate_original_matrix_protocol,
)
from .matched_supplement_report import (
    load_supplement_sources,
    write_supplement_artifacts,
    write_supplement_comparisons,
)
from .matched_formatting import format_sg_interval_record
from .provenance import manifest_files_match, sha256_file
from .uncertainty import HISTORICAL_RESAMPLES
from .matched_supplement_figures import (
    _condition,
    _render_cell_figures,
    _render_pair_figures,
    validate_planned_figure_identities,
)

_PATTERNS = PATTERN_NAMES
_ROOT_REQUIREMENTS = {"original_primary": 324, "original_control": 72}


_NUMERIC_ARTIFACT_NAMES = frozenset(
    {
        "REPORT_SUPPLEMENT.json",
        "REPORT_SUPPLEMENT.csv",
        "control_aliases.json",
        "REPORT_SUPPLEMENT_COMPARISONS.json",
        "REPORT_SUPPLEMENT_COMPARISONS.csv",
        "REPORT_SUPPLEMENT_COMPARISONS.md",
        "REPORT_ORIGINAL_SOURCE_INVENTORY.csv",
    }
)


def _load_numeric_parity_reference(path: Path | None) -> dict[str, Any] | None:
    """Bind an optional exact seven-file reference before any computation."""
    if path is None:
        return None
    path = Path(path).resolve()

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate numeric reference key")
            result[key] = value
        return result

    raw = path.read_bytes()
    values = json.loads(raw, object_pairs_hook=unique_object)
    if (
        not isinstance(values, dict)
        or set(values) != _NUMERIC_ARTIFACT_NAMES
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in values.values()
        )
    ):
        raise ValueError("numeric reference must contain exactly seven SHA256 hashes")
    digest = hashlib.sha256(raw).hexdigest()
    if sha256_file(path) != digest:
        raise ValueError("numeric reference changed while reading")
    return {"path": str(path), "sha256": digest, "artifact_sha256": values}


def _check_numeric_parity_reference(
    reference: dict[str, Any] | None, output_dir: Path
) -> None:
    """Recheck both the reference bytes and the generated numerical artifacts."""
    if reference is None:
        return
    if not isinstance(reference, dict) or set(reference) != {
        "path",
        "sha256",
        "artifact_sha256",
    }:
        raise ValueError("malformed numeric parity reference")
    current = _load_numeric_parity_reference(Path(reference["path"]))
    if current != reference:
        raise ValueError("numeric reference changed")
    for name, expected in reference["artifact_sha256"].items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"numeric artifact differs from expected bytes: {name}")


def _report_source_hashes() -> dict[str, str]:
    """Bind postprocessing code separately from the saved physics snapshot."""
    package = Path(__file__).parent
    names = (
        "matched_supplement_pipeline.py",
        "matched_formatting.py",
        "noise.py",
        "matched_supplement_figures.py",
        "matched_figure_identity.py",
        "matched_supplement_report.py",
        "matched_lean_control.py",
        "matched_report.py",
        "matched_report_plots.py",
        "academic_plots.py",
        "density.py",
        "matched_sensitivity.py",
        "course_score_vectors.py",
        "scenario_scoring.py",
        "uncertainty.py",
        "matched_pair_social.py",
    )
    return {name: sha256_file(package / name) for name in names}


def _source_cell_hashes(refs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Capture execution-time identities for every materialized source cell."""
    hashes = {}
    for ref in refs:
        key = f"{ref['source_protocol']}:{ref['cell']}"
        if key in hashes:
            raise ValueError(f"duplicate source cell identity: {key}")
        hashes[key] = {
            "source_root": str(Path(ref["source_root"]).resolve()),
            "status": ref["status"],
            "config": ref["config"],
            "manifest_sha256": ref.get("source_cell_manifest_sha256"),
            "summary_sha256": ref.get("source_summary_sha256"),
            "csv_sha256": ref.get("source_csv_sha256"),
        }
    return hashes


def _assert_source_cell_hashes(
    refs: list[dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> None:
    if _source_cell_hashes(refs) != expected:
        raise ValueError("source-cell artifacts changed during verification")


def validate_final_matrix_report(
    report: dict[str, Any], *, expected_cells: int
) -> None:
    """Require a complete final-stage matrix with 10k-shot configs."""
    if report.get("stage") != "final":
        raise ValueError("original matrix stage must be final")
    if report.get("planned_cells") != expected_cells:
        raise ValueError("original matrix planned cell count differs from contract")
    counts = report.get("counts")
    if not isinstance(counts, dict) or counts.get("not_run") != 0:
        raise ValueError("original final matrix is incomplete")
    cells = report.get("cells")
    if not isinstance(cells, list) or len(cells) != expected_cells:
        raise ValueError("original matrix cell inventory is incomplete")
    for cell in cells:
        if cell.get("config", {}).get("n_shots") != 10_000:
            raise ValueError("original final matrix cells must use 10,000 shots")


def render_original_source_inventory(reports: dict[str, dict]) -> str:
    """List every original planned cell and its saved availability state."""
    import csv
    import io

    output = io.StringIO()
    fields = (
        "source_protocol",
        "cell",
        "status",
        "unavailable_reason",
        "target_x_m",
        "n_shots",
        "source_plan_sha256",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for protocol in ("original_primary", "original_control"):
        report = reports[protocol]
        for cell in report.get("cells", []):
            summary = cell.get("summary", {})
            writer.writerow(
                {
                    "source_protocol": f"{protocol}/1",
                    "cell": cell.get("cell", ""),
                    "status": cell.get("status", ""),
                    "unavailable_reason": cell.get(
                        "unavailable_reason",
                        cell.get("reason", summary.get("reason", "")),
                    ),
                    "target_x_m": cell.get("target_x_m", ""),
                    "n_shots": cell.get("config", {}).get("n_shots", ""),
                    "source_plan_sha256": report.get("plan_sha256", ""),
                }
            )
    return output.getvalue()


def _load_original_sources(root_value: Path, protocol: str) -> tuple[dict, list[dict]]:
    root = Path(root_value).resolve()
    expected = _ROOT_REQUIREMENTS[protocol]
    report = load_matrix_report(root, require_complete=True)
    validate_final_matrix_report(report, expected_cells=expected)
    targets_path = root.parent / "common_targets.json"
    targets = None
    target_provenance = None
    reviewed_targets, target_provenance = load_reviewed_common_targets(root.parent)
    if protocol == "original_control":
        targets = reviewed_targets
    validate_original_matrix_protocol(report, protocol=protocol, common_targets=targets)
    report["common_targets_path"] = str(targets_path.resolve())
    report["common_targets_sha256"] = sha256_file(targets_path)
    report["common_target_provenance"] = target_provenance
    plan_path = root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    references = []
    for cell in report["cells"]:
        name = cell["cell"]
        config = cell["config"]
        cell_dir = root / name
        manifest_path = cell_dir / "manifest.json"
        csv_path = cell_dir / "shots.csv.gz"
        references.append(
            {
                "cell": name,
                "source_protocol": f"{protocol}/1",
                "source_root": str(root),
                "execution_kind": "unique",
                "status": cell["status"],
                "config": config,
                "summary": cell,
                "source_plan_sha256": report["plan_sha256"],
                "source_cell_manifest_sha256": report["cell_manifest_sha256"].get(name),
                "source_summary_sha256": (
                    sha256_file(cell_dir / "summary.json")
                    if (cell_dir / "summary.json").is_file()
                    else None
                ),
                "source_csv_sha256": sha256_file(csv_path)
                if csv_path.is_file()
                else None,
                "_execution": plan["execution"],
            }
        )
    return report, references


def select_explicit_pairs(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select only fully matched curve, lean, and shaft-SD condition pairs."""
    usable = [cell for cell in cells if cell.get("status") == "completed"]
    # Aliases deliberately win for a fixed-target PW control when they refer to
    # the same bytes as a unique source; all provenance remains in the result.
    priority = {"provenance_alias": 0, "original_control/1": 1, "original_primary/1": 2}
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in sorted(
        usable,
        key=lambda item: priority.get(
            "provenance_alias"
            if item.get("execution_kind") == "provenance_alias"
            else item.get("source_protocol", ""),
            3,
        ),
    ):
        identity = (cell["cell"], cell.get("source_csv_sha256") or "")
        deduplicated.setdefault(identity, cell)
    usable = list(deduplicated.values())

    def grouped(axis: str) -> dict[tuple, dict[float, dict[str, Any]]]:
        groups: dict[tuple, dict[float, dict[str, Any]]] = defaultdict(dict)
        for cell in usable:
            c = _condition(cell)
            if axis == "curve":
                key = (c["club"], c["lean"], c["rotation"], c["profile"], c["target"])
                value = c["curve"]
            elif axis == "rotation":
                key = (c["club"], c["lean"], c["curve"], c["profile"], c["target"])
                value = c["rotation"]
            else:
                key = (c["club"], c["curve"], c["rotation"], c["profile"], c["target"])
                value = c["lean"]
            existing = groups[key].get(value)
            if existing is None or priority.get(
                cell.get("source_protocol", ""), 3
            ) < priority.get(existing.get("source_protocol", ""), 3):
                groups[key][value] = cell
        return groups

    pairs: list[dict[str, Any]] = []
    for kind, axis, levels in (
        ("curve_strength", "curve", (1.0, 2.0)),
        ("shaft_rotation_sd", "rotation", (1.0, 2.0)),
    ):
        for by_value in grouped(axis).values():
            if all(level in by_value for level in levels):
                pairs.append(_pair_spec(kind, by_value[levels[0]], by_value[levels[1]]))
    for by_lean in grouped("lean").values():
        for first, second in ((0.0, 5.0), (0.0, 10.0), (5.0, 10.0)):
            if first in by_lean and second in by_lean:
                pairs.append(_pair_spec("across_lean", by_lean[first], by_lean[second]))
    pairs.sort(
        key=lambda pair: (
            pair["comparison_kind"],
            pair["target_x_m"],
            pair["first"]["cell"],
            pair["second"]["cell"],
        )
    )
    return pairs


def _pair_spec(kind: str, first: dict, second: dict) -> dict[str, Any]:
    return {
        "comparison_kind": kind,
        "first": first,
        "second": second,
        "target_x_m": float(first["summary"]["target_x_m"]),
        "physical_patterns": _PATTERNS,
    }


def _pair_result(
    pair: dict[str, Any],
    transition_path: Path,
    transition_sha256: str,
    resamples: int = HISTORICAL_RESAMPLES,
) -> dict[str, Any]:
    first, second = pair["first"], pair["second"]
    same_execution = first["_execution"] == second["_execution"]
    return analyze_explicit_source_pair(
        first,
        second,
        comparison_kind=pair["comparison_kind"],
        target_x_m=pair["target_x_m"],
        transition_receipt_path=None if same_execution else transition_path,
        transition_receipt_sha256=None if same_execution else transition_sha256,
        resamples=resamples,
    )


def _recheck_source_inputs(
    refs: list[dict[str, Any]],
    plan_hashes: dict[str, str],
    inventory_path: Path,
    inventory_sha256: str,
    common_targets_path: Path | None = None,
    common_targets_sha256: str | None = None,
    reviewed_targets_root: Path | None = None,
    reviewed_target_sources: dict[str, str] | None = None,
) -> None:
    for root_text, expected_hash in plan_hashes.items():
        plan_path = Path(root_text) / "matrix_plan.json"
        if not plan_path.is_file() or sha256_file(plan_path) != expected_hash:
            raise ValueError("a source matrix plan changed during report generation")
    for ref in refs:
        status = ref.get("status")
        if status == "not_run":
            continue
        name = ref.get("alias_of_primary", ref["cell"])
        cell_dir = Path(ref["source_root"]) / name
        manifest = cell_dir / "manifest.json"
        summary = cell_dir / "summary.json"
        if status not in {"completed", "unavailable"}:
            raise ValueError(f"invalid source status during report generation: {name}")
        if not manifest.is_file() or not manifest_files_match(manifest):
            raise ValueError(
                f"source cell manifest changed during report generation: {name}"
            )
        if sha256_file(manifest) != ref.get("source_cell_manifest_sha256"):
            raise ValueError(
                f"source cell manifest changed during report generation: {name}"
            )
        expected_summary_hash = ref.get("source_summary_sha256")
        if (
            expected_summary_hash is None
            or not summary.is_file()
            or sha256_file(summary) != expected_summary_hash
        ):
            raise ValueError(
                f"source cell summary changed during report generation: {name}"
            )
        summary_record = json.loads(summary.read_text(encoding="utf-8"))
        if summary_record.get("status") != status or summary_record.get(
            "config"
        ) != ref.get("config"):
            raise ValueError(f"source cell summary status/config changed: {name}")
        if status != "completed":
            continue
        csv_path = cell_dir / "shots.csv.gz"
        if not csv_path.is_file() or sha256_file(csv_path) != ref.get(
            "source_csv_sha256"
        ):
            raise ValueError(f"source cell changed during report generation: {name}")
    if sha256_file(inventory_path) != inventory_sha256:
        raise ValueError("supplement inventory changed during report generation")
    if common_targets_path is not None and (
        common_targets_sha256 is None
        or not common_targets_path.is_file()
        or sha256_file(common_targets_path) != common_targets_sha256
    ):
        raise ValueError(
            "original control common targets changed during report generation"
        )
    if reviewed_targets_root is not None:
        if reviewed_target_sources is None:
            raise ValueError("reviewed target source hashes are required")
        evidence_paths = {
            "acceptance_sha256": reviewed_targets_root / "FINAL_RUN_ACCEPTANCE.md",
            "audit_sha256": reviewed_targets_root
            / "validation/independent_control_pilot_audit.json",
            "pilot_plan_sha256": reviewed_targets_root
            / "control_pilot_v1/matrix_plan.json",
            "common_targets_sha256": reviewed_targets_root / "common_targets.json",
        }
        if set(evidence_paths) != set(reviewed_target_sources):
            raise ValueError("reviewed target source inventory is incomplete")
        for label, path in evidence_paths.items():
            if (
                not path.is_file()
                or sha256_file(path) != reviewed_target_sources[label]
            ):
                raise ValueError("reviewed common-target provenance changed")


def _recheck_transition_evidence(receipt_path: Path, expected_sha256: str) -> None:
    if not receipt_path.is_file() or sha256_file(receipt_path) != expected_sha256:
        raise ValueError("source-transition receipt changed during report generation")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    evidence_files = receipt.get("evidence_files", {})
    if not isinstance(evidence_files, dict) or not evidence_files:
        raise ValueError("source-transition receipt lacks bound review evidence")
    for relative, expected in evidence_files.items():
        path = (receipt_path.parent / relative).resolve()
        if not path.is_relative_to(receipt_path.parent) or not path.is_file():
            raise ValueError("source-transition evidence changed or escaped its root")
        if sha256_file(path) != expected:
            raise ValueError(
                "source-transition evidence changed during report generation"
            )


def _recheck_receipt_sources(receipt: dict[str, Any]) -> None:
    """Reload original and supplement sources named by a final receipt."""
    roots = receipt.get("original_sources", {})
    required = {"original_primary", "original_control"}
    if set(roots) != required:
        raise ValueError("pipeline receipt has an incomplete original source map")
    reports: dict[str, dict[str, Any]] = {}
    refs: list[dict[str, Any]] = []
    plan_hashes: dict[str, str] = {}
    for key in sorted(required):
        root = Path(roots[key]["root"]).resolve()
        report, source_refs = _load_original_sources(root, key)
        if report["plan_sha256"] != roots[key].get("plan_sha256"):
            raise ValueError(f"original {key} plan differs from pipeline receipt")
        reports[key] = report
        refs.extend(source_refs)
        plan_hashes[str(root)] = report["plan_sha256"]

    protocol_records = receipt.get("supplement_protocols", {})
    protocol_roots = {
        protocol: Path(record["source_root"])
        for protocol, record in protocol_records.items()
    }
    if len(protocol_roots) != 3:
        raise ValueError("pipeline receipt must name all three supplement protocols")
    supplement = load_supplement_sources(
        protocol_roots, Path(receipt["inventory_path"]), stage="final"
    )
    if supplement["inventory_sha256"] != receipt.get("supplement_inventory_sha256"):
        raise ValueError("supplement inventory differs from pipeline receipt")
    for protocol, record in supplement["protocols"].items():
        expected = protocol_records.get(protocol, {})
        if record.get("plan_sha256") != expected.get("plan_sha256"):
            raise ValueError(
                f"supplement {protocol} plan differs from pipeline receipt"
            )
        plan_hashes[str(Path(record["source_root"]).resolve())] = record["plan_sha256"]
    refs.extend(supplement["unique_cells"])
    expected_source_cells = receipt.get("source_cells_sha256")
    if not isinstance(expected_source_cells, dict):
        raise ValueError("pipeline receipt has no source-cell artifact inventory")
    _assert_source_cell_hashes(refs, expected_source_cells)
    control = reports["original_control"]
    _recheck_source_inputs(
        refs,
        plan_hashes,
        Path(receipt["inventory_path"]),
        supplement["inventory_sha256"],
        Path(control["common_targets_path"]),
        control["common_targets_sha256"],
        Path(roots["original_control"]["root"]).resolve().parent,
        control["common_target_provenance"],
    )
    # Reload after the byte checks so a source cannot change during receipt
    # validation and still be returned under an earlier snapshot.
    second_refs: list[dict[str, Any]] = []
    for key in sorted(required):
        _, source_refs = _load_original_sources(Path(roots[key]["root"]), key)
        second_refs.extend(source_refs)
    second_supplement = load_supplement_sources(
        protocol_roots, Path(receipt["inventory_path"]), stage="final"
    )
    second_refs.extend(second_supplement["unique_cells"])
    _assert_source_cell_hashes(second_refs, expected_source_cells)
    _recheck_transition_evidence(
        Path(receipt["source_transition_receipt_path"]),
        receipt["source_transition_receipt_sha256"],
    )


def _verify_figure_assets(output_dir: Path, ledger: dict[str, Any]) -> None:
    figure_root = (output_dir / "figures").resolve()
    seen_paths: set[Path] = set()
    for item in ledger.get("figures", []):
        if item.get("status") == "unavailable":
            if not item.get("reason") or any(
                key in item for key in ("relative_path", "sha256")
            ):
                raise ValueError(
                    "unavailable figure must state a reason without artifact claims"
                )
            continue
        if not isinstance(item.get("relative_path"), str) or not item["relative_path"]:
            raise ValueError("available figure must identify its artifact path")
        path = (figure_root / item["relative_path"]).resolve()
        if path in seen_paths:
            raise ValueError("duplicate figure artifact path")
        seen_paths.add(path)
        if not path.is_relative_to(figure_root) or not path.is_file():
            raise ValueError("supplement figure artifact is missing or unsafe")
        if sha256_file(path) != item.get("sha256"):
            raise ValueError("supplement figure artifact changed")


def load_verified_supplement_artifacts(output_dir: Path) -> dict[str, Any]:
    """Return final additive report data only after verifying saved inputs."""
    output_dir = Path(output_dir).resolve()
    receipt_path = output_dir / "REPORT_SUPPLEMENT_PIPELINE.json"
    if not receipt_path.is_file():
        raise ValueError("supplement pipeline receipt is missing")
    receipt_sha256 = sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "matched-supplement-pipeline-receipt/1"
        or receipt.get("stage") != "final"
    ):
        raise ValueError("supplement pipeline receipt is not final")
    artifact_hashes = receipt.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("pipeline receipt has no bound report artifacts")
    artifact_paths = {
        "report": "REPORT_SUPPLEMENT.json",
        "inventory": "REPORT_SUPPLEMENT.csv",
        "control_aliases": "control_aliases.json",
        "comparisons_json": "REPORT_SUPPLEMENT_COMPARISONS.json",
        "comparisons_csv": "REPORT_SUPPLEMENT_COMPARISONS.csv",
        "comparison_markdown": "REPORT_SUPPLEMENT_COMPARISONS.md",
        "original_source_inventory": "REPORT_ORIGINAL_SOURCE_INVENTORY.csv",
        "figure_ledger": "REPORT_SUPPLEMENT_FIGURES.json",
    }
    if set(artifact_hashes) != set(artifact_paths):
        raise ValueError("pipeline receipt artifact inventory is incomplete")
    for key, relative in artifact_paths.items():
        path = (output_dir / relative).resolve()
        if not path.is_relative_to(output_dir) or not path.is_file():
            raise ValueError(f"report artifact is missing or unsafe: {key}")
        if sha256_file(path) != artifact_hashes[key]:
            raise ValueError(f"report artifact changed: {key}")
    if receipt.get("postprocessing_source_sha256") != _report_source_hashes():
        raise ValueError("reporting or figure source changed after generation")
    ledger = json.loads(
        (output_dir / artifact_paths["figure_ledger"]).read_text(encoding="utf-8")
    )
    if ledger.get("schema") != "matched-supplement-figure-ledger/1":
        raise ValueError("supplement figure ledger has an unexpected schema")
    _check_numeric_parity_reference(receipt.get("numeric_parity_reference"), output_dir)
    _recheck_receipt_sources(receipt)
    _verify_figure_assets(output_dir, ledger)
    for key, relative in artifact_paths.items():
        if sha256_file(output_dir / relative) != artifact_hashes[key]:
            raise ValueError(f"report artifact changed during verification: {key}")
    if _report_source_hashes() != receipt.get("postprocessing_source_sha256"):
        raise ValueError("reporting or figure source changed during verification")
    if sha256_file(receipt_path) != receipt_sha256:
        raise ValueError("supplement pipeline receipt changed during verification")
    artifact_bytes = {
        key: (output_dir / relative).read_bytes()
        for key, relative in artifact_paths.items()
    }
    if any(
        hashlib.sha256(artifact_bytes[key]).hexdigest() != artifact_hashes[key]
        for key in artifact_paths
    ):
        raise ValueError("report artifact changed while reading verified outputs")
    if sha256_file(receipt_path) != receipt_sha256:
        raise ValueError("supplement pipeline receipt changed while reading outputs")
    _check_numeric_parity_reference(receipt.get("numeric_parity_reference"), output_dir)
    return {
        "report": json.loads(artifact_bytes["report"]),
        "comparisons": json.loads(artifact_bytes["comparisons_json"]),
        "figure_ledger": json.loads(artifact_bytes["figure_ledger"]),
        "receipt": receipt,
    }


def render_comparison_markdown(results: list[dict[str, Any]]) -> str:
    """Render a compact readable table without merging old report rows."""
    lines = [
        "# Supplemental Matched-Condition Comparisons",
        "",
        "These are paired simulation contrasts on shot IDs supported in all six pattern-by-condition populations. Nominal re-matching and the matched-condition changes remain part of each contrast; the results do not isolate a causal curvature effect.",
        "",
        "| Comparison | Pattern | Conditions | Common N / Total | Radial Error Change (m, 95% CI) | Paired Pattern SG Change (Strokes, 95% CI) | SG Relative-Benefit Interaction (Strokes, 95% CI) | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        sources = result.get("sources", [{}, {}])
        source_label = " → ".join(
            f"{source.get('protocol_id', '')}:{source.get('cell', '')}"
            for source in sources
        )
        effects = result.get("course_scoring_effect_modification", {})
        for pattern in _PATTERNS:
            metrics = result.get("patterns", {}).get(pattern, {})
            radial = metrics.get("radial_error_difference_m", {})
            pattern_sg = metrics.get("course_sg_difference", {})
            effect = effects.get(pattern, {}) if isinstance(effects, dict) else {}
            radial_text = _estimate_ci(radial, " m")
            pattern_sg_text = _estimate_ci(pattern_sg, "")
            effect_text = _estimate_ci(effect, "")
            common = result.get("common_supported_n", 0)
            total = result.get("total_n_per_population", 0)
            lines.append(
                f"| {result.get('comparison_kind', '')} | {pattern} | {source_label} | {common:,} / {total:,} | {radial_text} | {pattern_sg_text} | {effect_text} | {result.get('status', '')} |"
            )
    lines.extend(
        (
            "",
            "**Strokes-Gained Direction:** Paired pattern SG change is second condition minus first; positive is better. Relative-benefit interaction is the pattern change minus the Straight change; positive means greater relative benefit. Both use the same six-population shot-index intersection and require source-backed scoring availability.",
            "",
            "**Scope:** Illustrative matched-delivery simulations with assumed head properties. Carry endpoints and historical course-scoring context do not establish measured-player skill or causal effects.",
            "",
            "**Monte Carlo Uncertainty:** Grid cells share seeded primitive error draws. Paired bootstrap intervals and direction counts describe saved Monte Carlo samples; they are not independent experimental replications or multiplicity-adjusted discoveries.",
            "",
        )
    )
    return "\n".join(lines)


def _estimate_ci(value: dict[str, Any], unit: str) -> str:
    rendered = format_sg_interval_record(value)
    if rendered == "Unavailable":
        return rendered
    estimate, interval = rendered.split(" ", 1)
    return f"{estimate}{unit} {interval}"


@dataclass(frozen=True)
class _PipelineSources:
    original_reports: dict
    original_cells: list[dict]
    supplement: dict
    report_sources: dict
    pair_specs: list[dict]


@dataclass(frozen=True)
class _PublishedHashes:
    artifact_names: dict[str, str]
    artifact_sha256: dict[str, str]
    source_cells_sha256: dict


def _supplement_figure_sources(supplement: dict) -> list[dict]:
    """Expand verified aliases while retaining their source and requested config."""
    supplement_cells = list(supplement["unique_cells"])
    for alias in supplement["aliases"]:
        if alias["proof_status"] == "verified":
            source_ref = next(
                ref
                for ref in supplement["unique_cells"]
                if ref["cell"] == alias["alias_of_primary"]
            )
            alias_ref = {
                **source_ref,
                **alias,
                "cell": alias["alias_cell"],
                "execution_kind": "provenance_alias",
                "summary": source_ref["summary"],
                "config": alias["requested_config"],
                "_execution": supplement["execution"],
            }
            supplement_cells.append(alias_ref)
    for ref in supplement["unique_cells"]:
        ref["_execution"] = supplement["execution"]
    return supplement_cells


def _load_pipeline_sources(
    original_primary_root: Path,
    original_control_root: Path,
    protocol_roots: dict[str, Path],
    inventory_path: Path,
) -> _PipelineSources:
    original_reports: dict[str, dict] = {}
    original_cells: list[dict[str, Any]] = []
    for key, root in (
        ("original_primary", original_primary_root),
        ("original_control", original_control_root),
    ):
        report, refs = _load_original_sources(Path(root), key)
        original_reports[key] = report
        original_cells.extend(refs)
    supplement = load_supplement_sources(protocol_roots, inventory_path, stage="final")
    report_sources = _report_source_hashes()
    supplement_cells = _supplement_figure_sources(supplement)
    cells = [*original_cells, *supplement_cells]
    pair_specs = select_explicit_pairs(cells)
    validate_planned_figure_identities(pair_specs, supplement["unique_cells"])
    return _PipelineSources(
        original_reports, original_cells, supplement, report_sources, pair_specs
    )


SUPPLEMENT_TRANSITION_SHA256 = (
    "cf622694964b0dc6c2ca0c24b92f9fa8ee5365721219069283aca1f6804a8981"
)
"""Accepted hash of ``source_transition_supplement.json`` (see REPRODUCING.md)."""


def explicit_pair_specs(phase2_root: Path) -> list[dict[str, Any]]:
    """Load and validate every explicit pair from a restored phase-two tree.

    ``phase2_root`` uses the published layout: ``final_v1``,
    ``control_final_v1``, ``supplement_final_v1/<protocol>`` and
    ``review_followups/SUPPLEMENTAL_EXECUTION_INVENTORY.json``.
    """
    root = Path(phase2_root)
    supplement = root / "supplement_final_v1"
    return _load_pipeline_sources(
        root / "final_v1",
        root / "control_final_v1",
        {
            "pw-primary-face-bounds/1": supplement / "pw_primary_face",
            "pw-control-face-bounds/1": supplement / "pw_control_face",
            "driver-control-speed-bounds/1": supplement / "driver_control_speed",
        },
        root / "review_followups/SUPPLEMENTAL_EXECUTION_INVENTORY.json",
    ).pair_specs


def analyze_pair_spec(
    pair: dict[str, Any], phase2_root: Path, *, resamples: int
) -> dict[str, Any]:
    """Analyze one explicit pair against the accepted source transition."""
    return _pair_result(
        pair,
        Path(phase2_root) / "source_transition_supplement.json",
        SUPPLEMENT_TRANSITION_SHA256,
        resamples,
    )


def _write_numeric_artifacts(
    sources: _PipelineSources, comparison_results: list[dict], output_dir: Path
) -> dict[str, Path]:
    supplement, original_reports = sources.supplement, sources.original_reports
    # Freeze the selected-cell inputs before any figure is written. Figures are
    # derived only from validated saved CSV/flight-sample data.
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = write_supplement_artifacts(supplement, output_dir)
    original_inventory_path = output_dir / "REPORT_ORIGINAL_SOURCE_INVENTORY.csv"
    original_inventory_path.write_text(
        render_original_source_inventory(original_reports), encoding="utf-8"
    )
    paths["original_source_inventory"] = original_inventory_path
    report_path = output_dir / "REPORT_SUPPLEMENT.json"
    report_path.write_text(json.dumps(supplement, indent=2, sort_keys=True) + "\n")
    paths["report"] = report_path
    paths.update(write_supplement_comparisons(comparison_results, output_dir))
    markdown_path = output_dir / "REPORT_SUPPLEMENT_COMPARISONS.md"
    markdown_path.write_text(
        render_comparison_markdown(comparison_results), encoding="utf-8"
    )
    paths["comparison_markdown"] = markdown_path
    return paths


def _write_pipeline_figures(
    sources: _PipelineSources, comparison_results: list[dict], output_dir: Path
) -> tuple[Path, list[dict]]:
    supplement, pair_specs = sources.supplement, sources.pair_specs
    figure_dir = output_dir / "figures"
    figure_ledger = []
    for ref in supplement["unique_cells"]:
        figure_ledger.extend(_render_cell_figures(ref, figure_dir))
    for pair, result in zip(pair_specs, comparison_results, strict=True):
        if result.get("status") in {"available", "conditional_only"}:
            figure_ledger.extend(_render_pair_figures(pair, figure_dir))
    ledger_path = output_dir / "REPORT_SUPPLEMENT_FIGURES.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema": "matched-supplement-figure-ledger/1",
                "stage": "final",
                "source_inventory_sha256": supplement["inventory_sha256"],
                "figures": figure_ledger,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ledger_path, figure_ledger


def _recheck_pipeline_inputs(
    sources: _PipelineSources, roots: tuple[Path, Path], inventory_path: Path
) -> None:
    original_primary_root, original_control_root = roots
    original_reports, original_cells = sources.original_reports, sources.original_cells
    supplement = sources.supplement
    plan_hashes = {
        str(Path(original_primary_root).resolve()): original_reports[
            "original_primary"
        ]["plan_sha256"],
        str(Path(original_control_root).resolve()): original_reports[
            "original_control"
        ]["plan_sha256"],
        **{
            record["source_root"]: record["plan_sha256"]
            for record in supplement["protocols"].values()
        },
    }
    _recheck_source_inputs(
        [*original_cells, *supplement["unique_cells"]],
        plan_hashes,
        Path(inventory_path),
        supplement["inventory_sha256"],
        Path(original_reports["original_control"]["common_targets_path"]),
        original_reports["original_control"]["common_targets_sha256"],
        Path(original_control_root).resolve().parent,
        original_reports["original_control"]["common_target_provenance"],
    )


def _original_source_receipts(original_reports: dict, roots: tuple[Path, Path]) -> dict:
    original_primary_root, original_control_root = roots
    return {
        key: {
            "root": str(Path(root).resolve()),
            "plan_sha256": original_reports[key]["plan_sha256"],
            "planned_cells": original_reports[key]["planned_cells"],
            "counts": original_reports[key]["counts"],
            "common_targets_sha256": original_reports[key].get("common_targets_sha256"),
            "common_target_provenance": original_reports[key].get(
                "common_target_provenance"
            ),
        }
        for key, root in (
            ("original_primary", original_primary_root),
            ("original_control", original_control_root),
        )
    }


def _pipeline_receipt_payload(
    sources: _PipelineSources,
    roots: tuple[Path, Path],
    inventory_path: Path,
    output_dir: Path,
    transition: tuple[Path, str],
    comparison_results: list[dict],
    artifacts: _PublishedHashes,
    numeric_reference: dict | None,
) -> dict:
    original_reports, supplement = sources.original_reports, sources.supplement
    report_sources = sources.report_sources
    transition_path, transition_receipt_sha256 = transition
    original_inventory_path = output_dir / "REPORT_ORIGINAL_SOURCE_INVENTORY.csv"
    ledger_path = output_dir / "REPORT_SUPPLEMENT_FIGURES.json"
    artifact_names, artifact_sha256 = (
        artifacts.artifact_names,
        artifacts.artifact_sha256,
    )
    source_cells_sha256 = artifacts.source_cells_sha256
    return {
        **(
            {"numeric_parity_reference": numeric_reference}
            if numeric_reference is not None
            else {}
        ),
        "schema": "matched-supplement-pipeline-receipt/1",
        "stage": "final",
        "issue": 12000,
        "original_sources": _original_source_receipts(original_reports, roots),
        "supplement_inventory_sha256": supplement["inventory_sha256"],
        "original_source_inventory_sha256": sha256_file(original_inventory_path),
        "source_transition_receipt_path": str(transition_path),
        "inventory_path": str(Path(inventory_path).resolve()),
        "artifact_sha256": artifact_sha256,
        "artifact_paths": artifact_names,
        "source_cells_sha256": source_cells_sha256,
        "report_source_sha256": report_sources,
        "supplement_protocols": supplement["protocols"],
        "unique_executed_cells": supplement["unique_executed_cells"],
        "verified_control_aliases": [
            ref["alias_cell"]
            for ref in supplement["control_aliases"]["references"]
            if supplement["control_aliases"]["status_by_alias"]
            .get(ref["alias_cell"], {})
            .get("proof_status")
            == "verified"
        ],
        "comparison_count": len(comparison_results),
        "comparison_counts_by_kind": {
            kind: sum(
                result.get("comparison_kind") == kind for result in comparison_results
            )
            for kind in (
                "curve_strength",
                "shaft_rotation_sd",
                "across_lean",
                "speed_bound",
            )
        },
        "source_transition_receipt_sha256": transition_receipt_sha256,
        "comparison_analysis_source_sha256": sorted(
            {
                json.dumps(result.get("analysis_source_sha256", {}), sort_keys=True)
                for result in comparison_results
            }
        ),
        "postprocessing_source_sha256": report_sources,
        "figure_ledger_sha256": sha256_file(ledger_path),
        "interpretation": "separate add-on report; original matrices and inventories remain unchanged",
        "monte_carlo_uncertainty": "shared seeded primitive draws; paired intervals describe saved simulation samples, not independent replications or multiplicity-adjusted discoveries",
    }


def _write_pipeline_receipt(
    sources: _PipelineSources,
    roots: tuple[Path, Path],
    inventory_path: Path,
    output_dir: Path,
    transition: tuple[Path, str],
    comparison_results: list[dict],
    figure_ledger: list[dict],
    numeric_reference: dict | None,
) -> Path:
    original_cells, supplement = sources.original_cells, sources.supplement
    artifact_names = {
        "report": "REPORT_SUPPLEMENT.json",
        "inventory": "REPORT_SUPPLEMENT.csv",
        "control_aliases": "control_aliases.json",
        "comparisons_json": "REPORT_SUPPLEMENT_COMPARISONS.json",
        "comparisons_csv": "REPORT_SUPPLEMENT_COMPARISONS.csv",
        "comparison_markdown": "REPORT_SUPPLEMENT_COMPARISONS.md",
        "original_source_inventory": "REPORT_ORIGINAL_SOURCE_INVENTORY.csv",
        "figure_ledger": "REPORT_SUPPLEMENT_FIGURES.json",
    }
    artifact_sha256 = {
        key: sha256_file(output_dir / relative)
        for key, relative in artifact_names.items()
    }
    source_cells_sha256 = _source_cell_hashes(
        [*original_cells, *supplement["unique_cells"]]
    )
    _check_numeric_parity_reference(numeric_reference, output_dir)
    _verify_figure_assets(output_dir, {"figures": figure_ledger})
    payload = _pipeline_receipt_payload(
        sources,
        roots,
        inventory_path,
        output_dir,
        transition,
        comparison_results,
        _PublishedHashes(artifact_names, artifact_sha256, source_cells_sha256),
        numeric_reference,
    )
    pipeline_path = output_dir / "REPORT_SUPPLEMENT_PIPELINE.json"
    pipeline_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return pipeline_path


def build_final_supplement_report(
    *,
    original_primary_root: Path,
    original_control_root: Path,
    protocol_roots: dict[str, Path],
    inventory_path: Path,
    output_dir: Path,
    transition_receipt_path: Path,
    transition_receipt_sha256: str,
    expected_numeric_hashes_path: Path | None = None,
) -> dict[str, Path]:
    """Validate accepted roots, compare common-index populations, and export.

    A source-transition receipt is required because the supplement execution
    may differ from original matrix sources. The pair API validates it for each
    cross-snapshot pair; same-snapshot pairs need no receipt.
    """
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("supplement report output directory must be empty")
    numeric_reference = _load_numeric_parity_reference(expected_numeric_hashes_path)
    roots = (original_primary_root, original_control_root)
    sources = _load_pipeline_sources(*roots, protocol_roots, inventory_path)
    transition_path = Path(transition_receipt_path).resolve()
    if (
        not transition_path.is_file()
        or sha256_file(transition_path) != transition_receipt_sha256
    ):
        raise ValueError("source-transition receipt is missing or has the wrong hash")
    _recheck_transition_evidence(transition_path, transition_receipt_sha256)
    comparison_results = [
        _pair_result(pair, transition_path, transition_receipt_sha256)
        for pair in sources.pair_specs
    ]
    paths = _write_numeric_artifacts(sources, comparison_results, output_dir)
    _check_numeric_parity_reference(numeric_reference, output_dir)
    ledger_path, figure_ledger = _write_pipeline_figures(
        sources, comparison_results, output_dir
    )
    paths["figure_ledger"] = ledger_path
    _recheck_pipeline_inputs(sources, roots, inventory_path)
    _recheck_transition_evidence(transition_path, transition_receipt_sha256)
    if _report_source_hashes() != sources.report_sources:
        raise RuntimeError(
            "reporting or figure source changed during output generation"
        )
    paths["pipeline_receipt"] = _write_pipeline_receipt(
        sources,
        roots,
        inventory_path,
        output_dir,
        (transition_path, transition_receipt_sha256),
        comparison_results,
        figure_ledger,
        numeric_reference,
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-primary", type=Path, required=True)
    parser.add_argument("--original-control", type=Path, required=True)
    parser.add_argument("--pw-primary-face-bounds", type=Path, required=True)
    parser.add_argument("--pw-control-face-bounds", type=Path, required=True)
    parser.add_argument("--driver-control-speed-bounds", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path, required=True)
    parser.add_argument("--transition-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-numeric-hashes", type=Path)
    args = parser.parse_args(argv)
    paths = build_final_supplement_report(
        original_primary_root=args.original_primary,
        original_control_root=args.original_control,
        protocol_roots={
            "pw-primary-face-bounds/1": args.pw_primary_face_bounds,
            "pw-control-face-bounds/1": args.pw_control_face_bounds,
            "driver-control-speed-bounds/1": args.driver_control_speed_bounds,
        },
        inventory_path=args.inventory,
        output_dir=args.output_dir,
        transition_receipt_path=args.transition_receipt,
        transition_receipt_sha256=args.transition_sha256,
        expected_numeric_hashes_path=args.expected_numeric_hashes,
    )
    for label, path in paths.items():
        sys.stdout.write(f"{label}: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
