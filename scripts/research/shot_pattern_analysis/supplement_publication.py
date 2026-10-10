"""Explicit density-layout child publications; parent pipeline evidence is immutable.

The injected parent verifier must run the original public verifier under its
pinned source/runtime. This module never substitutes a relaxed parent validator.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import Any

ARTIFACTS = {
    "report": "REPORT_SUPPLEMENT.json",
    "inventory": "REPORT_SUPPLEMENT.csv",
    "control_aliases": "control_aliases.json",
    "comparisons_json": "REPORT_SUPPLEMENT_COMPARISONS.json",
    "comparisons_csv": "REPORT_SUPPLEMENT_COMPARISONS.csv",
    "comparison_markdown": "REPORT_SUPPLEMENT_COMPARISONS.md",
    "original_source_inventory": "REPORT_ORIGINAL_SOURCE_INVENTORY.csv",
    "figure_ledger": "REPORT_SUPPLEMENT_FIGURES.json",
}
PARENT_RECEIPT = "REPORT_SUPPLEMENT_PIPELINE.json"
REVISION_RECEIPT = "REPORT_SUPPLEMENT_PUBLICATION_REVISION.json"
CANDIDATE_RECEIPT = "REPORT_SUPPLEMENT_PUBLICATION_CANDIDATE.json"
OPERATION = "academic_pair_density_row_gutter"
Verifier = Callable[[Path], dict[str, Any]]
Renderer = Callable[[list[dict[str, Any]], Path], list[dict[str, Any]]]


def sha256(path: Path) -> str:
    """Hash bytes without loading large graphics into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def revision_source_hashes() -> dict[str, str]:
    """Bind this contract and the actual renderer, including density preparation."""
    root = Path(__file__).resolve().parents[3]
    paths = [
        Path(__file__),
        root / "src/tools/shot_pattern_analysis/matched_supplement_figures.py",
    ]
    verifier = Path(__file__).with_name("supplement_parent_verifier.py")
    if verifier.exists():
        paths.append(verifier)
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        _report_source_hashes,
    )

    return {**_report_source_hashes(), **{p.name: sha256(p) for p in paths}}


def safe_file(root: Path, relative: str) -> Path:
    """Require a regular, non-symlink descendant with a portable relative name."""
    p = Path(relative)
    if (
        not relative
        or p.is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in p.parts
    ):
        raise ValueError("unsafe publication path")
    target = root / p
    if any(
        part.is_symlink() for part in [target, *target.parents] if part != root.parent
    ):
        raise ValueError("symlink publication path")
    if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("missing publication file")
    return target


def figure_records(ledger: dict) -> dict[str, dict]:
    if ledger.get("schema") != "matched-supplement-figure-ledger/1":
        raise ValueError("wrong figure ledger schema")
    records = {}
    for row in ledger["figures"]:
        if row.get("status") == "unavailable":
            if not row.get("reason") or "relative_path" in row or "sha256" in row:
                raise ValueError("unavailable figure claims artifact")
            continue
        name = row.get("relative_path")
        if not isinstance(name, str) or name in records:
            raise ValueError("missing or duplicate figure identity")
        records[name] = row
    return records


def revised_paths(records: dict[str, dict]) -> set[str]:
    """Closed scope: only academic pair density in the three existing formats."""
    return {
        name
        for name, row in records.items()
        if row.get("presentation") == "academic"
        and row.get("source_kind") == "six_population_common_index_comparison"
        and row.get("figure_kind") in {"density_png", "density_pdf", "density_svg"}
        and Path(name).suffix in {".png", ".pdf", ".svg"}
    }


def _parent_snapshot(root: Path, loaded: dict) -> dict[str, str]:
    receipt = loaded["receipt"]
    if (
        receipt.get("schema") != "matched-supplement-pipeline-receipt/1"
        or receipt.get("stage") != "final"
    ):
        raise ValueError("parent is not a final pipeline")
    if json.loads(safe_file(root, PARENT_RECEIPT).read_text()) != receipt:
        raise ValueError("parent verifier returned a different receipt")
    if (
        json.loads(safe_file(root, ARTIFACTS["figure_ledger"]).read_text())
        != loaded["figure_ledger"]
    ):
        raise ValueError("parent verifier returned a different ledger")
    hashes = receipt["artifact_sha256"]
    if set(hashes) != set(ARTIFACTS):
        raise ValueError("parent artifact inventory differs")
    result = {PARENT_RECEIPT: sha256(safe_file(root, PARENT_RECEIPT))}
    for key, name in ARTIFACTS.items():
        value = sha256(safe_file(root, name))
        if value != hashes[key]:
            raise ValueError("parent artifact changed")
        result[name] = value
    for name, row in figure_records(loaded["figure_ledger"]).items():
        value = sha256(safe_file(root / "figures", name))
        if value != row["sha256"]:
            raise ValueError("parent graphic changed")
        result["figures/" + name] = value
    return result


def _unchanged(root: Path, snapshot: dict[str, str]) -> None:
    for name, value in snapshot.items():
        if sha256(safe_file(root, name)) != value:
            raise ValueError("parent changed during revision")


def _validate_child(child: Path, parent: Path, loaded: dict, receipt: dict) -> dict:
    if (
        receipt.get("schema") != "matched-supplement-publication-revision/1"
        or receipt.get("operation") != OPERATION
    ):
        raise ValueError("unsupported publication revision")
    if receipt["revision_source_sha256"] != revision_source_hashes():
        raise ValueError("revision producer source changed")
    old_sources = loaded["receipt"].get("postprocessing_source_sha256", {})
    current_sources = revision_source_hashes()
    if not old_sources or any(
        current_sources.get(name) != digest
        for name, digest in old_sources.items()
        if name != "matched_supplement_figures.py"
    ):
        raise ValueError("revision changed a protected reporting source")
    if sha256(safe_file(parent, PARENT_RECEIPT)) != receipt["parent_receipt_sha256"]:
        raise ValueError("parent receipt differs")
    parent_rows = figure_records(loaded["figure_ledger"])
    expected = revised_paths(parent_rows)
    hashes = receipt["child_artifact_sha256"]
    if set(hashes) != set(ARTIFACTS):
        raise ValueError("child artifact inventory differs")
    for key, name in ARTIFACTS.items():
        if sha256(safe_file(child, name)) != hashes[key]:
            raise ValueError("child report artifact changed")
        if (
            key != "figure_ledger"
            and hashes[key] != loaded["receipt"]["artifact_sha256"][key]
        ):
            raise ValueError("numerical report bytes changed")
    ledger = json.loads(safe_file(child, ARTIFACTS["figure_ledger"]).read_text())
    rows = figure_records(ledger)
    if set(rows) != set(parent_rows) or set(receipt["origins"]) != set(rows):
        raise ValueError("child figure inventory differs")
    if {
        p.relative_to(child / "figures").as_posix()
        for p in (child / "figures").rglob("*")
        if p.is_file()
    } != set(rows):
        raise ValueError("unlisted figure output")
    transformed = []
    for original in loaded["figure_ledger"]["figures"]:
        if original.get("status") == "unavailable":
            transformed.append(original)
            continue
        name = original["relative_path"]
        row = rows[name]
        if {k: v for k, v in row.items() if k != "sha256"} != {
            k: v for k, v in original.items() if k != "sha256"
        }:
            raise ValueError("child figure semantic identity differs")
        actual = sha256(safe_file(child / "figures", name))
        operation = "rerendered" if name in expected else "reused"
        if receipt["origins"][name] != {
            "operation": operation,
            "parent_sha256": original["sha256"],
            "child_sha256": actual,
        }:
            raise ValueError("figure origin differs")
        if actual != row["sha256"] or (
            operation == "reused" and actual != original["sha256"]
        ):
            raise ValueError("graphic bytes changed")
        transformed.append(row)
    if ledger != {**loaded["figure_ledger"], "figures": transformed}:
        raise ValueError("ledger metadata changed")
    return ledger


def create_density_revision(
    parent: Path,
    child: Path,
    *,
    parent_verifier: Verifier,
    render_density: Renderer,
    verifier_binding: dict | None = None,
) -> Path:
    """Stage graphics and a candidate receipt; independent review publishes later."""
    parent = Path(parent).resolve()
    child = Path(child).resolve()
    if child.exists() or child.is_relative_to(parent) or parent.is_relative_to(child):
        raise ValueError("child must be a new disjoint directory")
    sources = revision_source_hashes()
    loaded = parent_verifier(parent)
    snapshot = _parent_snapshot(parent, loaded)
    rows = figure_records(loaded["figure_ledger"])
    expected = revised_paths(rows)
    if not expected:
        raise ValueError("parent has no academic pair density outputs")
    child.mkdir(parents=True)
    for name in ARTIFACTS.values():
        shutil.copyfile(safe_file(parent, name), child / name)
    for name in rows:
        if name not in expected:
            out = child / "figures" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(parent / "figures", name), out)
    updates = render_density([rows[n] for n in sorted(expected)], child)
    changed = figure_records(
        {"schema": "matched-supplement-figure-ledger/1", "figures": updates}
    )
    if set(changed) != expected:
        raise ValueError("renderer returned wrong output scope")
    ledger = {
        **loaded["figure_ledger"],
        "figures": [
            changed.get(r.get("relative_path"), r)
            for r in loaded["figure_ledger"]["figures"]
        ],
    }
    (child / ARTIFACTS["figure_ledger"]).write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "schema": "matched-supplement-publication-revision/1",
        "operation": OPERATION,
        "parent_relative_path": os.path.relpath(parent, child),
        "parent_receipt_sha256": snapshot[PARENT_RECEIPT],
        "parent_verifier": verifier_binding,
        "revision_source_sha256": sources,
        "child_artifact_sha256": {k: sha256(child / n) for k, n in ARTIFACTS.items()},
        "origins": {
            n: {
                "operation": "rerendered" if n in expected else "reused",
                "parent_sha256": r["sha256"],
                "child_sha256": sha256(safe_file(child / "figures", n)),
            }
            for n, r in rows.items()
        },
    }
    _validate_child(child, parent, loaded, receipt)
    _unchanged(parent, snapshot)
    parent_verifier(parent)
    _unchanged(parent, snapshot)
    if sources != revision_source_hashes():
        raise ValueError("revision source changed during rendering")
    _validate_child(child, parent, loaded, receipt)
    temporary = child / (CANDIDATE_RECEIPT + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(child / CANDIDATE_RECEIPT)
    return child / CANDIDATE_RECEIPT


def verify_density_revision(child: Path, *, parent_verifier: Verifier) -> dict:
    child = Path(child).resolve()
    path = safe_file(child, REVISION_RECEIPT)
    before = sha256(path)
    receipt = json.loads(path.read_text())
    parent = (child / receipt["parent_relative_path"]).resolve()
    if parent == child:
        raise ValueError("self-referencing publication")
    loaded = parent_verifier(parent)
    snapshot = _parent_snapshot(parent, loaded)
    ledger = _validate_child(child, parent, loaded, receipt)
    _verify_qualification(child, loaded, receipt)
    parent_verifier(parent)
    _unchanged(parent, snapshot)
    _validate_child(child, parent, loaded, receipt)
    _verify_qualification(child, loaded, receipt)
    if sha256(path) != before:
        raise ValueError("child receipt changed during verification")
    return {
        **loaded,
        "figure_ledger": ledger,
        "publication_identity": {
            "kind": OPERATION,
            "receipt_filename": REVISION_RECEIPT,
            "receipt_sha256": before,
            "parent_receipt_sha256": receipt["parent_receipt_sha256"],
        },
    }


def load_verified_supplement_publication(directory: Path) -> dict:
    """Dispatch only explicit receipt types; never recover from a failed parent."""
    directory = Path(directory).resolve()
    revision = directory / REVISION_RECEIPT
    parent = directory / PARENT_RECEIPT
    if revision.is_file():
        if parent.exists():
            raise ValueError("ambiguous parent and child publication")
        from .supplement_parent_verifier import verify_parent_in_frozen_source

        binding = json.loads(revision.read_text()).get("parent_verifier")
        if not isinstance(binding, dict):
            raise ValueError("child has no pinned parent verifier")
        return verify_density_revision(
            directory,
            parent_verifier=lambda path: verify_parent_in_frozen_source(
                path, binding, base=directory
            ),
        )
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        load_verified_supplement_artifacts,
    )

    before = sha256(safe_file(directory, PARENT_RECEIPT))
    loaded = load_verified_supplement_artifacts(directory)
    if sha256(parent) != before:
        raise RuntimeError("receipt changed during supplement loading")
    return {
        **loaded,
        "publication_identity": {
            "kind": "pipeline",
            "receipt_filename": PARENT_RECEIPT,
            "receipt_sha256": before,
            "parent_receipt_sha256": before,
        },
    }


def approved_source_delta(loaded: dict, candidate: dict) -> dict:
    """Derive the actual old/new producer delta instead of trusting a claim."""
    old = loaded["receipt"]["postprocessing_source_sha256"]
    new = candidate["revision_source_sha256"]
    return {
        name: {"old_sha256": digest, "new_sha256": new[name]}
        for name, digest in old.items()
        if new.get(name) != digest
    }


def _check_review(
    child: Path, loaded: dict, candidate: dict, review_path: Path
) -> dict:
    review = json.loads(review_path.read_text())
    expected = {
        name: row["child_sha256"]
        for name, row in candidate["origins"].items()
        if row["operation"] == "rerendered"
    }
    if (
        review.get("schema") != "matched-supplement-publication-review/1"
        or review.get("accepted") is not True
        or review.get("review_scope") != "visual_and_contract"
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or review.get("candidate_sha256") != sha256(safe_file(child, CANDIDATE_RECEIPT))
        or review.get("source_delta") != approved_source_delta(loaded, candidate)
        or review.get("revised_asset_sha256") != expected
    ):
        raise ValueError(
            "independent publication review is missing or does not bind this candidate"
        )
    evidence = review.get("evidence_files")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("independent review has no bound evidence")
    for name, digest in evidence.items():
        if sha256(safe_file(review_path.parent, name)) != digest:
            raise ValueError("independent review evidence differs")
    return review


def _verify_qualification(child: Path, loaded: dict, final: dict) -> None:
    qualification = final.get("qualification")
    if not isinstance(qualification, dict):
        raise ValueError("publication lacks independent qualification")
    candidate_path = safe_file(child, CANDIDATE_RECEIPT)
    if sha256(candidate_path) != qualification.get("candidate_sha256"):
        raise ValueError("qualified candidate receipt changed")
    candidate = json.loads(candidate_path.read_text())
    if {k: v for k, v in final.items() if k != "qualification"} != candidate:
        raise ValueError("publication differs from reviewed candidate")
    review_path = safe_file(child, qualification["review_path"])
    if sha256(review_path) != qualification["review_sha256"]:
        raise ValueError("independent review changed")
    _check_review(child, loaded, candidate, review_path)


def finalize_density_revision(
    child: Path, *, review_path: Path, review_sha256: str, parent_verifier: Verifier
) -> Path:
    """Publish only a separately reviewed staged candidate, preserving review bytes."""
    child = Path(child).resolve()
    review_path = Path(review_path).resolve()
    if (child / REVISION_RECEIPT).exists():
        raise FileExistsError("publication already finalized")
    candidate_path = safe_file(child, CANDIDATE_RECEIPT)
    candidate_sha = sha256(candidate_path)
    candidate = json.loads(candidate_path.read_text())
    parent = (child / candidate["parent_relative_path"]).resolve()
    if sha256(review_path) != review_sha256:
        raise ValueError("review identity differs")
    loaded = parent_verifier(parent)
    snapshot = _parent_snapshot(parent, loaded)
    _validate_child(child, parent, loaded, candidate)
    review = _check_review(child, loaded, candidate, review_path)
    destination = child / "review"
    destination.mkdir()
    if "independent_review.json" in review["evidence_files"]:
        raise ValueError("review evidence name collides with review record")
    for name in review["evidence_files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(review_path.parent, name), target)
    shutil.copyfile(review_path, destination / "independent_review.json")
    final = {
        **candidate,
        "qualification": {
            "candidate_sha256": candidate_sha,
            "review_path": "review/independent_review.json",
            "review_sha256": review_sha256,
        },
    }
    parent_verifier(parent)
    _unchanged(parent, snapshot)
    _validate_child(child, parent, loaded, final)
    _verify_qualification(child, loaded, final)
    temporary = child / (REVISION_RECEIPT + ".tmp")
    temporary.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    temporary.replace(child / REVISION_RECEIPT)
    return child / REVISION_RECEIPT
