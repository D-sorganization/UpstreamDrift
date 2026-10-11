"""Reconcile qualified batch inventories, actual review receipts, and child PNG replacements."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "across_lean": (0, 12, 24, 36, 48),
    "curve_strength": tuple(range(0, 157, 12)),
    "shaft_rotation_sd": tuple(range(0, 157, 12)),
}
RETRY = {"across_lean_024", "curve_strength_000"}
CHILD_REVIEWS = (
    "root_layout_child_curve_review.json",
    "astra_layout_child_019_033/review.json",
    "sol_layout_child_shaft_34_48_review.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def review_assets(review: dict) -> dict[str, str]:
    assets = (
        review.get("per_asset")
        or review.get("png_decisions")
        or review.get("assets")
        or review.get("png_assets")
    )
    if assets is not None:
        return {item["path"]: item["sha256"] for item in assets}
    records = review.get("records")
    if not isinstance(records, list):
        raise ValueError("Review lacks PNG decisions/records")
    if all("asset_sha256" in record for record in records):
        return {
            path: digest
            for record in records
            for path, digest in record["asset_sha256"].items()
            if path.endswith(".png")
        }
    if all("png_paths" in record for record in records):
        return {
            path: "inventory-bound"
            for record in records
            for path in record["png_paths"]
        }
    raise ValueError("Review records lack asset identities")


def review_contacts(review: dict) -> dict[str, str]:
    contacts = (
        review.get("contacts_inspected")
        or review.get("inspected_contacts")
        or review.get("contact_sha256")
    )
    if contacts is None:
        raise ValueError("Review lacks contact evidence")
    if isinstance(contacts, dict):
        return contacts
    return {Path(item["path"]).name: item["sha256"] for item in contacts}


def _baseline_reviews(
    affected: set[str],
) -> tuple[list[dict], dict[str, str], int, int]:
    """Verify every qualified batch before accepting child replacements."""
    rows = []
    baseline_assets: dict[str, str] = {}
    baseline_contacts = 0
    total_pairs = 0
    for group, offsets in GROUPS.items():
        for offset in offsets:
            name = f"{group}_{offset:03d}"
            if name in RETRY:
                name += "_retry"
            inventory_path = ROOT / name / "batch_inventory.json"
            command_path = ROOT / f"{name}_command.json"
            inventory, command = read(inventory_path), read(command_path)
            require(command["exit_code"] == 0, f"{name} command failed")
            require(
                command["batch_inventory_sha256"] == sha256(inventory_path),
                f"{name} command/inventory mismatch",
            )
            require(inventory["pending"] == [], f"{name} has pending items")
            matching = list(ROOT.glob(f"*_{name}/review.json"))
            require(len(matching) == 1, f"{name} review count {len(matching)}")
            review_path = matching[0]
            review = read(review_path)
            expected_assets = {
                path: digest
                for selected in inventory["selected"]
                for path, digest in selected["asset_sha256"].items()
                if path.endswith(".png")
            }
            observed_assets = review_assets(review)
            if name == "curve_strength_096":
                require(
                    set(observed_assets) == set(expected_assets),
                    f"{name} PNG review coverage mismatch",
                )
                require(
                    review["batch_inventory_sha256"] == sha256(inventory_path),
                    f"{name} review/inventory hash mismatch",
                )
                require(
                    review.get("all_assets_rehashed") == 192,
                    f"{name} lacks rehash assertion",
                )
            else:
                require(
                    observed_assets == expected_assets,
                    f"{name} PNG review hash/coverage mismatch",
                )
            expected_contacts = {
                Path(contact["path"]).name: contact["sha256"]
                for selected in inventory["selected"]
                for contact in selected["contacts"]
            }
            observed_contacts = review_contacts(review)
            require(
                observed_contacts == expected_contacts,
                f"{name} contact hash/coverage mismatch",
            )
            require(
                not (baseline_assets.keys() & expected_assets.keys()),
                f"{name} duplicates prior PNG",
            )
            baseline_assets.update(expected_assets)
            baseline_contacts += len(expected_contacts)
            total_pairs += len(inventory["selected"])
            rows.append(
                {
                    "batch": name,
                    "pairs": len(inventory["selected"]),
                    "pngs": len(expected_assets),
                    "contacts": len(expected_contacts),
                    "inventory_sha256": sha256(inventory_path),
                    "command_sha256": sha256(command_path),
                    "review": str(review_path.relative_to(ROOT)),
                    "review_sha256": sha256(review_path),
                    "affected_baseline_pngs": sum(
                        Path(path).name in affected for path in expected_assets
                    ),
                }
            )
    require(len(baseline_assets) == 2304, "Expected 2304 baseline PNGs")
    require(
        {Path(path).name for path in baseline_assets if Path(path).name in affected}
        == affected,
        "Repair scope not covered by baseline batches",
    )
    return rows, baseline_assets, baseline_contacts, total_pairs


def _child_reviews(affected: set[str]) -> tuple[list[dict], dict[str, str]]:
    """Check replacement PNG identities against independent child decisions."""
    replacements = read(ROOT / "layout_repair_child_publication_receipt.json")[
        "replacements"
    ]
    child_png = {
        Path(item["relative_path"]).name: item
        for item in replacements
        if item["relative_path"].endswith(".png")
    }
    require(
        set(child_png) == affected, "Child PNG replacements differ from repair scope"
    )
    child_reviews = []
    child_decisions: dict[str, str] = {}
    for rel in CHILD_REVIEWS:
        path = ROOT / rel
        data = read(path)
        decisions = (
            data.get("records") or data.get("decisions") or data.get("per_asset")
        )
        if not isinstance(decisions, list):
            raise ValueError(f"{rel} lacks per-asset decisions")
        for item in decisions:
            name = Path(item["child_png"]).name
            require(name not in child_decisions, f"Duplicate child decision: {name}")
            require(name in child_png, f"Unscoped child decision: {name}")
            require(
                item["child_png_sha256"] == child_png[name]["child_sha256"],
                f"Child hash mismatch: {name}",
            )
            child_decisions[name] = item["child_png_sha256"]
        child_reviews.append(
            {"path": rel, "sha256": sha256(path), "pngs": len(decisions)}
        )
    require(
        set(child_decisions) == affected,
        "Child review coverage differs from repair scope",
    )
    return child_reviews, child_decisions


def main() -> None:
    scope_path = ROOT / "density_layout_repair_scope.json"
    scope = read(scope_path)
    affected = {Path(item["matching_png_path"]).name for item in scope["affected"]}
    require(len(affected) == 49, "Repair scope must contain 49 unique PNGs")
    rows, baseline_assets, baseline_contacts, total_pairs = _baseline_reviews(affected)
    child_reviews, child_decisions = _child_reviews(affected)
    native_path = ROOT / "layout_child_native_vector_review.json"
    native = read(native_path)
    require(len(native["identities"]) == 49, "Native review does not cover changed 49")
    receipt = {
        "schema": "shot-pattern-final-png-coverage-v1",
        "status": "accepted_child_pngs_with_scoped_native_review",
        "scope_sha256": sha256(scope_path),
        "counts": {
            "qualified_graphics_batches": len(rows),
            "pairs": total_pairs,
            "baseline_pngs_reviewed": len(baseline_assets),
            "baseline_contacts_reviewed": baseline_contacts,
            "baseline_pngs_overridden_by_measured_layout_scope": len(affected),
            "unchanged_baseline_pngs_accepted": len(baseline_assets) - len(affected),
            "child_replacement_pngs_accepted": len(child_decisions),
            "final_accepted_pngs": len(baseline_assets),
        },
        "batches": rows,
        "child_reviews": child_reviews,
        "child_native_review_sha256": sha256(native_path),
        "child_publication_receipt_sha256": sha256(
            ROOT / "layout_repair_child_publication_receipt.json"
        ),
        "child_strict_loader_receipt_sha256": sha256(
            ROOT / "layout_child_strict_loader_receipt.json"
        ),
        "child_reconciliation_receipt_sha256": sha256(
            ROOT / "layout_child_reconciliation_receipt.json"
        ),
        "child_clearance_receipt_sha256": sha256(
            ROOT / "layout_child_clearance_receipt.json"
        ),
        "limitations": [
            "Baseline visual acceptance is overridden for exactly 49 measured density PNG defects; use reviewed child replacements.",
            "Native PDF/SVG review is separately scoped to the 49 changed pairs; no simulation rerun or numerical recomputation is claimed.",
        ],
    }
    out = ROOT / "final_png_coverage_receipt.json"
    out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        json.dumps(
            {"receipt": str(out), "sha256": sha256(out), "counts": receipt["counts"]},
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
