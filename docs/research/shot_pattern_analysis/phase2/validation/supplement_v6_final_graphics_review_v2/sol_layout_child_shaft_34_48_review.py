"""Bind Sol's actual 15-image selective child PNG inspection."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONTACTS = HERE / "layout_child_png_contacts"
OUT = HERE / "sol_layout_child_shaft_34_48_review.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest_path = CONTACTS / "inventory.json"
    supplement_path = CONTACTS / "colorbar_native_inventory.json"
    manifest = json.loads(manifest_path.read_text())
    supplement = json.loads(supplement_path.read_text())
    items = manifest["contacts"][34:49]
    if [x["index"] for x in items] != list(range(34, 49)) or any(x["reviewer"] != "sol_shaft" for x in items):
        raise ValueError("review scope changed")
    native = {item["index"]: item for item in supplement["entries"]}
    decisions = []
    for item in items:
        for key in ("baseline_png", "child_png", "contact"):
            if sha(Path(item[key])) != item[key + "_sha256"] if key != "contact" else sha(Path(item[key])) != item["contact_sha256"]:
                raise ValueError(f"changed {key} at {item['index']}")
        index = item["index"]
        extra = None
        if index in (39, 40, 43, 44):
            extra = native[index]
            if sha(Path(extra["contact"])) != extra["sha256"]:
                raise ValueError(f"changed native crop at {index}")
            if extra["child_png_sha256"] != item["child_png_sha256"]:
                raise ValueError(f"native crop identity mismatch at {index}")
        decisions.append(
            {
                "index": index,
                "child_png": item["child_png"],
                "child_png_sha256": item["child_png_sha256"],
                "baseline_png_sha256": item["baseline_png_sha256"],
                "contact": item["contact"],
                "contact_sha256": item["contact_sha256"],
                "native_right_edge_contact": extra["contact"] if extra else None,
                "native_right_edge_sha256": extra["sha256"] if extra else None,
                "full_size_child_inspected": index in (34, 39, 45),
                "decision": "accepted_within_selective_child_png_scope",
                "observation": "Condition, target, density, axis and scale labels remain legible; legend/panel titles separate. Where measured, lower Fade title clears the colorbar. No new clipping or overlap seen.",
            }
        )
    # The comparisons are visual evidence; checks below only bind files and scope.
    scope_path = HERE / "density_layout_repair_scope.json"
    scope = json.loads(scope_path.read_text())
    if sha(scope_path) != manifest["scope_sha256"]:
        raise ValueError("scope binding changed")
    for index, decision in enumerate(decisions, 34):
        if Path(scope["affected"][index]["matching_png_path"]).name != Path(decision["child_png"]).name:
            raise ValueError(f"scope/child mismatch at {index}")
    receipt = {
        "schema": "independent-selective-child-png-review/1",
        "reviewer": "Sol independent reviewer",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Only selective layout-child shaft rotation SD density PNGs 34-48; 15 assets. No whole-publication or vector acceptance.",
        "accepted": True,
        "whole_publication_accepted": False,
        "counts": {"density_png_decisions": 15, "contacts_actually_inspected": 15, "native_right_edge_crops_actually_inspected": 4, "full_size_child_pngs_actually_inspected": 3},
        "manifest_sha256": sha(manifest_path),
        "colorbar_native_inventory_sha256": sha(supplement_path),
        "scope_sha256": sha(scope_path),
        "child_strict_loader_receipt_sha256": sha(HERE / "layout_child_strict_loader_receipt.json"),
        "child_reconciliation_receipt_sha256": sha(HERE / "layout_child_reconciliation_receipt.json"),
        "child_clearance_receipt_sha256": sha(HERE / "layout_child_clearance_receipt.json"),
        "per_asset": decisions,
        "limitations": ["Contacts include reduced whole images plus native-pixel crops; three original child PNGs were opened at full size.", "PDF and SVG graphics are outside this PNG decision.", "The geometric gates and loader are separate from visual review."],
    }
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(OUT), "sha256": sha(OUT), "accepted_pngs": 15}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
