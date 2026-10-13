"""Map prior scoped PNG reviews onto exact v6 bytes and scientific identities."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


phase = Path(__file__).resolve().parents[2]
validation = phase / "validation"
output = Path(__file__).resolve().parent
report = phase / "supplement_report_v1"
ledger = json.loads((report / "REPORT_SUPPLEMENT_FIGURES.json").read_text())["figures"]
comparisons = json.loads((report / "REPORT_SUPPLEMENT_COMPARISONS.json").read_text())[
    "comparisons"
]
cells = json.loads((report / "REPORT_SUPPLEMENT.json").read_text())["unique_cells"]


def pair_key(kind, sources, target, common):
    return (
        kind,
        tuple((s["protocol_id"], s["cell"], s["source_csv_sha256"]) for s in sources),
        target,
        common,
    )


new_pairs = {
    pair_key(
        r["comparison_kind"], r["sources"], r["target_x_m"], r["common_supported_n"]
    ): r
    for r in comparisons
}
new_by_simple = {}
for key in new_pairs:
    kind, sources, target, common = key
    simple = (kind, sources[0][:2], sources[1][:2], target, common)
    if simple in new_by_simple:
        raise ValueError("new pair scientific identity collision")
    new_by_simple[simple] = key

cell_dir = validation / "supplement_v5_cell_graphics_review_v1"
cell_review = json.loads((cell_dir / "review.json").read_text())
cell_checks_path = cell_dir / "cell_checks.json"
cell_checks = json.loads(cell_checks_path.read_text())
if (
    not cell_review["accepted"]
    or sha(cell_checks_path) != cell_review["cell_checks_sha256"]
):
    raise ValueError("old cell acceptance binding failed")
old_cells = {
    (r["protocol"], r["cell"], r["target_x_m"], r["source_summary_sha256"])
    for r in cell_checks["cells"]
}
new_cells = {
    (
        r["source_protocol"],
        r["cell"],
        r["summary"]["target_x_m"],
        r["source_summary_sha256"],
    )
    for r in cells
}
if old_cells != new_cells:
    raise ValueError("cell scientific identities differ")

old_pair_sha = {}
pair_receipts = {}
for directory, kind, allowed in (
    (
        "supplement_v5_across_lean_graphics_review_v1",
        "across_lean",
        {
            "density_straight_social.png",
            "density_draw_social.png",
            "density_fade_social.png",
            "overhead.png",
            "overhead_social.png",
        },
    ),
    (
        "supplement_v5_curve_graphics_first12_v1",
        "curve_strength",
        {
            "density_straight_social.png",
            "density_draw_social.png",
            "density_fade_social.png",
        },
    ),
):
    prior = validation / directory
    review_path = prior / "review.json"
    review = json.loads(review_path.read_text())
    inventory_path = prior / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    expected_inventory = review.get("inventory_sha256")
    if expected_inventory != sha(inventory_path):
        raise ValueError(f"prior inventory binding failed: {directory}")
    pair_receipts[kind] = {
        "review": str(review_path),
        "review_sha256": sha(review_path),
        "inventory": str(inventory_path),
        "inventory_sha256": sha(inventory_path),
    }
    for old in inventory["pairs"]:
        sources = old.get("source_cells", old.get("sources"))
        key = pair_key(
            kind,
            sources,
            old["target_x_m"],
            old.get("common_six_n", old.get("common_supported_n")),
        )
        if key not in new_pairs:
            continue
        prefix = old.get("prefix")
        if prefix is None:
            # Historical curve paths use an old cell-only stem. Strip a known suffix.
            prefix = next(
                name.rsplit("_density.png", 1)[0]
                for name in old["assets_sha256"]
                if name.endswith("_density.png")
            )
        for name, digest in old["assets_sha256"].items():
            if not name.endswith(".png"):
                continue
            suffix = name[len(prefix) + 1 :]
            if suffix in allowed:
                old_pair_sha[key, suffix] = digest

suffixes = (
    "density_straight_social.png",
    "density_draw_social.png",
    "density_fade_social.png",
    "density.png",
    "overhead_social.png",
    "overhead.png",
)
reuse = []
for row in ledger:
    path = row["relative_path"]
    if not path.endswith(".png"):
        continue
    digest = row["sha256"]
    if path.startswith("cells/"):
        if digest == cell_checks["asset_sha256"].get(Path(path).name):
            reuse.append(
                {
                    "path": path,
                    "sha256": digest,
                    "scientific_identity": [
                        row["source_protocol"],
                        row["cell"],
                        row["target_x_m"],
                    ],
                    "artifact_kind": row["figure_kind"],
                    "prior_review": str(cell_dir / "review.json"),
                    "prior_review_sha256": sha(cell_dir / "review.json"),
                    "prior_inventory": str(cell_checks_path),
                    "prior_inventory_sha256": sha(cell_checks_path),
                    "prior_scope": "accepted individual cell PNG",
                }
            )
        continue
    if not path.startswith("comparisons/"):
        continue
    common = row.get("common_supported_n")
    if common is None:
        matching = [
            k
            for k in new_by_simple
            if k[:4]
            == (
                row["comparison_kind"],
                (row["first_protocol"], row["first_cell"]),
                (row["second_protocol"], row["second_cell"]),
                row["target_x_m"],
            )
        ]
        if len(matching) != 1:
            raise ValueError("ledger pair identity did not resolve uniquely")
        common = matching[0][4]
    simple = (
        row["comparison_kind"],
        (row["first_protocol"], row["first_cell"]),
        (row["second_protocol"], row["second_cell"]),
        row["target_x_m"],
        common,
    )
    key = new_by_simple[simple]
    suffix = next((suffix for suffix in suffixes if path.endswith("_" + suffix)), None)
    if suffix and digest == old_pair_sha.get((key, suffix)):
        prior_info = pair_receipts[row["comparison_kind"]]
        reuse.append(
            {
                "path": path,
                "sha256": digest,
                "scientific_identity": [
                    key[0],
                    [list(s) for s in key[1]],
                    key[2],
                    key[3],
                ],
                "artifact_kind": row["figure_kind"],
                "prior_review": prior_info["review"],
                "prior_review_sha256": prior_info["review_sha256"],
                "prior_inventory": prior_info["inventory"],
                "prior_inventory_sha256": prior_info["inventory_sha256"],
                "prior_scope": "previously visually inspected subkind; parent publication rejected"
                if key[0] == "curve_strength"
                else "previously accepted social/overhead subkind; academic density rejected",
            }
        )

receipt = {
    "scope": "Exact-Byte PNG Reuse Candidates; Final Visual Adjudication Pending",
    "ledger_sha256": sha(report / "REPORT_SUPPLEMENT_FIGURES.json"),
    "candidate_count": len(reuse),
    "counts_by_prior_scope": dict(Counter(row["prior_scope"] for row in reuse)),
    "candidates": reuse,
}
(output / "png_reuse_candidates.json").write_text(json.dumps(receipt, indent=2) + "\n")
sys.stdout.write(
    json.dumps({"candidates": len(reuse), "counts": receipt["counts_by_prior_scope"]})
    + "\n"
)
