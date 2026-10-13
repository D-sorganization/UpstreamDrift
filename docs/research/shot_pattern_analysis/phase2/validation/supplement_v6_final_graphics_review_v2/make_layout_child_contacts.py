"""Make bound before/after contacts for independent density PNG inspection."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
CHILD = PHASE / "supplement_report_v2_layout_repair"
OUT = HERE / "layout_child_png_contacts"
REVIEWERS = ((0, 19, "root_curve"), (19, 34, "astra_shaft"), (34, 49, "sol_shaft"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    scope_path = HERE / "density_layout_repair_scope.json"
    scope = json.loads(scope_path.read_text())["affected"]
    assert len(scope) == 49
    OUT.mkdir(exist_ok=True)
    inventory = []
    for index, entry in enumerate(scope):
        original = Path(entry["matching_png_path"])
        child = CHILD / "figures/comparisons" / original.name
        if not original.is_file() or not child.is_file():
            raise FileNotFoundError((original, child))
        if sha(Path(entry["svg_path"])) != entry["baseline_svg_sha256"]:
            raise ValueError(f"baseline drift at index {index}")
        with Image.open(original) as im:
            before = im.convert("RGB")
        with Image.open(child) as im:
            after = im.convert("RGB")
        if (
            min(before.width, after.width) < 4000
            or min(before.height, after.height) < 2400
        ):
            raise ValueError(f"unexpected image geometry at index {index}")
        # Preserve source images; these are marked review contacts only.
        board = Image.new("RGB", (2300, 2240), "white")
        draw = ImageDraw.Draw(board)
        draw.text((20, 8), f"{index:02d}  {original.name}", fill="black")
        draw.text((20, 36), "Baseline", fill="black")
        draw.text((1160, 36), "Layout Child", fill="black")
        for x, image in ((20, before), (1160, after)):
            board.paste(image.resize((1100, 632)), (x, 62))
            # Native-pixel crops expose legend/title and Fade/colorbar clearance.
            board.paste(image.crop((0, 0, 1100, 700)), (x, 710))
            right_start = int(image.width * 0.57)
            board.paste(
                image.crop((right_start, 940, right_start + 1100, 1740)), (x, 1420)
            )
        reviewer = next(
            name for start, stop, name in REVIEWERS if start <= index < stop
        )
        contact = OUT / f"{index:02d}_{reviewer}.jpg"
        board.save(contact, quality=94, subsampling=0)
        inventory.append(
            {
                "index": index,
                "reviewer": reviewer,
                "failed_gates": entry["failed_gates"],
                "baseline_png": str(original),
                "baseline_png_sha256": sha(original),
                "child_png": str(child),
                "child_png_sha256": sha(child),
                "baseline_svg_sha256": entry["baseline_svg_sha256"],
                "contact": str(contact),
                "contact_sha256": sha(contact),
                "contact_description": "reduced full images, native-pixel upper-left and lower-right crops",
            }
        )
    manifest = {
        "schema": "shot-pattern-layout-child-png-contacts/1",
        "scope_sha256": sha(scope_path),
        "contact_count": len(inventory),
        "reviewer_ranges": [
            {"start": start, "stop_exclusive": stop, "reviewer": name}
            for start, stop, name in REVIEWERS
        ],
        "contacts": inventory,
    }
    target = OUT / "inventory.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(
        json.dumps(
            {
                "inventory": str(target),
                "sha256": sha(target),
                "contact_count": len(inventory),
            }
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
