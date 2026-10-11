"""Add native-pixel right-edge comparisons for six colorbar clearance cases."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "layout_child_png_contacts"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    inventory = json.loads((OUT / "inventory.json").read_text())["contacts"]
    selected = []
    for item in inventory:
        if "all_density_colorbar_diagnostic.json" not in item["failed_gates"]:
            continue
        board = Image.new("RGB", (2240, 910), "white")
        draw = ImageDraw.Draw(board)
        draw.text(
            (20, 10),
            f"{item['index']:02d} lower Fade title and colorbar: baseline | child",
            fill="black",
        )
        for x, key in ((20, "baseline_png"), (1130, "child_png")):
            with Image.open(item[key]) as source:
                image = source.convert("RGB")
            board.paste(
                image.crop((image.width - 1100, 1050, image.width, 1850)), (x, 70)
            )
        target = OUT / f"{item['index']:02d}_{item['reviewer']}_colorbar_native.jpg"
        board.save(target, quality=95, subsampling=0)
        selected.append(
            {
                "index": item["index"],
                "reviewer": item["reviewer"],
                "contact": str(target),
                "sha256": sha(target),
                "baseline_png_sha256": item["baseline_png_sha256"],
                "child_png_sha256": item["child_png_sha256"],
            }
        )
    if len(selected) != 6:
        raise ValueError("expected six measured colorbar cases")
    target = OUT / "colorbar_native_inventory.json"
    target.write_text(
        json.dumps(
            {
                "schema": "shot-pattern-layout-child-colorbar-crops/1",
                "entries": selected,
            },
            indent=2,
        )
        + "\n"
    )
    sys.stdout.write(
        json.dumps(
            {"inventory": str(target), "sha256": sha(target), "count": len(selected)}
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
