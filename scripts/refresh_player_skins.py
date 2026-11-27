"""Refresh extracted player appearances without rebuilding their physical specs.

Example: ``python -m scripts.refresh_player_skins --root output/sized``.
Extract a previous verified delivery first. Review-pose coordinates remain
unchanged; only the target XML hash is rebound after the visual-only export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.model_appearance.player_models import (
    player_appearance,
    player_profile,
)
from src.shared.python.model_appearance.schema import (
    document_to_dict,
    physics_spec_sha256,
    save_appearance,
)


def refresh_player(folder: Path) -> Path:
    """Replace visual exports, retaining exact physical JSON and verified pose q.

    Require an existing model, catalog ID and, when present, a pose bound to
    that physical spec and old XML. Reject stale poses before any writes.
    """
    from scripts.export_player_models import _validated_review_pose

    raw = (folder / "model.json").read_bytes()
    spec = json.loads(raw)
    pose_path = folder / "review_pose.json"
    pose = json.loads(pose_path.read_text()) if pose_path.exists() else None
    if pose is not None:
        _validated_review_pose(pose, spec, (folder / "model.xml").read_bytes())
    receipt_path = folder / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be a JSON object")
    doc = player_appearance(folder.name)
    xml, metadata = export_full_body_mjcf(raw, appearance=doc)
    target = folder / "model.xml"
    target.write_text(xml, encoding="utf-8")
    save_appearance(doc, folder / "model.appearance.json")
    (folder / "model.visual.json").write_text(
        json.dumps({**spec, "appearance": document_to_dict(doc)}, indent=2) + "\n"
    )
    receipt.update(
        profile=player_profile(folder.name),
        export=metadata,
        physics_spec_sha256=physics_spec_sha256(spec),
        appearance_revision="portrait-reference-2026-11-27",
        physics_json_bytes_preserved=True,
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if pose is not None:
        pose["target_model_xml_sha256"] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
        pose_path.write_text(json.dumps(pose, indent=2) + "\n")
    return target


def main() -> int:
    """Refresh each existing player folder under one extracted variant root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    folders = sorted(p.parent for p in args.root.glob("*/model.json"))
    if not folders:
        parser.error("root must contain extracted player/model.json files")
    for folder in folders:
        refresh_player(folder)
        print(folder.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
