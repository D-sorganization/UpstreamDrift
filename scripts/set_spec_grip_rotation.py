"""Set a committed full-body spec document's fitted hand rotation (#12221).

The hand-to-club rotation of each wrist depends on the capture, so it is
refitted per capture (``evidence/anthropometry/fit_grip_rotation.py``, one
invocation per capture with its own ``--receipt``). ``apply`` reads the
receipt's ``sides.lead.total_deg`` and ``sides.trail.total_deg`` and sets the
document's ``subject.grip_rotation_deg`` (``L``, ``R``) and the wrist
``child_to_follower`` to them, idempotently, appending a provenance note.
Committed documents carry hand-tuned contact parameters, so they are not
rebuilt.

Usage::

    python3 -m scripts.set_spec_grip_rotation apply \
        docs/development/full_body_models/evidence/anthropometry/fit_driver.json \
        docs/development/full_body_models/full_body_spec_anthro_driver.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from scripts.mirror_spec_left_hip import format_json
from src.shared.python.motion_matching.execution.grip_rotation import (
    apply_grip_rotation,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
SIDE_KEYS = {"lead": "L", "trail": "R"}


def _label(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return path.name


def read_totals(receipt_path: Path) -> dict[str, list[float]]:
    """The ``L``/``R`` totals of a fit receipt; ``ValueError`` if either is absent."""
    sides = json.loads(receipt_path.read_text(encoding="utf-8")).get("sides", {})
    totals = {}
    for side, key in SIDE_KEYS.items():
        total = sides.get(side, {}).get("total_deg")
        if total is None:
            raise ValueError(f"{receipt_path}: sides.{side}.total_deg is missing")
        totals[key] = total
    return totals


def set_document_grip_rotation(receipt_path: Path, document_path: Path) -> bool:
    """Apply the fitted totals in ``receipt_path`` to ``document_path``.

    Returns True if the file changed. Raises ``ValueError`` when the receipt
    lacks totals or the document is not in the committed layout.
    """
    totals = read_totals(receipt_path)
    text = document_path.read_text(encoding="utf-8")
    document = json.loads(text)
    if format_json(document) + "\n" != text:
        raise ValueError(f"{document_path} is not in the committed layout")
    note = f"; grip rotation refitted per capture from {_label(receipt_path)} (#12221)"
    patched, changed = apply_grip_rotation(document, totals, note=note)
    if changed:
        document_path.write_text(format_json(patched) + "\n", encoding="utf-8")
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    apply = sub.add_parser("apply", help="set a document's hand rotation")
    apply.add_argument("receipt", type=Path)
    apply.add_argument("document", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    changed = set_document_grip_rotation(args.receipt, args.document)
    LOGGER.info("%s: %s", args.document, "updated" if changed else "already set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
