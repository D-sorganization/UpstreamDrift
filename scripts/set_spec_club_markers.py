"""Give a committed full-body spec document its own club marker cluster (#12210).

The anthropometric documents took their ``Clubhead`` markers from the
driver-fitted native candidate, so the 7-iron document carried the driver's
head triad and shaft cluster. ``derive`` writes a ``club-markers/1`` file from
a static-seed calibrated receipt; ``apply`` sets a document's ``Clubhead``
offsets from that file, idempotently, and appends a provenance note. Committed
documents carry hand-tuned contact parameters, so they are not rebuilt. The
spec builder reads the same file (``assets.get_club_markers``).

Usage::

    python3 -m scripts.set_spec_club_markers derive --club iron7 \
        --receipt-commit 1bef55345715f5c4787ab80f2b54bc19736df633 \
        docs/development/full_body_models/evidence/ground_support/anthro_iron_seeds_zmp/receipt.json \
        docs/development/full_body_models/evidence/club_markers/iron7.json
    python3 -m scripts.set_spec_club_markers apply \
        docs/development/full_body_models/evidence/club_markers/iron7.json \
        docs/development/full_body_models/full_body_spec_anthro_iron7.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from scripts.mirror_spec_left_hip import format_json
from src.shared.python.motion_matching.execution.club_markers import (
    apply_club_markers,
    derive_club_markers,
    load_club_markers,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _note(markers_path: Path) -> str:
    return f"; club markers from {markers_path.name} (#12210)"


def set_document_club_markers(markers_path: Path, document_path: Path) -> bool:
    """Apply the club cluster in ``markers_path`` to ``document_path``.

    Returns True if the file changed. Raises ``ValueError`` when the document
    is not in the committed layout or is for another club.
    """
    markers = load_club_markers(markers_path)
    text = document_path.read_text(encoding="utf-8")
    document = json.loads(text)
    if format_json(document) + "\n" != text:
        raise ValueError(f"{document_path} is not in the committed layout")
    if document["club"]["name"] != markers.club:
        raise ValueError(
            f"{document_path} is a {document['club']['name']} document, "
            f"not {markers.club}"
        )
    if not apply_club_markers(document, markers.offsets_m):
        return False
    note = _note(markers_path)
    if note not in document.get("provenance", ""):
        document["provenance"] = document.get("provenance", "") + note
    document_path.write_text(format_json(document) + "\n", encoding="utf-8")
    return True


def write_derived(receipt: Path, club: str, out: Path, commit: str | None) -> None:
    """Derive the club cluster from ``receipt`` and write it to ``out``."""
    label = receipt.resolve().relative_to(REPO_ROOT).as_posix()
    markers = derive_club_markers(
        receipt, club=club, receipt_label=label, receipt_commit=commit
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_json(markers.to_document()) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    derive = sub.add_parser("derive", help="club-markers/1 file from a receipt")
    derive.add_argument("--club", required=True)
    derive.add_argument(
        "--receipt-commit", default=None, help="commit the receipt was read at"
    )
    derive.add_argument("receipt", type=Path)
    derive.add_argument("out", type=Path)
    apply = sub.add_parser("apply", help="set documents' Clubhead offsets")
    apply.add_argument("markers", type=Path)
    apply.add_argument("documents", nargs="+", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "derive":
        write_derived(args.receipt, args.club, args.out, args.receipt_commit)
        LOGGER.info("%s: derived from %s", args.out, args.receipt)
        return 0
    for path in args.documents:
        changed = set_document_club_markers(args.markers, path)
        LOGGER.info("%s: %s", path, "updated" if changed else "already set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
