"""Apply the static tibial-torsion offset to committed full-body specs (#12789).

The committed anthropometric documents carry hand-tuned contact parameters and
refitted grips, so they are not rebuilt. This script applies only the shared
``tibial_torsion.apply_tibial_torsion`` rotation of the ankle base frames,
records the offset in the document, and appends a provenance note. It is
idempotent: a document that already records the same offset is left alone.

Usage::

    python3 -m scripts.apply_tibial_torsion \
        docs/development/full_body_models/full_body_spec_anthro_driver.json \
        docs/development/full_body_models/full_body_spec_anthro_iron7.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from scripts.mirror_spec_left_hip import format_json
from src.shared.python.motion_matching.tibial_torsion import (
    DEFAULT_TIBIAL_TORSION_OFFSET_DEG,
    SIDES,
    TIBIAL_TORSION_KEY,
    apply_tibial_torsion,
)

LOGGER = logging.getLogger(__name__)
NOTE = "; static external tibial-torsion offset {offset:g} deg per ankle (#12789)"


def apply_file(
    path: Path, offset_deg: float = DEFAULT_TIBIAL_TORSION_OFFSET_DEG
) -> bool:
    """Apply ``offset_deg`` to both ankles of the document at ``path``.

    Returns True if the file changed. Raises ``ValueError`` when the file is
    not in the committed layout, or already records a different offset.
    """
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    if format_json(document) + "\n" != text:
        raise ValueError(f"{path} is not in the committed layout; refusing to rewrite")
    record = document.get(TIBIAL_TORSION_KEY)
    if record is not None:
        if all(record.get(side) == offset_deg for side in SIDES):
            return False
        raise ValueError(f"{path} already records {TIBIAL_TORSION_KEY}={record}")
    updated = apply_tibial_torsion(document, dict.fromkeys(SIDES, offset_deg))
    if updated == document:
        return False
    updated["provenance"] = updated.get("provenance", "") + NOTE.format(
        offset=offset_deg
    )
    path.write_text(format_json(updated) + "\n", encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument(
        "--offset", type=float, default=DEFAULT_TIBIAL_TORSION_OFFSET_DEG
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in args.documents:
        state = "applied" if apply_file(path, args.offset) else "already applied"
        LOGGER.info("%s: %s", path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
