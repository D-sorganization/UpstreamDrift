"""Mirror the right knee of committed full-body spec documents (#12057).

The spec's knee follows the OpenSim gait2392 convention: flexion is negative on
both sides (hinge axis ``+z`` in the femur frame). The right knee was built with
a ``-z`` axis, so its declared ``[-120, 10]`` degree range capped flexion at 10
degrees. Committed documents carry hand-tuned contact parameters, so they are
not rebuilt; this script applies only the knee-frame conjugation, idempotently,
and appends a provenance note.

Usage::

    python3 -m scripts.mirror_spec_right_knee \
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
from src.shared.python.motion_matching.execution.spec_builder import (
    mirror_document_knee,
)

LOGGER = logging.getLogger(__name__)
NOTE = "; right knee axis mirrored to the OpenSim flexion-negative convention (#12057)"


def mirror_file(path: Path, sides: Sequence[str] = ("r",)) -> bool:
    """Mirror the knees of ``sides`` in the document at ``path``.

    Returns True if the file changed. Raises ``ValueError`` when the file is not
    in the committed layout (it would be reformatted).
    """
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    if format_json(document) + "\n" != text:
        raise ValueError(f"{path} is not in the committed layout; refusing to rewrite")
    changed = [side for side in sides if mirror_document_knee(document, side)]
    if not changed:
        return False
    if NOTE not in document.get("provenance", ""):
        document["provenance"] = document.get("provenance", "") + NOTE
    path.write_text(format_json(document) + "\n", encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("documents", nargs="+", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in args.documents:
        state = "mirrored" if mirror_file(path) else "already mirrored"
        LOGGER.info("%s: %s", path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
