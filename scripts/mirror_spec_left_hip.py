"""Mirror the left hip of committed full-body spec documents (OSV-6, #11737).

The spec builder now keeps OpenSim's left-hip convention (``hip_adduction_l``
and ``hip_rotation_l`` coefficients of -1, read by ``spec_builder.hip_axis_signs``).
Committed documents carry hand-tuned contact parameters, so they are not rebuilt;
this script applies only the hip-frame conjugation, idempotently, and appends a
provenance note.

Usage::

    python3 -m scripts.mirror_spec_left_hip \
        docs/development/full_body_models/full_body_spec_anthro_driver.json \
        docs/development/full_body_models/full_body_spec_anthro_iron7.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching.execution.spec_builder import (
    hip_axis_signs,
    mirror_document_hip,
)

LOGGER = logging.getLogger(__name__)
OSIM = Path("src/engines/physics_engines/opensim/models/golf_humanoid.osim")
NOTE = "; left hip axes mirrored to the OpenSim convention (OSV-6 #11737)"


WIDTH = 80


def _scalar(value: Any) -> str:
    text = json.dumps(value)
    if isinstance(value, float):
        text = re.sub(r"e([+-]?)0*(\d)", lambda m: f"e{m[1].lstrip('+')}{m[2]}", text)
    return text


def _is_matrix(value: list) -> bool:
    """Prettier always breaks arrays of two or more multi-element arrays/objects."""
    return len(value) > 1 and all(
        isinstance(v, (list, dict)) and len(v) > 1 for v in value
    )


def format_json(value: Any, indent: int = 0, used: int = 0, tail: int = 0) -> str:
    """JSON in the committed (Prettier) layout.

    Sorted keys, 2-space indent, arrays inline when the whole line (``used``
    columns before the value, ``tail`` after it) fits in :data:`WIDTH`; longer
    number arrays fill lines, other arrays take one item per line; matrix-like
    arrays always break.
    """
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        keys = sorted(value)
        items = []
        for i, key in enumerate(keys):
            prefix = f"{pad}  {json.dumps(key)}: "
            comma = 1 if i < len(keys) - 1 else 0
            items.append(
                prefix + format_json(value[key], indent + 2, len(prefix), comma)
            )
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        scalars = not any(isinstance(v, (list, dict)) for v in value)
        if scalars or not _is_matrix(value):
            inline = "[" + ", ".join(format_json(v, indent) for v in value) + "]"
            if "\n" not in inline and used + len(inline) + tail <= WIDTH:
                return inline
        inner = " " * (indent + 2)
        numbers = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        )
        if numbers:
            lines, line = [], ""
            for i, v in enumerate(value):
                item = _scalar(v) + ("," if i < len(value) - 1 else "")
                candidate = f"{line} {item}" if line else item
                if line and len(inner) + len(candidate) > WIDTH:
                    lines.append(line)
                    candidate = item
                line = candidate
            lines.append(line)
            return "[\n" + "\n".join(inner + ln for ln in lines) + f"\n{pad}]"
        items = [
            inner + format_json(v, indent + 2, len(inner), int(i < len(value) - 1))
            for i, v in enumerate(value)
        ]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return _scalar(value)


def mirror_file(path: Path, sides: Sequence[str]) -> bool:
    """Mirror ``sides`` of the document at ``path``; return True if it changed."""
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    if format_json(document) + "\n" != text:
        raise ValueError(f"{path} is not in the committed layout; refusing to rewrite")
    changed = [side for side in sides if mirror_document_hip(document, side)]
    if not changed:
        return False
    if NOTE not in document.get("provenance", ""):
        document["provenance"] = document.get("provenance", "") + NOTE
    path.write_text(format_json(document) + "\n", encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--osim", type=Path, default=OSIM)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sides = [side for side, sign in hip_axis_signs(args.osim).items() if sign < 0]
    for path in args.documents:
        state = "mirrored" if mirror_file(path, sides) else "already mirrored"
        LOGGER.info("%s: %s (%s)", path, state, ",".join(sides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
