"""The committed full-body-v2 showpiece driver spec, derived from v1 (SHOW-A3, D4).

``docs/development/full_body_models/full_body_spec_anthro_driver_v2.json`` is
generated data: the v1 anthropometric driver spec migrated to the v2 regional
trunk at the HRG-D2 landmark joint locations, carrying the (off by default)
``trunk_coupling`` record. Never edit it by hand; a change to the v1 driver
spec (the A2 grip offset, the A4 neck) reaches it by regeneration, and
``--check`` (run by the drift test) fails when the committed bytes are stale::

    python -m src.shared.python.motion_matching.full_body_v2_showpiece --write
    python -m src.shared.python.motion_matching.full_body_v2_showpiece --check
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require
from src.shared.python.motion_matching.full_body_v2_coupling import (
    with_trunk_coupling,
)
from src.shared.python.motion_matching.full_body_v2_geometry import (
    derive_landmark_v2_spec,
)

__all__ = ["derive_showpiece_v2", "main", "showpiece_v2_bytes"]

_MODELS = Path(__file__).resolve().parents[4] / "docs/development/full_body_models"
SHOWPIECE_V1 = _MODELS / "full_body_spec_anthro_driver.json"
SHOWPIECE_V2 = _MODELS / "full_body_spec_anthro_driver_v2.json"
PROVENANCE = (
    "SHOW-A3 (#14552): full-body-v2 showpiece driver derived from "
    "full_body_spec_anthro_driver.json at the HRG-D2 landmark trunk locations"
)


def derive_showpiece_v2(v1: Mapping[str, Any]) -> dict[str, Any]:
    """The v2 showpiece document of ``v1`` (a ``full-body-v1`` driver spec).

    Postcondition: ``full-body-v2`` with an inactive ``trunk_coupling`` record;
    ``v1`` is never modified.
    """
    require(v1.get("schema_version") == "full-body-v1", "v1 driver spec required")
    return with_trunk_coupling(derive_landmark_v2_spec(v1, provenance=PROVENANCE))


def showpiece_v2_bytes(v1: Mapping[str, Any]) -> bytes:
    """Canonical committed bytes of :func:`derive_showpiece_v2` (sorted keys)."""
    return json.dumps(derive_showpiece_v2(v1), sort_keys=True).encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """``--write`` regenerates the v2 spec; ``--check`` exits 1 when it is stale."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--v1", type=Path, default=SHOWPIECE_V1)
    parser.add_argument("--v2", type=Path, default=SHOWPIECE_V2)
    args = parser.parse_args(argv)
    fresh = showpiece_v2_bytes(json.loads(args.v1.read_text(encoding="utf-8")))
    if args.write:
        args.v2.write_bytes(fresh)
        return 0
    if not args.v2.exists() or args.v2.read_bytes() != fresh:
        sys.stderr.write(f"{args.v2.name} is stale; rerun with --write\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
