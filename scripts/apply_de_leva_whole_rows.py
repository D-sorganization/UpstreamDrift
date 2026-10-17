"""Apply the whole-row de Leva table to committed full-body documents (#13037).

``anthropometry.DE_LEVA_MALE`` is now derived from whole rows of de Leva (1996)
Table 4. Three rows of the earlier hand-copied table mixed two endpoint rows:

* ``trunk``: the length was the suprasternale-to-mid-hip row's (0.5319 at the
  reference subject); the CM and radii were the cervicale-to-mid-hip row's
  (0.6033). The builder uses this length from the hips to the cervicale.
* ``upper_trunk``: length and CM of the suprasternale-to-xiphion row, radii of
  no row. It is now the cervicale-to-xiphion row.
* ``shank``: length of the knee-to-lateral-malleolus row. The documents' legs
  are the scaled Rajagopal model, so only their ``anthropometry`` block holds it.

The capture fixed the subject's trunk length (``scan_geometry.py``: scale
factors on the old length, canonical ``trunk_scale`` 1.15). That length is
kept: ``trunk_scale`` becomes ``1.15 * 0.5319 / 0.6033``, so every joint,
frame and marker stays where it is, which this script checks against a fresh
``build_upper_body``. The upper-trunk solid is rebuilt from its whole row, and
the hashes, subject and ``anthropometry`` block are updated. Committed documents
carry hand-tuned contact parameters, so they are not rebuilt; the edit is
idempotent and appends a provenance note.

Usage::

    python3 -m scripts.apply_de_leva_whole_rows \\
        --receipt docs/development/full_body_models/evidence/anthropometry/de_leva_whole_rows_receipt.json \\
        docs/development/full_body_models/full_body_spec_anthro_driver.json \\
        docs/development/full_body_models/full_body_spec_anthro_iron7.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.mirror_spec_left_hip import format_json
from src.shared.python.body_segment_parameters.de_leva_1996 import table4_row
from src.shared.python.motion_matching.anthropometric_geometry import (
    build_upper_body,
)
from src.shared.python.motion_matching.anthropometry import (
    DE_LEVA_MALE,
    de_leva_table_dict,
    de_leva_table_sha256,
)
from src.shared.python.motion_matching.club_models import CLUBS
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

LOGGER = logging.getLogger(__name__)
NATIVE = Path(
    "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
NOTE = (
    "; de Leva table derived from whole Table 4 rows, trunk length kept by "
    "rescaling trunk_scale and upper-trunk solid rebuilt (#13037)"
)
#: The fit-path trunk length before #13037 was this row's.
SUPERSEDED_TRUNK_ROW = "trunk_supr_midh"
REBUILT_SOLID = "/upper_trunk"
#: Joints, frames and the other solids must agree with a fresh build this well.
GEOMETRY_TOL = 1e-9


def preserved_trunk_scale(old_scale: float) -> float:
    """``trunk_scale`` giving the same hips-to-cervicale length on the new row.

    Precondition: ``old_scale`` is positive and finite. Postcondition:
    ``result * new_trunk_length == old_scale * superseded_trunk_length``.
    """
    if not np.isfinite(old_scale) or old_scale <= 0:
        raise ValueError(f"trunk_scale must be positive and finite, got {old_scale}")
    old_length = table4_row(SUPERSEDED_TRUNK_ROW, "male").length_m
    return float(old_scale * old_length / DE_LEVA_MALE["trunk"].length_m)


def _solids(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["name"]: s for b in document["bodies"] for s in b["solids"]}


def _solid_gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return float(
        max(
            abs(a["mass_kg"] - b["mass_kg"]),
            np.max(np.abs(np.subtract(a["com_m"], b["com_m"]))),
            np.max(np.abs(np.subtract(a["inertia_com_kg_m2"], b["inertia_com_kg_m2"]))),
        )
    )


def _translation_gap(
    document: Mapping[str, Any], upper: Mapping[str, Any], key: str, field: str
) -> float:
    committed = {item["name"]: item for item in document[key]}
    gaps = [
        np.max(
            np.abs(
                np.asarray(committed[item["name"]][field])[:3, 3]
                - np.asarray(item[field])[:3, 3]
            )
        )
        for item in upper[key]
    ]
    return float(max(gaps))


def _fresh_upper_body(
    subject: Mapping[str, Any], club: str, trunk_scale: float, native: Mapping
) -> dict[str, Any]:
    return build_upper_body(
        native,
        stature_m=float(subject["stature_m"]),
        mass_kg=float(subject["mass_kg"]),
        trunk_scale=trunk_scale,
        arm_scale=float(subject["arm_scale"]),
        shoulder_scale=float(subject["shoulder_scale"]),
        club=CLUBS[club],
        grip_roll_deg=float(subject["grip_roll_deg"]),
        grip_rotation_deg=subject["grip_rotation_deg"],
    )


def apply_whole_rows(
    document: dict[str, Any], native: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Update ``document`` in place; ``None`` when it already has the table.

    Raises ``ValueError`` when the rescaled trunk would move any joint, frame
    or solid other than the upper trunk by more than :data:`GEOMETRY_TOL`.
    Postcondition: the document's table hashes equal ``de_leva_table_sha256()``.
    """
    current = de_leva_table_sha256()
    if document.get("de_leva_table_sha256") == current:
        return None
    subject = document["subject"]
    old_scale = float(subject["trunk_scale"])
    new_scale = preserved_trunk_scale(old_scale)
    upper = _fresh_upper_body(subject, document["club"]["name"], new_scale, native)
    committed, fresh = _solids(document), _solids(upper)
    solid_gap = max(
        _solid_gap(committed[name], solid)
        for name, solid in fresh.items()
        if not name.endswith(REBUILT_SOLID)
    )
    geometry_gap = max(
        solid_gap,
        _translation_gap(document, upper, "joints", "parent_to_base"),
        _translation_gap(document, upper, "frames", "placement"),
    )
    if geometry_gap > GEOMETRY_TOL:
        raise ValueError(
            f"rescaled trunk moves the document's geometry by {geometry_gap:.3e}"
        )
    rebuilt = {}
    for body in document["bodies"]:
        for i, solid in enumerate(body["solids"]):
            if solid["name"].endswith(REBUILT_SOLID):
                rebuilt[solid["name"]] = {
                    "before": solid,
                    "after": fresh[solid["name"]],
                }
                body["solids"][i] = fresh[solid["name"]]
    if not rebuilt:
        raise ValueError(f"document has no {REBUILT_SOLID} solid")
    previous = document.get("de_leva_table_sha256")
    subject.update(trunk_scale=new_scale, de_leva_table_sha256=current)
    document["de_leva_table_sha256"] = current
    document["anthropometry"].update(
        de_leva_table_sha256=current, segments=de_leva_table_dict()
    )
    if NOTE not in document.get("provenance", ""):
        document["provenance"] = document.get("provenance", "") + NOTE
    return {
        "de_leva_table_sha256": {"before": previous, "after": current},
        "trunk_scale": {"before": old_scale, "after": new_scale},
        "max_geometry_change_m": geometry_gap,
        "rebuilt_solids": rebuilt,
    }


def apply_file(path: Path, native: Mapping[str, Any]) -> dict[str, Any] | None:
    """Apply :func:`apply_whole_rows` to the committed document at ``path``.

    Returns the change record, or ``None`` when the file already has the table.
    Raises ``ValueError`` when the file is not in the committed layout.
    """
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    if format_json(document) + "\n" != text:
        raise ValueError(f"{path} is not in the committed layout; refusing to rewrite")
    before = {
        "file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "canonical_sha256": canonical_sha256(document),
    }
    change = apply_whole_rows(document, native)
    if change is None:
        return None
    out = format_json(document) + "\n"
    path.write_text(out, encoding="utf-8")
    after = {
        "file_sha256": hashlib.sha256(out.encode("utf-8")).hexdigest(),
        "canonical_sha256": canonical_sha256(document),
    }
    return {"document": path.as_posix(), "before": before, "after": after, **change}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--native", type=Path, default=NATIVE)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    native = json.loads(args.native.read_text(encoding="utf-8"))
    records = []
    for path in args.documents:
        record = apply_file(path, native)
        LOGGER.info("%s: %s", path, "updated" if record else "already current")
        if record:
            records.append(record)
    if args.receipt and records:
        receipt = {
            "issue": 13037,
            "table": "de Leva (1996) Table 4, males, whole rows",
            "superseded_trunk_row": SUPERSEDED_TRUNK_ROW,
            "geometry_tolerance_m": GEOMETRY_TOL,
            "documents": records,
        }
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
