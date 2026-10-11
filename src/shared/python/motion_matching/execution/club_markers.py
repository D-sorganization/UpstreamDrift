"""Per-club clubhead marker clusters for the anthropometric documents (#12210).

The native candidate (``returned81_candidate.json``) was fitted to the driver
capture, so its ``Clubhead`` markers describe a driver head and shaft cluster.
A club whose head differs carries its own cluster in a committed file
(``club-markers/1``) that records where the offsets were measured. The spec
builder and ``scripts/set_spec_club_markers.py`` both read that file, so the
committed documents and a rebuilt document agree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "club-markers/1"
CLUB_MARKER_BODY = "Clubhead"
HEAD_TRIAD = ("Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3")
DERIVATION_METHOD = (
    "static-seed calibration: ik.attachments_m of the --static-seeds run on the "
    "club's capture (the static trial places every marker, club cluster included)"
)

Offset = tuple[float, float, float]


@dataclass(frozen=True)
class ClubMarkers:
    """A club's ``Clubhead``-frame marker offsets and their measured source."""

    club: str
    offsets_m: dict[str, Offset]
    source: dict[str, Any]

    @property
    def head_triad_edges_m(self) -> list[float]:
        """Sorted edge lengths of the head triad (m)."""
        return triad_edges_m(self.offsets_m)

    def to_document(self) -> dict[str, Any]:
        """The ``club-markers/1`` JSON document."""
        return {
            "schema": SCHEMA,
            "club": self.club,
            "body": CLUB_MARKER_BODY,
            "offsets_m": {k: list(v) for k, v in sorted(self.offsets_m.items())},
            "head_triad_edges_m": self.head_triad_edges_m,
            "source": self.source,
        }


def _offset(value: Sequence[float], label: str) -> Offset:
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,) or not np.isfinite(vec).all():
        raise ValueError(f"offset of {label} must be a finite 3-vector")
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def triad_edges_m(offsets: Mapping[str, Sequence[float]]) -> list[float]:
    """Sorted head-triad edge lengths (m); raises if a triad marker is missing."""
    missing = [label for label in HEAD_TRIAD if label not in offsets]
    if missing:
        raise ValueError(f"head triad markers missing: {missing}")
    p = [np.asarray(offsets[label], dtype=float) for label in HEAD_TRIAD]
    return sorted(
        float(np.linalg.norm(p[i] - p[j])) for i, j in ((0, 1), (1, 2), (0, 2))
    )


def load_club_markers(path: Path) -> ClubMarkers:
    """Read and validate a ``club-markers/1`` file.

    Raises ``ValueError`` for a wrong schema or body, a non-finite offset or
    a missing head triad.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA}")
    if data.get("body") != CLUB_MARKER_BODY:
        raise ValueError(f"{path}: body must be {CLUB_MARKER_BODY}")
    offsets = {k: _offset(v, k) for k, v in data["offsets_m"].items()}
    triad_edges_m(offsets)
    return ClubMarkers(str(data["club"]), offsets, dict(data["source"]))


def derive_club_markers(
    receipt_path: Path,
    *,
    club: str,
    receipt_label: str,
    receipt_commit: str | None = None,
) -> ClubMarkers:
    """Club cluster from a static-seed calibrated pipeline receipt.

    Takes every ``Clubhead`` attachment of ``ik.attachments_m``. The source
    block records ``receipt_label`` (repository path), the receipt's file
    SHA-256, the commit it was read at (the receipt is later regenerated on
    the new document) and its capture. Raises ``ValueError`` when the receipt
    has no ``Clubhead`` attachments.
    """
    raw = Path(receipt_path).read_bytes()
    receipt = json.loads(raw)
    attachments = receipt.get("ik", {}).get("attachments_m") or {}
    offsets = {
        label: _offset(att["offset_m"], label)
        for label, att in attachments.items()
        if att.get("body") == CLUB_MARKER_BODY
    }
    if not offsets:
        raise ValueError(f"{receipt_path}: no Clubhead ik.attachments_m to derive from")
    triad_edges_m(offsets)
    source = {
        "method": DERIVATION_METHOD,
        "receipt": receipt_label,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_commit": receipt_commit,
        "capture": receipt.get("capture"),
        "capture_sha256": receipt.get("capture_sha256"),
    }
    return ClubMarkers(club, offsets, source)


def apply_club_markers(
    document: dict[str, Any], offsets: Mapping[str, Sequence[float]]
) -> bool:
    """Set the ``Clubhead`` marker offsets of a spec document in place.

    Every label must already be attached to ``Clubhead`` and the labels must
    cover the document's whole ``Clubhead`` cluster (``ValueError``
    otherwise). Returns True when an offset changed.
    """
    attachments = document["marker_attachments"]
    for label in offsets:
        if attachments.get(label, {}).get("body") != CLUB_MARKER_BODY:
            raise ValueError(f"{label} is not a {CLUB_MARKER_BODY} marker")
    cluster = {k for k, v in attachments.items() if v["body"] == CLUB_MARKER_BODY}
    if cluster != set(offsets):
        raise ValueError(
            f"club marker labels {sorted(offsets)} do not cover {sorted(cluster)}"
        )
    changed = False
    for label, value in offsets.items():
        new = list(_offset(value, label))
        if attachments[label]["offset_m"] != new:
            attachments[label]["offset_m"] = new
            changed = True
    return changed
