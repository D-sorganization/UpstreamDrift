"""JSON loader producing a :class:`BodyTarget` from the pose-interchange schema.

This loader complements :mod:`loaders.c3d_body` by accepting the
single-frame (or multi-frame) JSON artifacts emitted by
:func:`pose_interchange.pose_io.save_motion_match_target`. The schema is
deliberately small - just the fields a :class:`BodyTarget` needs:

.. code-block:: json

    {
      "schema": "body_target_json_v1",
      "time_s":         [<seconds>, ...],
      "marker_names":   ["pelvis", ...],
      "marker_xyz":     [[[x, y, z], ...], ...],
      "impact_idx":     0,
      "events":         [{"label": "address", "frame": 0, "time_s": 0.0}],
      "source":         {"filename": "...", "format": "synthetic", ...},
      "coordinate_frame": "z_up_right_handed"
    }

The loader is intentionally permissive about ``opts`` / ``marker_set``:
because the JSON artifact is already on a uniform timegrid, no
resampling is performed. ``opts`` is accepted for dispatcher symmetry
and ignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..body_target import BodyEvent, BodyTarget
from ..club_target import AlignOptions, ClubTarget, SourceProvenance

logger = logging.getLogger(__name__)

JSON_BODY_TARGET_SCHEMA: str = "body_target_json_v1"
_HEX_DIGITS: frozenset[str] = frozenset(string.hexdigits)
_ZERO_SHA256: str = "0" * 64


def _resolve_source_sha256(raw_bytes: bytes, src: dict[str, Any], path: Path) -> str:
    """Validate declared sha256 or compute digest from loaded file bytes.

    Preconditions:
        raw_bytes: The exact bytes read from the file.
        src: The parsed source dictionary.
        path: Path to the loaded file.

    Returns:
        A 64-hex non-all-zero SHA-256 digest string.

    Raises:
        ValueError: If a declared sha256 is not a 64-hex string or is all zeros.
    """
    if "sha256" not in src or src["sha256"] is None:
        return hashlib.sha256(raw_bytes).hexdigest()
    declared = src["sha256"]
    if not isinstance(declared, str):
        raise ValueError(
            f"{path}: sha256 digest must be a 64-hex string, got {type(declared).__name__}"
        )
    cleaned = declared.strip().lower()
    if len(cleaned) != 64 or not all(c in _HEX_DIGITS for c in cleaned):
        raise ValueError(
            f"{path}: sha256 digest must be a 64-character hexadecimal string, got {declared!r}"
        )
    if cleaned == _ZERO_SHA256:
        raise ValueError(f"{path}: all-zero placeholder sha256 digest is prohibited")
    return cleaned


def _resolve_source_format(
    src: dict[str, Any], payload: dict[str, Any], path: Path
) -> str:
    """Resolve format from source or derive from schema; fail closed if absent.

    Preconditions:
        src: The parsed source dictionary.
        payload: The top-level parsed JSON object.
        path: Path to the loaded file.

    Returns:
        The format string.

    Raises:
        ValueError: If format is empty, or missing and cannot be derived.
    """
    raw_fmt = src.get("format")
    if raw_fmt is not None:
        fmt = str(raw_fmt).strip()
        if fmt:
            return fmt
        raise ValueError(f"{path}: source record contains empty 'format'")

    schema = payload.get("schema")
    if isinstance(schema, str) and schema.strip():
        return schema.strip()

    raise ValueError(
        f"{path}: body-target JSON source record missing required 'format'"
    )


def _build_source_provenance(
    raw_bytes: bytes, payload: dict[str, Any], path: Path
) -> SourceProvenance:
    """Build fail-closed source provenance from the payload's ``source`` record.

    Raises:
        ValueError: If ``source`` is not an object, or its format or digest
            fails validation.
    """
    src = payload["source"]
    if not isinstance(src, dict):
        raise ValueError(
            f"{path}: 'source' must be an object, got {type(src).__name__}"
        )
    return SourceProvenance(
        filename=str(src.get("filename", path.name)),
        format=_resolve_source_format(src, payload, path),
        subject_id=str(src.get("subject_id", "")),
        trial_id=str(src.get("trial_id", "")),
        sha256=_resolve_source_sha256(raw_bytes, src, path),
    )


def _read_body_target_payload(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read the raw bytes and validated top-level object of a body-target JSON.

    Raises:
        ValueError: If the payload is not an object, carries the wrong schema
            tag, or is missing required keys.
    """
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path}: body-target JSON must contain an object, got {type(payload).__name__}"
        )
    schema = payload.get("schema")
    if schema != JSON_BODY_TARGET_SCHEMA:
        raise ValueError(
            f"{path}: unsupported body-target JSON schema {schema!r}, "
            f"expected {JSON_BODY_TARGET_SCHEMA!r}"
        )
    required = {
        "time_s",
        "marker_names",
        "marker_xyz",
        "impact_idx",
        "events",
        "source",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(
            f"{path}: body-target JSON missing required keys: {sorted(missing)}"
        )
    return raw_bytes, payload


def load_body_target_json(
    path: Path,
    opts: AlignOptions | None = None,  # noqa: ARG001 - dispatcher symmetry
    *,
    marker_set: Sequence[str] | None = None,
    impact_source: ClubTarget | None = None,  # noqa: ARG001 - dispatcher symmetry
) -> BodyTarget:
    """Load a :class:`BodyTarget` from a pose-interchange JSON file.

    Parameters
    ----------
    path
        JSON file path. Must conform to ``body_target_json_v1``.
    opts
        Accepted for dispatcher symmetry; ignored (the JSON artifact is
        already on a uniform timegrid).
    marker_set
        Optional explicit subset of marker names to keep. ``None``
        keeps all markers in the file.
    impact_source
        Accepted for dispatcher symmetry; ignored.

    Returns
    -------
    BodyTarget
        Validated body target on the JSON's timegrid.

    Raises
    ------
    ValueError
        If the JSON schema tag, marker matrix shape, or other invariants
        are not satisfied.
    """
    p = Path(path)
    raw_bytes, payload = _read_body_target_payload(p)

    time = np.asarray(payload["time_s"], dtype=float)
    marker_names = tuple(str(n) for n in payload["marker_names"])
    marker_xyz = np.asarray(payload["marker_xyz"], dtype=float)
    if marker_xyz.ndim != 3 or marker_xyz.shape[2] != 3:
        raise ValueError(
            f"{p}: marker_xyz must have shape (N, M, 3), got {marker_xyz.shape}"
        )
    if marker_xyz.shape[0] != time.shape[0]:
        raise ValueError(
            f"{p}: time vector length {time.shape[0]} does not match "
            f"marker_xyz frame count {marker_xyz.shape[0]}"
        )
    if marker_xyz.shape[1] != len(marker_names):
        raise ValueError(
            f"{p}: marker_names length {len(marker_names)} does not match "
            f"marker_xyz marker count {marker_xyz.shape[1]}"
        )

    if marker_set is not None:
        keep_set = set(marker_set)
        keep_idx = [i for i, name in enumerate(marker_names) if name in keep_set]
        if not keep_idx:
            raise ValueError(
                f"{p}: marker_set {sorted(keep_set)!r} matched none of "
                f"{list(marker_names)!r}"
            )
        marker_xyz = marker_xyz[:, keep_idx, :]
        marker_names = tuple(marker_names[i] for i in keep_idx)

    events = tuple(
        BodyEvent(
            label=str(ev["label"]),
            frame=int(ev["frame"]),
            time_s=float(ev["time_s"]),
        )
        for ev in payload["events"]
    )

    source = _build_source_provenance(raw_bytes, payload, p)

    return BodyTarget(
        time=time,
        marker_xyz=marker_xyz,
        marker_names=marker_names,
        impact_idx=int(payload["impact_idx"]),
        events=events,
        source=source,
        coordinate_frame=payload.get("coordinate_frame", "z_up_right_handed"),
    )
