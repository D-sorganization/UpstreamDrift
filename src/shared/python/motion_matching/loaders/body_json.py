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

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..body_target import BodyEvent, BodyTarget
from ..club_target import AlignOptions, ClubTarget, SourceProvenance

logger = logging.getLogger(__name__)

JSON_BODY_TARGET_SCHEMA: str = "body_target_json_v1"


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
    payload = json.loads(p.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != JSON_BODY_TARGET_SCHEMA:
        raise ValueError(
            f"{p}: unsupported body-target JSON schema {schema!r}, "
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
            f"{p}: body-target JSON missing required keys: {sorted(missing)}"
        )

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

    src = payload["source"]
    source = SourceProvenance(
        filename=str(src.get("filename", p.name)),
        format=str(src.get("format", "synthetic")),
        subject_id=str(src.get("subject_id", "")),
        trial_id=str(src.get("trial_id", "")),
        sha256=str(src.get("sha256", "0" * 64)),
    )

    return BodyTarget(
        time=time,
        marker_xyz=marker_xyz,
        marker_names=marker_names,
        impact_idx=int(payload["impact_idx"]),
        events=events,
        source=source,
        coordinate_frame=payload.get("coordinate_frame", "z_up_right_handed"),
    )
