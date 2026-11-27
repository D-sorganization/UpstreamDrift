"""Best-match driver model on the v2 lumbar+thoracic trunk (SHOW-A8, #14732).

The v1 driver spec is pinned by many engines and fixtures, so it is never edited.
The best-match path consumes the derived ``full_body_spec_anthro_driver_v2.json``
and the v1 fit's coordinates are migrated onto it by ``TrunkMigrator``.  That is a
replay of the v1 fit on the new trunk, not a refit (the refit is A9).

Design by Contract:
    Preconditions: ``q`` is (T, n) in ``order``; ``order`` equals the v1 spec's
    ``coordinate_order``; ``v2_spec`` was derived from exactly ``v1_spec``.
    Postconditions: the result is (T, len(v2 coordinate_order)) and finite.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.full_body_migration import TrunkMigrator

Array = NDArray[np.float64]


def trunk_segments(spec: Mapping[str, Any]) -> list[str]:
    """Names of the spec bodies that belong to the trunk (pelvis to thorax)."""
    marks = ("LowerTorso", "UpperTorsoBase", "trunk_lumbar", "COMRod")
    return [b["name"] for b in spec["bodies"] if b["name"].endswith(marks)]


def migrate_q_to_v2(
    v1_spec: Mapping[str, Any],
    v2_spec: Mapping[str, Any],
    q: Array,
    order: Sequence[str],
) -> Array:
    """Migrate ``(T, n)`` v1 coordinates to the v2 coordinate order."""
    q = np.asarray(q, dtype=float)
    require(
        list(order) == list(v1_spec["coordinate_order"]),
        "q must be in the v1 spec coordinate_order",
        (len(order), len(v1_spec["coordinate_order"])),
    )
    require(q.ndim == 2 and q.shape[1] == len(order), "q must be (T, n)", q.shape)
    out = TrunkMigrator(v1_spec, v2_spec).migrate_coordinates(q, order)
    ensure(
        out.shape == (q.shape[0], len(v2_spec["coordinate_order"]))
        and bool(np.isfinite(out).all()),
        "migrated q must be finite and in v2 coordinate_order",
    )
    return np.asarray(out, dtype=float)


def migrate_bundle(
    v1_bundle: Path, v1_spec_path: Path, v2_spec_path: Path, out: Path
) -> Path:
    """Write a v2 same-input bundle whose q is ``v1_bundle`` migrated to v2."""
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.shared.python.motion_matching.same_input.reference import (
        generate_reference_bundle,
    )

    v1 = json.loads(v1_spec_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_spec_path.read_text(encoding="utf-8"))
    loaded = InputBundle.load(v1_bundle)
    q_v2 = migrate_q_to_v2(v1, v2, loaded.reference_q, loaded.coordinate_order)
    times = np.asarray(loaded.dt_s * np.arange(len(q_v2)), dtype=np.float64)
    bundle = generate_reference_bundle(
        v2_spec_path.read_bytes(),
        times,
        q_v2,
        provenance={"issue": "14732", "migration": "replay, not a refit"},
    )
    bundle.save(out)
    return out
