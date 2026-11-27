"""Carry a run's calibration from its v1 spec onto a v2 spec (#15670).

The v2 trunk is derived from an uncalibrated v1 spec.
``derive_landmark_v2_spec`` places L5/S1 from the hip joint centres. It
refuses a hip-calibrated spec that moves those centres without an HRG-C3
pelvis relocation. A pipeline run is tracked on the hip-calibrated, scaled
spec, though. Replaying its coordinates on the uncalibrated body puts the
feet in the wrong place, and the floating root slides off its contact (the
``driver_v2_bundle.npz`` fall at about 0.8 s).

:func:`carry_calibration` copies onto the v2 spec every part the
calibration changed (the hip and ankle frames, the shank bodies, the ground
contact and the subject fields). It refuses a part the v2 migration also
changed. The result records the calibrated spec as its source, so
``TrunkMigrator(calibrated, result)`` migrates the run's own coordinates.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from src.shared.python.motion_matching.full_body_migration import canonical_sha256

#: Lists of named parts, compared entry by entry.
_NAMED = ("joints", "bodies")
#: Top-level fields that describe a document rather than the model.
_IGNORED = frozenset({"provenance", "source_spec_sha256", "schema_version"})


def _by_name(parts: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {part["name"]: part for part in parts}


def _require_same_layout(
    base: Mapping[str, Any], calibrated: Mapping[str, Any]
) -> None:
    if list(base["coordinate_order"]) != list(calibrated["coordinate_order"]):
        raise ValueError("calibrated spec must keep the base coordinate_order")
    for kind in _NAMED:
        if [p["name"] for p in base[kind]] != [p["name"] for p in calibrated[kind]]:
            raise ValueError(
                "calibrated spec must have the same joints and bodies as the base"
            )


def _unchanged_in_v2(v2_part: object, base_part: object, label: str) -> None:
    if v2_part != base_part:
        raise ValueError(
            f"{label} is changed by both the calibration and the v2 migration"
        )


def _carry_named(
    out: dict[str, Any],
    base: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    kind: str,
) -> None:
    base_parts = _by_name(base[kind])
    v2_parts = _by_name(out[kind])
    carried = {}
    for part in calibrated[kind]:
        name = part["name"]
        if part != base_parts[name]:
            _unchanged_in_v2(v2_parts.get(name), base_parts[name], f"{kind}[{name}]")
            carried[name] = copy.deepcopy(part)
    out[kind] = [carried.get(part["name"], part) for part in out[kind]]


def carry_calibration(
    v2_spec: Mapping[str, Any],
    base_v1: Mapping[str, Any],
    calibrated_v1: Mapping[str, Any],
) -> dict[str, Any]:
    """``v2_spec`` with the parts ``calibrated_v1`` changed from ``base_v1``.

    Preconditions: ``v2_spec`` was derived from ``base_v1`` (its
    ``source_spec_sha256``). ``calibrated_v1`` has the base's coordinate order,
    joints and bodies. No part changed by the calibration was changed by
    the v2 migration.

    Postconditions: every part the calibration changed equals the
    calibrated part. Every other part, and the v2 provenance, is unchanged.
    ``source_spec_sha256`` is the calibrated spec's canonical hash. The
    inputs are not mutated.

    Raises:
        ValueError: If a precondition does not hold.
    """
    if v2_spec.get("source_spec_sha256") != canonical_sha256(base_v1):
        raise ValueError("v2_spec source_spec_sha256 is not the base v1 document")
    _require_same_layout(base_v1, calibrated_v1)
    out = copy.deepcopy(dict(v2_spec))
    for kind in _NAMED:
        _carry_named(out, base_v1, calibrated_v1, kind)
    for key, value in calibrated_v1.items():
        if key in _IGNORED or key in _NAMED or value == base_v1.get(key):
            continue
        _unchanged_in_v2(out.get(key), base_v1.get(key), key)
        out[key] = copy.deepcopy(value)
    out["source_spec_sha256"] = canonical_sha256(calibrated_v1)
    return out
