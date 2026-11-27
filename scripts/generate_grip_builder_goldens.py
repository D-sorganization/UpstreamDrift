#!/usr/bin/env python3
"""Regenerate HRG-GB3 golden checksum table (issue #15125)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from src.shared.python.grip_build.builder import BUILDER_VERSION, build_grip
from src.shared.python.grip_build.full_builder import BUILDER_VERSION as GB3_VERSION
from src.shared.python.grip_build.goldens import GOLDEN_RECEIPT_PATH
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize
from src.shared.python.hand_size import HAND_LENGTH_RANGE_M
from src.shared.python.motion_matching.grip_contract import GRIP_STRENGTH_PRESETS_DEG

STRENGTHS = ("neutral", "strong", "weak")
STYLES = tuple(GripStyle)
HANDEDNESS = ("right", "left")
SIZES = ("small", "medium", "large")


def _hand_length(size: str) -> float:
    lo, hi = HAND_LENGTH_RANGE_M
    return {"small": lo, "medium": (lo + hi) * 0.5, "large": hi}[size]


def _spec(strength: str, style: GripStyle, handedness: str, size: str) -> GripSpec:
    delta = GRIP_STRENGTH_PRESETS_DEG[strength]
    lead = replace(HandGripSpec(), v_azimuth_deg=HandGripSpec().v_azimuth_deg + delta)
    trail = replace(
        HandGripSpec(
            v_azimuth_deg=HandGripSpec().v_azimuth_deg + delta,
            thumb_azimuth_deg=45.0,
        ),
    )
    base = replace(
        GripSpec.neutral(),
        lead=lead,
        trail=trail,
        style=style,
        handedness=handedness,  # type: ignore[arg-type]
    )
    return replace(base, hand_size=HandSize(length_m=_hand_length(size)))


def main() -> None:
    cases = []
    for strength in STRENGTHS:
        for style in STYLES:
            for handedness in HANDEDNESS:
                for size in SIZES:
                    key = f"{strength}_{style.value}_{handedness}_{size}"
                    spec = _spec(strength, style, handedness, size)
                    result = build_grip(spec)
                    joints_json = json.dumps(
                        result.joint_angles_rad, sort_keys=True, separators=(",", ":")
                    )
                    joint_sha = hashlib.sha256(joints_json.encode()).hexdigest()
                    contact_json = json.dumps(
                        sorted(result.active_contact_pairs),
                        separators=(",", ":"),
                    )
                    contact_sha = hashlib.sha256(contact_json.encode()).hexdigest()
                    cases.append(
                        {
                            "key": key,
                            "spec_sha256": spec.spec_sha256,
                            "status": result.status.value,
                            "joint_sha256": joint_sha,
                            "contact_sha256": contact_sha,
                        }
                    )
    receipt = {
        "schema_version": 1,
        "builder_version": GB3_VERSION,
        "grip_build_builder_version": BUILDER_VERSION,
        "cases": cases,
    }
    GOLDEN_RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {GOLDEN_RECEIPT_PATH} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
