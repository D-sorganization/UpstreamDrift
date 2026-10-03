#!/usr/bin/env python3
"""Generate glyph-set-examples.json fixtures using build_glyphs (#11288)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs


def build_example_cases() -> list[dict]:
    """Build the 4 canonical synthetic fixture cases."""
    style = ForceGlyphStyle()

    # Case 1: force only
    w_force = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground_0",
        body="left_foot",
        point_m=(0.1, 0.0, 0.0),
        force_n=(0.0, 0.0, 500.0),
        torque_nm=None,
        source="synthetic:contact",
    )
    frame_force = ForceTorqueFrame(time_s=0.1, engine="synthetic", wrenches=(w_force,))
    glyphs_force = build_glyphs(frame_force, style)

    # Case 2: torque only (+z)
    w_torque_z = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:yaw_actuator",
        body="torso",
        point_m=(0.0, 0.0, 1.0),
        force_n=None,
        torque_nm=(0.0, 0.0, 25.0),
        source="synthetic:actuator",
    )
    frame_torque_z = ForceTorqueFrame(
        time_s=0.2, engine="synthetic", wrenches=(w_torque_z,)
    )
    glyphs_torque_z = build_glyphs(frame_torque_z, style)

    # Case 3: torque with diagonal axis
    diag_mag = 30.0
    diag_comp = diag_mag / math.sqrt(3.0)
    w_torque_diag = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="joint:shoulder_reaction",
        body="upper_arm",
        point_m=(0.2, 0.3, 1.4),
        force_n=None,
        torque_nm=(diag_comp, diag_comp, diag_comp),
        source="synthetic:reaction",
    )
    frame_torque_diag = ForceTorqueFrame(
        time_s=0.3, engine="synthetic", wrenches=(w_torque_diag,)
    )
    glyphs_torque_diag = build_glyphs(frame_torque_diag, style)

    # Case 4: clamped (exceeding upper and lower bounds)
    w_large = OverlayWrench(
        kind=WrenchKind.EXTERNAL,
        label="external:impact_large",
        body="clubhead",
        point_m=(0.5, 0.5, 0.2),
        force_n=(10000.0, 0.0, 0.0),
        torque_nm=None,
        source="synthetic:impact",
    )
    w_small = OverlayWrench(
        kind=WrenchKind.GRIP,
        label="grip:finger_small",
        body="handle",
        point_m=(0.0, 0.0, 0.8),
        force_n=(0.0, 5.0, 0.0),
        torque_nm=None,
        source="synthetic:grip",
    )
    frame_clamped = ForceTorqueFrame(
        time_s=0.4, engine="synthetic", wrenches=(w_large, w_small)
    )
    glyphs_clamped = build_glyphs(frame_clamped, style)

    return [
        {
            "name": "synthetic_force_only",
            "valid": True,
            "description": "Synthetic frame with pure force arrow",
            "data": glyphs_force.to_dict(),
        },
        {
            "name": "synthetic_torque_only_z",
            "valid": True,
            "description": "Synthetic frame with torque arc about +z axis",
            "data": glyphs_torque_z.to_dict(),
        },
        {
            "name": "synthetic_torque_diagonal",
            "valid": True,
            "description": "Synthetic frame with torque arc along diagonal (1, 1, 1) axis",
            "data": glyphs_torque_diag.to_dict(),
        },
        {
            "name": "synthetic_clamped",
            "valid": True,
            "description": "Synthetic frame with clamped upper and lower magnitude forces",
            "data": glyphs_clamped.to_dict(),
        },
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_path = repo_root / "schemas" / "glyph-set-examples.json"
    payload = {
        "description": "Cross-runtime glyph-set wire schema conformance fixtures (ADR-0052, #11288).",
        "cases": build_example_cases(),
    }
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Generated {target_path}")


if __name__ == "__main__":
    main()
