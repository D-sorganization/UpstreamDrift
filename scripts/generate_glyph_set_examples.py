"""Generate canonical glyph set examples for schema validation and visual verification (FTO-3, #11288)."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import build_glyphs

OUTPUT_FILE = ROOT / "schemas" / "glyph-set-examples.json"


def generate_synthetic_frames() -> dict[str, ForceTorqueFrame]:
    frames = {}

    # 1. Force only
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground_right",
        body="calcn_r",
        point_m=(0.1, 0.0, 0.0),
        force_n=(0.0, 50.0, 500.0),
        torque_nm=None,
        source="synthetic_test",
    )
    frames["synthetic_force_only"] = ForceTorqueFrame(
        time_s=0.1,
        engine="synthetic",
        wrenches=(w1,),
    )

    # 2. Torque only (+z)
    w2 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint_actuator:torso_z",
        body="torso",
        point_m=(0.0, 0.0, 1.0),
        force_n=None,
        torque_nm=(0.0, 0.0, 40.0),
        source="synthetic_test",
    )
    frames["synthetic_torque_only_pos_z"] = ForceTorqueFrame(
        time_s=0.2,
        engine="synthetic",
        wrenches=(w2,),
    )

    # 3. Torque with a diagonal axis
    w3 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint_actuator:shoulder_diagonal",
        body="humerus_r",
        point_m=(0.2, 0.2, 1.4),
        force_n=None,
        torque_nm=(15.0, 15.0, 15.0),
        source="synthetic_test",
    )
    frames["synthetic_torque_diagonal_axis"] = ForceTorqueFrame(
        time_s=0.3,
        engine="synthetic",
        wrenches=(w3,),
    )

    # 4. Clamped
    w4_large = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:impact_large",
        body="clubhead",
        point_m=(0.5, 0.0, 0.1),
        force_n=(0.0, 10000.0, 0.0),
        torque_nm=None,
        source="synthetic_test",
    )
    w4_small = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:drag_small",
        body="shaft",
        point_m=(0.0, 0.0, 1.2),
        force_n=(0.0, 5.0, 0.0),
        torque_nm=None,
        source="synthetic_test",
    )
    frames["synthetic_clamped"] = ForceTorqueFrame(
        time_s=0.4,
        engine="synthetic",
        wrenches=(w4_large, w4_small),
    )

    return frames


def build_examples_dict() -> dict[str, dict]:
    frames = generate_synthetic_frames()
    examples = {}
    for name, frame in frames.items():
        glyphs = build_glyphs(frame)
        examples[name] = glyphs.to_dict()
    return examples


def main() -> None:
    examples = build_examples_dict()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(examples)} glyph set examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
