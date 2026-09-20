"""Build the anthropometric full-body candidate document (AN-1, #10099).

Upper body from ``anthropometric_geometry.build_upper_body`` (subject stature
and mass), Rajagopal legs through the same ``leg_extension`` as the qualified
builder with a fixed pelvis alignment (OpenSim pelvis axes onto the pelvis
frame, hip centres at the subject's hip half width), toe spheres and the
stiffer contact law of the ground-support driver, and anatomical seed
offsets for every marker (the calibration refines them). Writes
``full_body_spec_anthro_v1.json`` and ``build_receipt_anthro_v1.json``.
The result is unqualified until Simscape carries the same geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_full_body_spec import (  # noqa: E402
    LEG_MARKER_BODIES,
    leg_extension,
    read_osim,
    transform,
)

from src.shared.python.motion_matching.anthropometry import (  # noqa: E402
    de_leva_table_dict,
    de_leva_table_sha256,
)
from src.shared.python.motion_matching.anthropometric_geometry import (  # noqa: E402
    GRIP_ROTATION_DEG,
    LEG_VISUAL_RADIUS_M,
    build_upper_body,
    pelvis_alignment_for,
)
from src.shared.python.motion_matching.club_models import CLUBS  # noqa: E402
from src.shared.python.motion_matching.range_of_motion import (  # noqa: E402
    HUMAN_RANGES_DEG,
    as_document,
)
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    ContactSpec,
    ContactSphere,
    MarkerAttachment,
    canonical_sha256,
    derive_full_body_spec,
    save_full_body_spec,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    MARKER_SEGMENTS,
    tracked_labels,
)

# Calibrated contact parameters from MS-20 (#10335) identifiability sweep
# Evidence receipt: docs/development/full_body_models/evidence/contact_id/receipt.json
CONTACT_RECEIPT_SHA256 = (
    "9bbe5fc4e2fef05d0160af03f1214cc2768068fe0c6f3c71e9cae11c4a7e9556"
)
CONTACT_STIFFNESS_N_M = 1.0e5
CONTACT_DISSIPATION_S_M = 0.5
CONTACT_STATIC_FRICTION = 0.6
CONTACT_DYNAMIC_FRICTION = 0.4
CONTACT_VISCOUS_FRICTION = 0.01
CONTACT_TRANSITION_VELOCITY_M_S = 0.01
SPHERES = (
    ("heel", (0.01, -0.005, 0.0), 0.035),
    ("forefoot", (0.16, -0.005, 0.0), 0.03),
    ("toe", (0.23, -0.010, 0.0), 0.025),
)


def marker_seeds(
    hub_height_m: float,
    clav_m: float,
    upper_arm_m: float,
    forearm_m: float,
    native: dict,
) -> dict[str, MarkerAttachment]:
    """Anatomical seed offsets in the new frames (x forward, y left, z up)."""
    club = {
        label: MarkerAttachment(native_body, tuple(offset))
        for label, native_body, offset in (
            (lbl, native["marker_bodies"][i], native["marker_offsets_m"][i])
            for i, lbl in enumerate(native["marker_labels"])
        )
        if native_body == "Clubhead"
    }
    seeds: dict[str, MarkerAttachment] = {
        "WaistLeft": MarkerAttachment("Hip", (0.02, 0.15, 0.08)),
        "WaistRight": MarkerAttachment("Hip", (0.02, -0.15, 0.08)),
        "WaistLBack": MarkerAttachment("Hip", (-0.10, 0.07, 0.10)),
        "WaistRBack": MarkerAttachment("Hip", (-0.10, -0.07, 0.10)),
        "BackTop": MarkerAttachment("Spine", (-0.10, 0.0, hub_height_m + 0.02)),
        "BackLeft": MarkerAttachment("Spine", (-0.11, 0.09, hub_height_m * 0.7)),
        "BackRight": MarkerAttachment("Spine", (-0.11, -0.09, hub_height_m * 0.7)),
        "HeadTop": MarkerAttachment("Head", (0.0, 0.0, 0.24)),
        "HeadFront": MarkerAttachment("Head", (0.10, 0.0, 0.16)),
        "HeadSide": MarkerAttachment("Head", (0.0, 0.08, 0.16)),
        "LShoulderTop": MarkerAttachment("LScap", (0.0, clav_m, 0.05)),
        "LShoulderBack": MarkerAttachment("LScap", (-0.06, clav_m - 0.01, 0.02)),
        "RShoulderTop": MarkerAttachment("RScap", (0.0, -clav_m, 0.05)),
        "RShoulderBack": MarkerAttachment("RScap", (-0.06, -clav_m + 0.01, 0.02)),
        "LUArmHigh": MarkerAttachment("LS", (0.01, 0.04, -upper_arm_m * 0.35)),
        "LElbowOut": MarkerAttachment("LS", (0.0, 0.04, -upper_arm_m)),
        "RUArmHigh": MarkerAttachment("RS", (0.01, -0.04, -upper_arm_m * 0.35)),
        "RElbowOut": MarkerAttachment("RS", (0.0, -0.04, -upper_arm_m)),
        "LWristTop": MarkerAttachment("LF", (0.03, 0.0, -forearm_m / 2 + 0.01)),
        "RWristTop": MarkerAttachment("RF", (0.03, 0.0, -forearm_m / 2 + 0.01)),
        **club,
    }
    for label in MARKER_SEGMENTS["left_leg"] + MARKER_SEGMENTS["right_leg"]:
        side = "l" if label.startswith("L") else "r"
        seeds[label] = MarkerAttachment(f"{LEG_MARKER_BODIES[label[1:]]}_{side}", None)
    missing = set(tracked_labels()) - set(seeds)
    if missing:
        raise ValueError(f"No seed for {sorted(missing)}")
    return {label: seeds[label] for label in tracked_labels()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--osim", type=Path, required=True)
    parser.add_argument("--native-candidate", type=Path, required=True)
    parser.add_argument("--stature", type=float, required=True)
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trunk-scale", type=float, default=1.0)
    parser.add_argument("--arm-scale", type=float, default=1.0)
    parser.add_argument("--shoulder-scale", type=float, default=1.0)
    parser.add_argument("--club", choices=sorted(CLUBS), default="driver")
    parser.add_argument(
        "--grip-roll", type=float, default=0.0, help="hand roll about the shaft, deg"
    )
    parser.add_argument(
        "--lead-grip-rotation",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="lead hand rotation on the wrist, extrinsic x-y-z deg "
        "(default GRIP_ROTATION_DEG['L'], fitted from the matches)",
    )
    parser.add_argument(
        "--trail-grip-rotation",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="trail hand rotation on the wrist (default GRIP_ROTATION_DEG['R'])",
    )
    parser.add_argument(
        "--name", default=None, help="default full_body_spec_anthro_<club>"
    )
    args = parser.parse_args()
    if args.name is None:
        args.name = f"full_body_spec_anthro_{args.club}"
    native = json.loads(args.native.read_text())
    upper = build_upper_body(
        native,
        stature_m=args.stature,
        mass_kg=args.mass,
        trunk_scale=args.trunk_scale,
        arm_scale=args.arm_scale,
        shoulder_scale=args.shoulder_scale,
        club=CLUBS[args.club],
        grip_roll_deg=args.grip_roll,
        grip_rotation_deg={
            "L": args.lead_grip_rotation or GRIP_ROTATION_DEG["L"],
            "R": args.trail_grip_rotation or GRIP_ROTATION_DEG["R"],
        },
    )
    bodies, joints = read_osim(args.osim)
    rotation, hip_half = pelvis_alignment_for(args.stature)
    hips = np.array(
        [joints["hip_r"]["parent_translation"], joints["hip_l"]["parent_translation"]]
    )
    mid = rotation @ hips.mean(axis=0)
    alignment = transform(rotation, -mid)
    hip_frame = next(f for f in upper["frames"] if f["name"] == "Hip")
    extension, notes = leg_extension(
        bodies,
        joints,
        hip_frame["body"],
        np.asarray(hip_frame["placement"]) @ alignment,
    )
    # Hip centres at the subject's half width, symmetric about the pelvis centre.
    edges = []
    for joint in extension.joints:
        if joint.name in ("hip_r", "hip_l"):
            base = np.asarray(joint.parent_to_base, dtype=float)
            base[:3, 3] = [0.0, -hip_half if joint.name == "hip_r" else hip_half, 0.0]
            joint = type(joint)(
                joint.name,
                joint.parent,
                joint.child,
                base.tolist(),
                joint.child_to_follower,
                joint.primitives,
                joint.coordinates,
            )
        edges.append(joint)
    extension = type(extension)(
        extension.bodies,
        edges,
        extension.provenance
        + "; hip centres at the subject hip half width on the pelvis frame",
    )
    contact = ContactSpec(
        law="hunt_crossley_coulomb",
        parameters={
            "stiffness_n_m": CONTACT_STIFFNESS_N_M,
            "dissipation_s_m": CONTACT_DISSIPATION_S_M,
            "static_friction": CONTACT_STATIC_FRICTION,
            "dynamic_friction": CONTACT_DYNAMIC_FRICTION,
            "viscous_friction": CONTACT_VISCOUS_FRICTION,
            "transition_velocity_m_s": CONTACT_TRANSITION_VELOCITY_M_S,
        },
        spheres=tuple(
            ContactSphere(f"{kind}_{side}", f"calcn_{side}", position, radius)
            for side in ("r", "l")
            for kind, position, radius in SPHERES
        ),
        ground_normal_policy="opposite_gravity",
        ground_height_m=None,
        provenance=(
            f"MS-20 (#10335) calibrated contact parameters; receipt SHA256 {CONTACT_RECEIPT_SHA256[:16]}"
        ),
    )
    hub_frame = next(f for f in upper["frames"] if f["name"] == "Hub")
    hub_h = float(np.asarray(hub_frame["placement"])[2, 3])
    clav = float(
        np.linalg.norm(
            np.asarray(
                next(j for j in upper["joints"] if j["child"].endswith("LUpperArm"))[
                    "parent_to_base"
                ]
            )[:3, 3]
        )
    )
    upper_arm = float(
        np.linalg.norm(
            np.asarray(
                next(
                    j for j in upper["joints"] if j["child"].endswith("Spherical Solid")
                )["parent_to_base"]
            )[:3, 3]
        )
    )
    forearm = 2 * float(
        np.linalg.norm(
            np.asarray(
                next(
                    j for j in upper["joints"] if j["child"].endswith("LLowerForearm")
                )["parent_to_base"]
            )[:3, 3]
        )
    )
    candidate = json.loads(args.native_candidate.read_text())
    markers = marker_seeds(hub_h, clav, upper_arm, forearm, candidate)
    document = derive_full_body_spec(
        upper,
        extension,
        contact,
        markers,
        provenance=(
            f"anthropometric full-body candidate v1 at {args.stature:.3f} m, {args.mass:.1f} kg "
            "(build_anthropometric_spec.py; AN-1 #10099); upper body unqualified; "
            "marker offsets are anatomical seeds to be calibrated"
        ),
    )
    document["subject"] = upper["subject"]
    document["de_leva_table_sha256"] = de_leva_table_sha256()
    document["anthropometry"] = {
        "table": "de_leva_1996_male",
        "de_leva_table_sha256": de_leva_table_sha256(),
        "segments": de_leva_table_dict(),
    }
    document["coordinate_ranges_deg"] = {
        **upper["coordinate_ranges_deg"],
        **as_document(
            {k: v for k, v in HUMAN_RANGES_DEG.items() if k[-2:] in ("_r", "_l")}
        ),
    }
    document["address_seed_deg"] = upper["address_seed_deg"]
    document["club"] = upper["club"]
    document["visual_hints"] = upper["visual_hints"]
    document["visual_hints"]["capsule_radius_m"].update(
        {
            f"{segment}_{side}": LEG_VISUAL_RADIUS_M[segment]
            for segment in LEG_VISUAL_RADIUS_M
            for side in ("r", "l")
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    spec_path = save_full_body_spec(document, args.output / f"{args.name}.json")
    receipt = {
        "spec_sha256": canonical_sha256(document),
        "spec_file_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "de_leva_table_sha256": de_leva_table_sha256(),
        "subject": upper["subject"],
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.native, args.osim, args.native_candidate)
        },
        "hip_half_width_m": hip_half,
        "hub_height_above_l5s1_m": hub_h,
        "total_mass_kg": float(
            sum(s["mass_kg"] for b in document["bodies"] for s in b["solids"])
        ),
        "simplifications": notes,
        "qualification": "anthropometric geometry; unqualified until Simscape parity (AN-1)",
    }
    (
        args.output / f"build_receipt_{args.name.replace('full_body_spec_', '')}.json"
    ).write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
