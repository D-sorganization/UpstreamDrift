"""Build the anthropometric full-body candidate document (AN-1, #10099, #10520).

Upper body from ``anthropometric_geometry.build_upper_body`` (subject stature
and mass), Rajagopal legs through ``leg_extension`` with fixed pelvis alignment
(OpenSim pelvis axes onto the pelvis frame, hip centres at subject hip half width),
toe spheres, calibrated contact law, and anatomical seed offsets.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree as ET

from src.shared.python.motion_matching.anthropometric_geometry import (
    GRIP_ROTATION_DEG,
    LEG_VISUAL_RADIUS_M,
    build_upper_body,
    pelvis_alignment_for,
)
from src.shared.python.motion_matching.anthropometry import (
    de_leva_table_dict,
    de_leva_table_sha256,
)
from src.shared.python.motion_matching.club_models import CLUBS
from src.shared.python.motion_matching.full_body_spec import (
    BodySpec,
    ContactSpec,
    ContactSphere,
    JointSpec,
    LowerLimbExtension,
    MarkerAttachment,
    canonical_sha256,
    derive_full_body_spec,
    save_full_body_spec,
)
from src.shared.python.motion_matching.range_of_motion import (
    HUMAN_RANGES_DEG,
    as_document,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    tracked_labels,
)

# Calibrated contact parameters from MS-20 (#10335) identifiability sweep
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

LEG_BODIES = ("femur", "tibia", "talus", "calcn", "toes")
LEG_MARKER_BODIES = {
    "KneeOut": "femur",
    "AnkleOut": "tibia",
    "ToeIn": "calcn",
    "ToeOut": "calcn",
}

HIP_PERMUTATION = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
KNEE_PERMUTATION = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


def body_fixed_xyz(angles: np.ndarray) -> np.ndarray:
    """SimTK BodyFixed XYZ Euler rotation: R = Rx(a) Ry(b) Rz(c)."""
    a, b, c = angles

    def rx(t: float) -> np.ndarray:
        return np.array(
            [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
        )

    def ry(t: float) -> np.ndarray:
        return np.array(
            [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]
        )

    def rz(t: float) -> np.ndarray:
        return np.array(
            [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )

    return rx(a) @ ry(b) @ rz(c)


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def read_osim(osim: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model = ET.parse(str(osim)).getroot().find("Model")
    if model is None:
        raise ValueError(f"No Model tag found in OpenSim file {osim}")
    bodies = {}
    for body in model.findall("BodySet/objects/Body"):
        bodies[body.get("name")] = {
            "mass": float(body.findtext("mass")),  # type: ignore[arg-type]
            "com": np.array(body.findtext("mass_center").split(), dtype=float),  # type: ignore[union-attr]
            "inertia": np.array(body.findtext("inertia").split(), dtype=float),  # type: ignore[union-attr]
        }
    joints = {}
    for joint in model.findall("JointSet/objects/*"):
        frames = joint.findall("frames/PhysicalOffsetFrame")
        parent, child = frames[0], frames[1]
        joints[joint.get("name")] = {
            "parent_body": parent.findtext("socket_parent").rsplit("/", 1)[1],  # type: ignore[union-attr]
            "child_body": child.findtext("socket_parent").rsplit("/", 1)[1],  # type: ignore[union-attr]
            "parent_translation": np.array(
                parent.findtext("translation").split(),
                dtype=float,  # type: ignore[union-attr]
            ),
            "parent_orientation": np.array(
                parent.findtext("orientation").split(),
                dtype=float,  # type: ignore[union-attr]
            ),
            "child_translation": np.array(
                child.findtext("translation").split(),
                dtype=float,  # type: ignore[union-attr]
            ),
            "child_orientation": np.array(
                child.findtext("orientation").split(),
                dtype=float,  # type: ignore[union-attr]
            ),
        }
    return bodies, joints


def _frame(entry: dict[str, Any], which: str) -> np.ndarray:
    return transform(
        body_fixed_xyz(entry[f"{which}_orientation"]), entry[f"{which}_translation"]
    )


def leg_extension(
    bodies: dict[str, Any],
    joints: dict[str, Any],
    pelvis_body: str,
    hip_frame_in_pelvis: np.ndarray,
) -> tuple[LowerLimbExtension, list[str]]:
    specs: list[BodySpec] = []
    edges: list[JointSpec] = []
    notes = [
        "patella bodies and patellofemoral coupler constraints dropped",
        "walker knee reduced to a hinge about its primary axis; coupled spline "
        "translations ignored",
        "OpenSim body inertia interpreted as about the mass centre in the body frame",
    ]
    for side in ("r", "l"):
        for name in LEG_BODIES:
            b = bodies[f"{name}_{side}"]
            specs.append(
                BodySpec(
                    f"{name}_{side}", b["mass"], tuple(b["com"]), tuple(b["inertia"])
                )
            )
        hip = joints[f"hip_{side}"]
        edges.append(
            JointSpec(
                f"hip_{side}",
                pelvis_body,
                f"femur_{side}",
                (
                    hip_frame_in_pelvis
                    @ _frame(hip, "parent")
                    @ transform(HIP_PERMUTATION, np.zeros(3))
                ).tolist(),
                (
                    _frame(hip, "child") @ transform(HIP_PERMUTATION, np.zeros(3))
                ).tolist(),
                ("Rx", "Ry", "Rz"),
                (
                    f"hip_flexion_{side}",
                    f"hip_adduction_{side}",
                    f"hip_rotation_{side}",
                ),
            )
        )
        knee = joints[f"walker_knee_{side}"]
        edges.append(
            JointSpec(
                f"knee_{side}",
                f"femur_{side}",
                f"tibia_{side}",
                (
                    _frame(knee, "parent") @ transform(KNEE_PERMUTATION, np.zeros(3))
                ).tolist(),
                (
                    _frame(knee, "child") @ transform(KNEE_PERMUTATION, np.zeros(3))
                ).tolist(),
                ("Rz",),
                (f"knee_angle_{side}",),
            )
        )
        for joint_name, parent, child, coordinate in (
            (f"ankle_{side}", f"tibia_{side}", f"talus_{side}", f"ankle_angle_{side}"),
            (
                f"subtalar_{side}",
                f"talus_{side}",
                f"calcn_{side}",
                f"subtalar_angle_{side}",
            ),
            (f"mtp_{side}", f"calcn_{side}", f"toes_{side}", f"mtp_angle_{side}"),
        ):
            pin = joints[joint_name]
            edges.append(
                JointSpec(
                    joint_name,
                    parent,
                    child,
                    _frame(pin, "parent").tolist(),
                    _frame(pin, "child").tolist(),
                    ("Rz",),
                    (coordinate,),
                )
            )
    return (
        LowerLimbExtension(
            specs,
            edges,
            provenance="Rajagopal et al. 2016 via golf_humanoid.osim (OpenSense variant); "
            + "; ".join(notes),
        ),
        notes,
    )


def marker_seeds(
    hub_height_m: float,
    clav_m: float,
    upper_arm_m: float,
    forearm_m: float,
    native: dict[str, Any],
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


def _build_aligned_leg_extension(
    osim_path: Path,
    stature_m: float,
    hip_frame: dict[str, Any],
) -> tuple[Any, list[str], float]:
    bodies, joints = read_osim(osim_path)
    rotation, hip_half = pelvis_alignment_for(stature_m)
    hips = np.array(
        [joints["hip_r"]["parent_translation"], joints["hip_l"]["parent_translation"]]
    )
    mid = rotation @ hips.mean(axis=0)
    alignment = transform(rotation, -mid)
    extension, notes = leg_extension(
        bodies,
        joints,
        hip_frame["body"],
        np.asarray(hip_frame["placement"]) @ alignment,
    )

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
    return extension, notes, hip_half


def _create_contact_spec() -> ContactSpec:
    return ContactSpec(
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


def _extract_marker_seeds(
    upper: dict[str, Any], native_candidate_path: Path
) -> tuple[float, dict[str, Any]]:
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
    candidate = json.loads(native_candidate_path.read_text(encoding="utf-8"))
    markers = marker_seeds(hub_h, clav, upper_arm, forearm, candidate)
    return hub_h, markers


def _write_outputs(
    document: dict[str, Any],
    output_dir: Path,
    spec_name: str,
    subject: dict[str, Any],
    input_paths: tuple[Path, ...],
    hip_half: float,
    hub_h: float,
    notes: list[str],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = save_full_body_spec(document, output_dir / f"{spec_name}.json")
    receipt = {
        "spec_sha256": canonical_sha256(document),
        "spec_file_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "de_leva_table_sha256": de_leva_table_sha256(),
        "subject": subject,
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths
        },
        "hip_half_width_m": hip_half,
        "hub_height_above_l5s1_m": hub_h,
        "total_mass_kg": float(
            sum(s["mass_kg"] for b in document["bodies"] for s in b["solids"])
        ),
        "simplifications": notes,
        "qualification": "anthropometric geometry; unqualified until Simscape parity (AN-1)",
    }
    receipt_suffix = spec_name.replace("full_body_spec_", "")
    receipt_path = output_dir / f"build_receipt_{receipt_suffix}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return spec_path, receipt_path


def build_anthropometric_spec(
    *,
    native_path: Path,
    osim_path: Path,
    native_candidate_path: Path,
    stature_m: float,
    mass_kg: float,
    output_dir: Path,
    club: str = "driver",
    **kwargs: Any,
) -> tuple[Path, Path]:
    """Build candidate spec and receipt, writing both to output_dir."""
    trunk_scale = float(kwargs.get("trunk_scale", 1.0))
    arm_scale = float(kwargs.get("arm_scale", 1.0))
    shoulder_scale = float(kwargs.get("shoulder_scale", 1.0))
    grip_roll_deg = float(kwargs.get("grip_roll_deg", 0.0))
    lead_grip_rotation = kwargs.get("lead_grip_rotation")
    trail_grip_rotation = kwargs.get("trail_grip_rotation")
    name = kwargs.get("name")
    spec_name = name or f"full_body_spec_anthro_{club}"

    native = json.loads(native_path.read_text(encoding="utf-8"))
    upper = build_upper_body(
        native,
        stature_m=stature_m,
        mass_kg=mass_kg,
        trunk_scale=trunk_scale,
        arm_scale=arm_scale,
        shoulder_scale=shoulder_scale,
        club=CLUBS[club],
        grip_roll_deg=grip_roll_deg,
        grip_rotation_deg={
            "L": lead_grip_rotation or GRIP_ROTATION_DEG["L"],
            "R": trail_grip_rotation or GRIP_ROTATION_DEG["R"],
        },
    )

    hip_frame = next(f for f in upper["frames"] if f["name"] == "Hip")
    extension, notes, hip_half = _build_aligned_leg_extension(
        osim_path, stature_m, hip_frame
    )
    contact = _create_contact_spec()
    hub_h, markers = _extract_marker_seeds(upper, native_candidate_path)

    document = derive_full_body_spec(
        upper,
        extension,
        contact,
        markers,
        provenance=(
            f"anthropometric full-body candidate v1 at {stature_m:.3f} m, {mass_kg:.1f} kg "
            "(spec_builder.py; AN-1 #10099); upper body unqualified; "
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

    return _write_outputs(
        document=document,
        output_dir=output_dir,
        spec_name=spec_name,
        subject=upper["subject"],
        input_paths=(native_path, osim_path, native_candidate_path),
        hip_half=hip_half,
        hub_h=hub_h,
        notes=notes,
    )


def build_parser() -> argparse.ArgumentParser:
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
        help="lead hand rotation on wrist extrinsic x-y-z deg",
    )
    parser.add_argument(
        "--trail-grip-rotation",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="trail hand rotation on wrist",
    )
    parser.add_argument(
        "--name", default=None, help="default full_body_spec_anthro_<club>"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    build_anthropometric_spec(
        native_path=args.native,
        osim_path=args.osim,
        native_candidate_path=args.native_candidate,
        stature_m=args.stature,
        mass_kg=args.mass,
        output_dir=args.output,
        trunk_scale=args.trunk_scale,
        arm_scale=args.arm_scale,
        shoulder_scale=args.shoulder_scale,
        club=args.club,
        grip_roll_deg=args.grip_roll,
        lead_grip_rotation=(
            tuple(args.lead_grip_rotation) if args.lead_grip_rotation else None
        ),
        trail_grip_rotation=(
            tuple(args.trail_grip_rotation) if args.trail_grip_rotation else None
        ),
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
