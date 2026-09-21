"""Build full_body_spec_v1.json: qualified upper body plus Rajagopal lower limbs.

Sources (all recorded with hashes in the receipt):
- the qualified native upper-body spec (native_geometry_spec_9967.json);
- lower-limb masses, centres of mass, inertias and joint frames read from the
  packaged golf_humanoid.osim (Rajagopal et al. 2016, OpenSense variant);
- the native candidate's waist marker offsets in the native "Hip" frame and
  the OS-3 OpenSim pelvis-frame offsets of the same four markers, which give
  the pelvis alignment between the two models by rigid registration.

Simplifications, stated once here and in the document provenance: the knee is
a pure hinge about the Rajagopal walker-knee primary axis (the coupled spline
translations and the patella bodies are dropped); OpenSim body inertias are
taken as about the mass centre in the body frame; contact sphere placements
are placeholders sized from the calcaneus geometry and the ground height is
left uncalibrated for FB-4. No engine is built here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from defusedxml import ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    BodySpec,
    ContactSpec,
    ContactSphere,
    JointSpec,
    LowerLimbExtension,
    MarkerAttachment,
    canonical_sha256,
    derive_full_body_spec,
    pelvis_alignment,
    save_full_body_spec,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    MARKER_SEGMENTS,
    tracked_labels,
)

LEG_BODIES = ("femur", "tibia", "talus", "calcn", "toes")
WAIST = ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack")
LEG_MARKER_BODIES = {
    "KneeOut": "femur",
    "AnkleOut": "tibia",
    "ToeIn": "calcn",
    "ToeOut": "calcn",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


# Permutations Q (det +1) whose COLUMNS are the OpenSim axes our primitives turn
# about: the base frame is ``offset_frame @ Q`` so primitive Rx' turns about
# Q[:, 0], Ry' about Q[:, 1] and Rz' about Q[:, 2].
# Hip (OpenSim sequence Z, X, Y): Rx' about z, Ry' about x, Rz' about y.
HIP_PERMUTATION = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
# Walker knee: the single primitive Rz' turns about the offset frame x axis.
KNEE_PERMUTATION = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
# v1 (spec SHA 06272a18...) had these two matrices swapped, so hip flexion turned
# about the femur's long axis and the knee about the femur's y axis; v2 fixes it.


def read_osim(osim: Path) -> tuple[dict, dict]:
    model = ET.parse(str(osim)).getroot().find("Model")
    bodies = {}
    for body in model.findall("BodySet/objects/Body"):
        bodies[body.get("name")] = {
            "mass": float(body.findtext("mass")),
            "com": np.array(body.findtext("mass_center").split(), dtype=float),
            "inertia": np.array(body.findtext("inertia").split(), dtype=float),
        }
    joints = {}
    for joint in model.findall("JointSet/objects/*"):
        frames = joint.findall("frames/PhysicalOffsetFrame")
        parent, child = frames[0], frames[1]
        joints[joint.get("name")] = {
            "parent_body": parent.findtext("socket_parent").rsplit("/", 1)[1],
            "child_body": child.findtext("socket_parent").rsplit("/", 1)[1],
            "parent_translation": np.array(
                parent.findtext("translation").split(), dtype=float
            ),
            "parent_orientation": np.array(
                parent.findtext("orientation").split(), dtype=float
            ),
            "child_translation": np.array(
                child.findtext("translation").split(), dtype=float
            ),
            "child_orientation": np.array(
                child.findtext("orientation").split(), dtype=float
            ),
        }
    return bodies, joints


def frame(entry: dict, which: str) -> np.ndarray:
    return transform(
        body_fixed_xyz(entry[f"{which}_orientation"]), entry[f"{which}_translation"]
    )


def leg_extension(
    bodies: dict, joints: dict, pelvis_body: str, hip_frame_in_pelvis: np.ndarray
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
                    @ frame(hip, "parent")
                    @ transform(HIP_PERMUTATION, np.zeros(3))
                ).tolist(),
                (
                    frame(hip, "child") @ transform(HIP_PERMUTATION, np.zeros(3))
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
                    frame(knee, "parent") @ transform(KNEE_PERMUTATION, np.zeros(3))
                ).tolist(),
                (
                    frame(knee, "child") @ transform(KNEE_PERMUTATION, np.zeros(3))
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
                    frame(pin, "parent").tolist(),
                    frame(pin, "child").tolist(),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper", type=Path, required=True)
    parser.add_argument("--osim", type=Path, required=True)
    parser.add_argument("--native-candidate", type=Path, required=True)
    parser.add_argument("--os3-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upper = json.loads(args.upper.read_text())
    candidate = json.loads(args.native_candidate.read_text())
    os3 = json.loads(args.os3_receipt.read_text())
    bodies, joints = read_osim(args.osim)

    # Pelvis alignment: OpenSim pelvis frame -> native Hip frame -> LowerTorso body.
    native_offsets = {
        label: candidate["marker_offsets_m"][candidate["marker_labels"].index(label)]
        for label in WAIST
    }
    native_bodies = {
        candidate["marker_bodies"][candidate["marker_labels"].index(label)]
        for label in WAIST
    }
    if native_bodies != {"Hip"}:
        raise ValueError("Waist markers must all sit on the native Hip frame")
    opensim_offsets = {label: os3["offsets"][label]["offset_m"] for label in WAIST}
    hip_from_pelvis, residual = pelvis_alignment(native_offsets, opensim_offsets)
    hip_frame = next(f for f in upper["frames"] if f["name"] == "Hip")
    lower_torso_from_pelvis = np.asarray(hip_frame["placement"]) @ hip_from_pelvis
    extension, notes = leg_extension(
        bodies, joints, hip_frame["body"], lower_torso_from_pelvis
    )

    contact = ContactSpec(
        law="hunt_crossley_coulomb",
        parameters={
            "stiffness_n_m": 5.0e4,
            "dissipation_s_m": 1.0,
            "static_friction": 0.9,
            "dynamic_friction": 0.8,
            "viscous_friction": 0.0,
            "transition_velocity_m_s": 0.05,
        },
        spheres=tuple(
            ContactSphere(f"{kind}_{side}", f"calcn_{side}", position, radius)
            for side in ("r", "l")
            for kind, position, radius in (
                ("heel", (0.01, -0.005, 0.0), 0.035),
                ("forefoot", (0.16, -0.005, 0.0), 0.03),
            )
        ),
        ground_normal_policy="opposite_gravity",
        ground_height_m=None,
        provenance="placeholder parameters and sphere placements sized from Rajagopal "
        "calcaneus geometry (mtp joint at x=0.162 m); FB-3/FB-4 calibrate them",
    )

    markers: dict[str, MarkerAttachment] = {}
    for label in tracked_labels():
        if label in candidate["marker_labels"]:
            i = candidate["marker_labels"].index(label)
            markers[label] = MarkerAttachment(
                candidate["marker_bodies"][i], tuple(candidate["marker_offsets_m"][i])
            )
        elif label in MARKER_SEGMENTS["left_leg"] + MARKER_SEGMENTS["right_leg"]:
            side = "l" if label.startswith("L") else "r"
            body = LEG_MARKER_BODIES[label[1:]]
            markers[label] = MarkerAttachment(f"{body}_{side}", None)
        else:
            # Same segment as a native-attached marker (e.g. RShoulderTop, which
            # the native lane excluded for having only 128 valid samples).
            segment = next(
                g for g, labels in MARKER_SEGMENTS.items() if label in labels
            )
            sibling = next(
                (
                    sib
                    for sib in MARKER_SEGMENTS[segment]
                    if sib in candidate["marker_labels"]
                ),
                None,
            )
            if sibling is None:
                raise ValueError(f"No attachment source for tracked label {label}")
            body = candidate["marker_bodies"][candidate["marker_labels"].index(sibling)]
            markers[label] = MarkerAttachment(body, None)

    document = derive_full_body_spec(
        upper,
        extension,
        contact,
        markers,
        provenance=(
            "full-body-v1 built by build_full_body_spec.py from the qualified native "
            "upper-body spec and Rajagopal lower limbs; pelvis alignment by rigid "
            f"registration of four waist markers (RMS {residual:.4f} m)"
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    spec_path = save_full_body_spec(document, args.output / "full_body_spec_v2.json")
    receipt = {
        "spec_sha256": canonical_sha256(document),
        "spec_file_sha256": sha(spec_path),
        "inputs": {
            str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha(p)
            for p in (
                args.upper,
                args.osim,
                args.native_candidate,
                args.os3_receipt,
                Path(__file__),
            )
        },
        "native_candidate_sha256": candidate.get("source_sha256"),
        "coordinates": len(document["coordinate_order"]),
        "bodies": len(document["bodies"]),
        "joints": len(document["joints"]),
        "pelvis_alignment": {
            "hip_from_opensim_pelvis": hip_from_pelvis.tolist(),
            "rms_residual_m": residual,
            "native_offsets_hip_frame_m": native_offsets,
            "opensim_offsets_pelvis_frame_m": opensim_offsets,
            "note": "OS-3 offsets came from a 6.5 cm RMS unscaled fit; FB-4 recalibrates",
        },
        "simplifications": notes,
        "marker_attachments_with_offsets": sum(
            1
            for m in document["marker_attachments"].values()
            if m["offset_m"] is not None
        ),
    }
    (args.output / "build_receipt_v2.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
