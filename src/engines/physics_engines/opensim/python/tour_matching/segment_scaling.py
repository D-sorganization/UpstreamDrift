"""Consistent OpenSim segment scaling module (OG-02, #10396).

Applies consistent physical and geometric scaling transformations to an OpenSim
model XML document:
1. Joint frames: Scales the translation vector of `PhysicalOffsetFrame` elements
   connected to scaled segments.
2. Bone visual meshes: Updates `<scale_factors>` on `<Mesh>` elements within
   `<attached_geometry>` for all affected segment bodies (e.g. humerus, radius,
   ulna, femur, tibia, calcn, torso).
3. Body center of mass (`mass_center`): Scales COM offset along the segment coordinates.
4. Mass & Inertia tensor:
   - `fixed_mass` policy (default): Mass $m' = m$; inertia $I' = s^2 I$ ($I'_{xx}, I'_{yy}, I'_{zz}$).
   - `density_preserving` policy: Mass $m' = s^3 m$; inertia $I' = s^5 I$.
5. Repeat-scaling protection: Embeds a `<ScalingMetadata>` record to guard against
   accidental compound double-scaling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path


import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml

from defusedxml import ElementTree as SafeET


from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (
    parse_model,
    write_model,
)
from src.shared.python.contracts import ensure, require


class ScalingPolicy(str, Enum):
    """Physical mass and inertia scaling policy."""

    FIXED_MASS = "fixed_mass"
    DENSITY_PRESERVING = "density_preserving"


# Body groups that share uniform scale factors from segment estimation
SEGMENT_BODY_MAP: Mapping[str, tuple[str, ...]] = {
    "humerus_r": ("humerus_r",),
    "humerus_l": ("humerus_l",),
    "radius_r": ("radius_r", "ulna_r"),
    "radius_l": ("radius_l", "ulna_l"),
    "femur_r": ("femur_r",),
    "femur_l": ("femur_l",),
    "tibia_r": ("tibia_r",),
    "tibia_l": ("tibia_l",),
    "calcn_r": ("calcn_r",),
    "calcn_l": ("calcn_l",),
    "torso": ("torso",),
}


@dataclass(frozen=True)
class SegmentScalingReport:
    """Receipt of applied segment scaling transformations."""

    model_path: str
    output_path: str
    policy: str
    applied_scales: dict[str, float]
    bodies_scaled: tuple[str, ...]
    meshes_updated: int
    frames_updated: int


def _parse_vec(text: str | None) -> list[float]:
    if not text or not text.strip():
        return [0.0, 0.0, 0.0]
    parts = text.strip().split()
    return [float(p) for p in parts]


def _format_vec(values: Sequence[float]) -> str:
    return " ".join(f"{v:.8g}" for v in values)


def apply_segment_scaling(
    model_source: Path | str | ET.ElementTree,
    scale_factors: Mapping[str, float],
    *,
    out_path: Path | str | None = None,
    policy: ScalingPolicy | str = ScalingPolicy.FIXED_MASS,
) -> Path:
    """Apply consistent anatomical and physical scaling to an OpenSim model.

    Preconditions:
    - Scale factors must be positive finite floats.
    - Model must not already be scaled (repeat-scaling protection).
    - Policy must be a valid ScalingPolicy.

    Postconditions:
    - Output model written to disk with updated joint frames, meshes, COM, and inertia.
    - Output model marked with scaling metadata.
    """
    if isinstance(policy, str):
        policy = ScalingPolicy(policy)

    # Validate scale factors
    require(len(scale_factors) > 0, "At least one scale factor must be provided")
    for seg, s in scale_factors.items():
        require(
            math.isfinite(s) and s > 0.0,
            f"Scale factor for {seg} must be a positive finite float; got {s}",
        )

    if isinstance(model_source, ET.ElementTree):
        tree = model_source
    else:
        src_path = Path(model_source)
        require(src_path.is_file(), f"Model file not found: {src_path}")
        tree = parse_model(src_path)

    root = tree.getroot()
    require(root is not None, "Model document has no root element")
    assert root is not None

    model_elem = root.find("Model")
    if model_elem is None and root.tag == "Model":
        model_elem = root
    require(model_elem is not None, "Document has no Model element")
    assert model_elem is not None

    # Check repeat scaling protection
    if model_elem.find(".//ScalingMetadata") is not None:
        raise ValueError(
            "Model document is already scaled (ScalingMetadata element present). "
            "Refusing repeat scaling to prevent corrupt compounding."
        )

    # Resolve target body scales from segment scale factors
    body_scales: dict[str, float] = {}
    for seg, s in scale_factors.items():
        if seg in SEGMENT_BODY_MAP:
            for bname in SEGMENT_BODY_MAP[seg]:
                body_scales[bname] = s
        else:
            body_scales[seg] = s

    # 1. Scale Joint PhysicalOffsetFrame translations
    frames_updated = 0
    for joint in root.findall(".//JointSet/objects/*"):
        for frame in joint.findall("frames/PhysicalOffsetFrame"):
            parent_elem = frame.find("socket_parent")
            trans_elem = frame.find("translation")
            if (
                parent_elem is not None
                and parent_elem.text
                and trans_elem is not None
                and trans_elem.text
            ):
                parent_path = parent_elem.text.strip()
                bname = parent_path.split("/")[-1]
                if bname in body_scales:
                    s = body_scales[bname]
                    coords = _parse_vec(trans_elem.text)
                    scaled_coords = [c * s for c in coords]
                    trans_elem.text = _format_vec(scaled_coords)
                    frames_updated += 1

    # 2. Scale Attached Meshes, Body COM, Mass, and Inertia
    meshes_updated = 0
    bodies_scaled: list[str] = []

    for body in root.findall(".//BodySet/objects/Body"):
        bname = body.get("name", "")
        if bname in body_scales:
            s = body_scales[bname]
            bodies_scaled.append(bname)

            # Update mesh scale factors
            for mesh in body.findall(".//Mesh"):
                scale_elem = mesh.find("scale_factors")
                if scale_elem is None:
                    scale_elem = ET.SubElement(mesh, "scale_factors")
                current_scales = (
                    _parse_vec(scale_elem.text) if scale_elem.text else [1.0, 1.0, 1.0]
                )
                new_scales = [c * s for c in current_scales]
                scale_elem.text = _format_vec(new_scales)
                meshes_updated += 1

            # Update Body center of mass (mass_center)
            mc_elem = body.find("mass_center")
            if mc_elem is not None and mc_elem.text:
                coords = _parse_vec(mc_elem.text)
                mc_elem.text = _format_vec([c * s for c in coords])

            # Update Mass & Inertia according to policy
            if policy == ScalingPolicy.FIXED_MASS:
                inertia_scale = s * s
            elif policy == ScalingPolicy.DENSITY_PRESERVING:
                mass_elem = body.find("mass")
                if mass_elem is not None and mass_elem.text:
                    mass_elem.text = f"{float(mass_elem.text) * (s**3):.8g}"
                inertia_scale = s**5
            else:
                inertia_scale = s * s

            inertia_elem = body.find("inertia")
            if inertia_elem is not None and inertia_elem.text:
                vals = _parse_vec(inertia_elem.text)
                # Format: Ixx Iyy Izz Ixy Ixz Iyz
                scaled_vals = [v * inertia_scale for v in vals]
                inertia_elem.text = _format_vec(scaled_vals)

    # 3. Add ScalingMetadata record to document
    meta = ET.SubElement(model_elem, "ScalingMetadata")
    ET.SubElement(meta, "policy").text = policy.value
    scales_sub = ET.SubElement(meta, "scale_factors")
    for k, v in scale_factors.items():
        ET.SubElement(scales_sub, k).text = f"{v:.8g}"

    if out_path is None:
        if isinstance(model_source, (str, Path)):
            out_target = Path(model_source)
        else:
            raise ValueError(
                "out_path must be specified when model_source is an ElementTree"
            )
    else:
        out_target = Path(out_path)

    saved_path = write_model(tree, out_target)
    ensure(saved_path.is_file(), "Scaled model must exist after writing")
    return saved_path
