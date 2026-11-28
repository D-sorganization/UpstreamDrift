"""Calibrate neutral shoe-to-head visual height for a sized candidate.

Height is the world-z span from the lowest procedural sole mesh vertex to the
CC0 skin's highest vertex, at zero joint coordinates, with the default head
sizing and no hair or cap. This is an operational visual convention including
shoes, not a measured barefoot stature or evidence of individual anatomy.
Changing trunk scale changes the plant; captured motion needs a new fit.
The spec anthropometry segments are the reference table, not resolved lengths;
selected leg priors are reported separately in the calibration receipt.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import brentq

from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.foot import foot_parts
from src.shared.python.model_appearance.head import (
    build_head_parts,
    drawn_head_length,
    head_frame_in_body,
    resolve_head_anchor,
)
from src.shared.python.model_appearance.schema import HeadSettings
from src.shared.python.model_appearance.skinned_body import spec_body_poses


def neutral_visual_height(spec: Mapping[str, Any]) -> float:
    """Return the native shared head/sole vertical span at all-zero coordinates."""
    poses = spec_body_poses(spec)
    anchor = resolve_head_anchor(spec)
    if anchor is None:
        raise ValueError("a head anchor is required to measure visual stature")
    head = HeadSettings(headwear="none")
    parts = build_head_parts(drawn_head_length(spec, anchor, head), head)
    skin = next(part.mesh.vertices for part in parts if part.name == "skin_mesh")
    pose = poses[anchor.body] @ head_frame_in_body(anchor, head)
    top = float((skin @ pose[:3, :3].T + pose[:3, 3])[:, 2].max())
    soles = []
    for body, items in foot_parts(spec, body_sections(spec)).items():
        for item in items:
            if item.role == "sole":
                pose = poses[body]
                world = item.mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
                soles.append(float(world[:, 2].min()))
    if not soles:
        raise ValueError("procedural soles are required to measure visual stature")
    height = top - min(soles)
    if not math.isfinite(height) or height <= 0:
        raise ValueError("neutral visual stature must be positive and finite")
    return height


def build_stature_calibrated(
    stature_m: float, mass_kg: float, club: str = "driver", *, frame_build: float = 0.0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build current golfer with nominal mass, de Leva legs and calibrated trunk.

    Bracket [0.6, 1.2] prevents unconstrained anatomical extrapolation; a
    target outside it raises instead of silently producing a wrong model.
    The residual and chosen scale are returned for provenance.
    """
    if not all(math.isfinite(v) and v > 0 for v in (stature_m, mass_kg)):
        raise ValueError("stature and mass must be positive and finite")
    from src.shared.python.motion_matching.anthropometry import segment_parameters
    from src.shared.python.motion_matching.execution.spec_builder import (
        compose_anthropometric_document,
    )
    from src.shared.python.motion_matching.execution.spec_revisions import (
        with_current_golfer_model,
        with_grip_revision,
    )
    from src.shared.python.model_appearance.canonical_golfer_gate import (
        require_canonical_golfer,
    )
    from src.shared.python.model_appearance.player_frame_inertia import (
        apply_frame_inertia,
        validate_frame_build,
    )

    validate_frame_build(frame_build)

    lengths = {
        key: segment_parameters(stature_m, mass_kg, key).length_m
        for key in ("thigh", "shank")
    }
    count = 0

    def build(scale: float) -> Any:
        """Build a candidate with fixed leg priors at this trunk scale."""
        return compose_anthropometric_document(
            stature_m=stature_m,
            mass_kg=mass_kg,
            club=club,
            match_subject_mass=True,
            trunk_scale=scale,
            thigh_length_m=lengths["thigh"],
            shank_length_m=lengths["shank"],
        )

    def residual(scale: float) -> float:
        """Measure signed height error and count bracket evaluations."""
        nonlocal count
        count += 1
        return neutral_visual_height(build(scale).document) - stature_m

    scale = brentq(residual, 0.6, 1.2, xtol=1e-7, maxiter=32)
    built = build(scale)
    document = apply_frame_inertia(built.document, frame_build)
    spec, _ = with_current_golfer_model(
        *with_grip_revision(document, built.subject, None)
    )
    require_canonical_golfer(spec)
    height = neutral_visual_height(spec)
    if abs(height - stature_m) > 0.0001:
        raise ValueError("current-model revision changed the calibrated stature")
    receipt = {
        "trunk_scale": scale,
        "selected_leg_lengths_m": lengths,
        "neutral_visual_height_m": height,
        "residual_m": height - stature_m,
        "iterations": count,
        "convention": "zero-q CC0 skin top to lowest procedural shoe sole; shoes included",
    }
    if frame_build:
        receipt["frame_inertia"] = {
            "frame_build": frame_build,
            "transverse_scale": 1 + 0.18 * frame_build,
            "method": "affine torso and limb second-moment estimate; fixed mass and COM",
        }
    return spec, receipt
