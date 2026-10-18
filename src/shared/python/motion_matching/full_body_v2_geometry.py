"""Landmark geometry, mass and observability of the full-body-v2 trunk (HRG-D2).

HRG-D1 (#13170) declared the ``full-body-v2`` schema with stand-in joint
centres. This module gives the regional joints their landmark locations and
reports what the trunk's mass and the capture's back markers can and cannot
support. Frames: the pelvis frame at q = 0 (x forward, y left, z up), metres,
kilograms and radians unless a name says ``deg``.

Locations (:func:`landmark_trunk_locations`):

* **L5/S1** is the HRG-C3 (#13172) landmark: the hip-joint-centre (HJC)
  midpoint plus the design L5/S1 offset from it. A C3 revision records that
  offset in ``spec_revision``; an uncalibrated builder document has its
  pelvis origin at the design midpoint, so the offset is its pivot. A
  calibrated document that was never relocated (HJCs moved, pivot not) fails
  closed: run ``pelvis_frame.relocate_pelvis`` first.
* **C7/T1** is the neck joint centre, which the anthropometric builder places
  at the de Leva cervicale.
* **T12/L1** lies on the L5/S1 to C7/T1 line at the lumbar share of the
  thoracolumbar column length, 18 / (18 + 28) cm (Gray's Anatomy, 20th ed.,
  1918, "The Vertebral Column as a Whole": cervical 12.5, thoracic about 28,
  lumbar 18, sacrum and coccyx 12.5 cm in the male).

Neutral lordosis and kyphosis (:func:`neutral_curvature_report`): a lumped
regional joint represents each region by its chord. For a region of uniform
curvature the chord bisects the end tangents, so the thoracic chord is
extended relative to the lumbar chord by ``(LL - TK) / 2`` with LL the T12-S1
lordosis and TK the T1-T12 kyphosis. Reference: Yukawa et al. 2018 (626
asymptomatic adults), LL 49.7 +- 11.2 deg and TK 36.0 +- 10.1 deg.

Observability (:func:`trunk_observability`): the linearised map from the six
trunk coordinates to the capture's trunk back markers, with the pelvis held
(it is fixed by the waist markers). The ``bruno2015_regional`` prior enters
only as a soft regulariser (ADR-0055 item 7); a coordinate it dominates is
reported, never called measured.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.anthropometric_geometry import NECK_JOINT
from src.shared.python.motion_matching.full_body_migration import RegionalPrior
from src.shared.python.motion_matching.full_body_schema import require_full_body_schema
from src.shared.python.motion_matching.full_body_spec import V1_TRUNK
from src.shared.python.motion_matching.full_body_v2 import (
    V2_TRUNK,
    TrunkJointLocations,
    derive_full_body_v2_spec,
    trunk_bodies,
)
from src.shared.python.motion_matching.hip_centre_map import PRIOR_DOMINATED_WEIGHT
from src.shared.python.motion_matching.identifiability import (
    compute_spec_marker_positions,
    probe_synthetic_chain_identifiability,
)
from src.shared.python.motion_matching.tour_capture_contract import MARKER_SEGMENTS

Array = NDArray[np.float64]

__all__ = [
    "BACK_MARKERS",
    "EXPORT_STATUS",
    "GRAY_REGION_LENGTH_CM",
    "LUMBAR_LENGTH_FRACTION",
    "MassProperties",
    "YUKAWA_2018_DEG",
    "c7t1_landmark",
    "derive_landmark_v2_spec",
    "hjc_midpoint",
    "human_mass_ledger",
    "l5s1_landmark",
    "landmark_trunk_locations",
    "neutral_curvature_report",
    "spec_mass_properties",
    "trunk_observability",
]

HIP_JOINTS = ("hip_r", "hip_l")
C3_ISSUE = 13172
#: Gray's Anatomy (20th ed., 1918), male regional lengths of the column (cm).
GRAY_REGION_LENGTH_CM: Mapping[str, float] = MappingProxyType(
    {"lumbar": 18.0, "thoracic": 28.0}
)
LUMBAR_LENGTH_FRACTION = GRAY_REGION_LENGTH_CM["lumbar"] / (
    GRAY_REGION_LENGTH_CM["lumbar"] + GRAY_REGION_LENGTH_CM["thoracic"]
)
GRAY_REFERENCE = (
    "Gray H. Anatomy of the Human Body, 20th ed. (1918), The Vertebral Column "
    "as a Whole: male column 71 cm; cervical 12.5, thoracic about 28, lumbar "
    "18, sacrum and coccyx 12.5 cm"
)
#: Yukawa et al. 2018, Eur Spine J 27(2):426-432, doi:10.1007/s00586-016-4807-7:
#: mean and SD (deg) in 626 asymptomatic adults (abstract values).
YUKAWA_2018_DEG: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "lumbar_lordosis_T12_S1": (49.7, 11.2),
        "thoracic_kyphosis_T1_T12": (36.0, 10.1),
        "sacral_slope": (39.4, 8.0),
    }
)
YUKAWA_REFERENCE = (
    "Yukawa Y et al. Normative data for parameters of sagittal spinal "
    "alignment in healthy subjects (626 asymptomatic individuals). Eur Spine J "
    "27(2):426-432 (2018), doi:10.1007/s00586-016-4807-7"
)
#: The capture's trunk back markers (tour capture contract, segment "trunk").
BACK_MARKERS: tuple[str, ...] = tuple(MARKER_SEGMENTS["trunk"])
#: Waist-marker residual on the best driver capture (HRG-C3 reference), m.
DEFAULT_MARKER_NOISE_M = 0.0133
#: Declared SD of the regional-split regulariser ``D phi`` (not a measurement).
DEFAULT_PRIOR_SD_RAD = math.radians(5.0)
_RANK_RTOL = 1e-8
_NULL_SHARE = 1e-6

#: Export decision per engine for full-body-v2 (HRG-D2 acceptance item 5).
EXPORT_STATUS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "mujoco": {
            "status": "exported",
            "reason": "physics MJCF; the visual layer is HRG-D3",
        },
        "drake_urdf": {"status": "exported", "reason": "URDF bundle and plant"},
        "pinocchio": {
            "status": "exported",
            "reason": "full-body and regional-trunk upper-body models",
        },
        "opensim": {
            "status": "exported",
            "reason": "CustomJoint model; the muscle graft is deferred to HRG-Q1",
        },
        "myosuite": {
            "status": "mapped_thorax_only",
            "reason": (
                "MyoFullBody is reached by segment-orientation IK from the spec "
                "thorax; it has one lumbar joint and no counterpart of the "
                "trunk_lumbar body, so the regional lumbar/thoracic split is "
                "not carried (exempt); myosuite assets are optional"
            ),
        },
    }
)


def _vector(value: Any) -> Array:
    return np.asarray(value, dtype=float)[:3, 3]


def hjc_midpoint(spec: Mapping[str, Any]) -> Array:
    """Midpoint of the ``hip_r`` and ``hip_l`` joint centres in the pelvis frame.

    Raises ``ValueError`` unless both hip joints hang on the pelvis.
    """
    pelvis = trunk_bodies(spec)["pelvis"]
    hips = {j["name"]: j for j in spec["joints"] if j["name"] in HIP_JOINTS}
    if set(hips) != set(HIP_JOINTS) or any(
        j["parent"] != pelvis for j in hips.values()
    ):
        raise ValueError("hjc_midpoint needs hip_r and hip_l on the pelvis")
    return 0.5 * (
        _vector(hips["hip_r"]["parent_to_base"])
        + _vector(hips["hip_l"]["parent_to_base"])
    )


def _design_l5s1_offset(
    spec: Mapping[str, Any], mid: Array, pivot: Array, tolerance_m: float
) -> tuple[Array, str]:
    revision = spec.get("spec_revision")
    if isinstance(revision, Mapping) and revision.get("issue") == C3_ISSUE:
        offsets = revision["design_offsets_from_hjc_midpoint_m"]
        return np.asarray(offsets["l5s1"], dtype=float), (
            "HRG-C3 revision: HJC midpoint plus the recorded design L5/S1 offset"
        )
    if float(np.linalg.norm(mid)) <= tolerance_m:
        return pivot - mid, (
            "builder design: pelvis origin at the HJC midpoint, L5/S1 at the "
            "design offset (0.23 x trunk length) above it"
        )
    raise ValueError(
        f"The HJC midpoint is {np.round(mid, 4).tolist()} m from the pelvis "
        "origin but the document records no HRG-C3 relocation, so the L5/S1 "
        "design offset is unknown; apply pelvis_frame.relocate_pelvis first"
    )


def l5s1_landmark(
    spec: Mapping[str, Any], *, tolerance_m: float = 1e-9
) -> tuple[Array, str]:
    """L5/S1 in the pelvis frame from the HRG-C3 landmark definition.

    Returns ``(point, source)``. Raises ``ValueError`` for a calibrated but
    unrelocated document (see module docstring) or when the document's pivot
    is not at the landmark within ``tolerance_m``.
    """
    require_full_body_schema(spec, consumer="l5s1_landmark", accepted=("full-body-v1",))
    if not tolerance_m > 0.0:
        raise ValueError("tolerance_m must be positive")
    mid = hjc_midpoint(spec)
    pivot = _vector(V1_TRUNK.locate(spec)[0]["parent_to_base"])
    design, source = _design_l5s1_offset(spec, mid, pivot, tolerance_m)
    landmark = mid + design
    gap = float(np.linalg.norm(landmark - pivot))
    if gap > tolerance_m:
        raise ValueError(
            f"The L5/S1 pivot is {gap * 1e3:.3f} mm from its HRG-C3 landmark; "
            "apply pelvis_frame.relocate_pelvis first"
        )
    return landmark, source


def c7t1_landmark(spec: Mapping[str, Any]) -> Array:
    """C7/T1 in the pelvis frame at q = 0: the neck joint centre (cervicale).

    Raises ``ValueError`` unless exactly one neck joint hangs on the thorax.
    """
    roles = trunk_bodies(spec)
    necks = [j for j in spec["joints"] if j["name"] == NECK_JOINT]
    if len(necks) != 1 or necks[0]["parent"] != roles["thorax"]:
        raise ValueError(f"c7t1_landmark needs one {NECK_JOINT!r} on the thorax")
    poses = spec_body_poses(spec)
    pose = np.linalg.inv(poses[roles["pelvis"]]) @ poses[necks[0]["parent"]]
    return _vector(pose @ np.asarray(necks[0]["parent_to_base"], dtype=float))


def _tilts_deg(chord: Array) -> tuple[float, float]:
    """Forward (+x) and leftward (+y) tilt of a chord from the pelvis z axis."""
    return (
        math.degrees(math.atan2(float(chord[0]), float(chord[2]))),
        math.degrees(math.atan2(float(chord[1]), float(chord[2]))),
    )


def neutral_curvature_report(
    l5s1: ArrayLike, t12l1: ArrayLike, c7t1: ArrayLike
) -> dict[str, Any]:
    """Neutral chord angles of the two regions against Yukawa et al. 2018.

    ``thoracolumbar_chord_extension_deg`` (ISB sign, extension positive) is
    the thoracic chord relative to the lumbar chord in the sagittal plane;
    its reference is ``(LL - TK) / 2`` with SD ``hypot(sd_LL, sd_TK) / 2``
    (independence assumed, which overstates the SD for correlated LL and TK).
    The lumbar chord tilt is reported but not compared: the pelvis frame is
    the waist-marker plane, not the sacral endplate. Raises ``ValueError`` for
    degenerate chords.
    """
    points = [np.asarray(p, dtype=float) for p in (l5s1, t12l1, c7t1)]
    lumbar, thoracic = points[1] - points[0], points[2] - points[1]
    if min(np.linalg.norm(lumbar), np.linalg.norm(thoracic)) < 1e-3:
        raise ValueError("Trunk chords must be at least 1 mm long")
    lumbar_fwd, lumbar_left = _tilts_deg(lumbar)
    thoracic_fwd, thoracic_left = _tilts_deg(thoracic)
    ll, ll_sd = YUKAWA_2018_DEG["lumbar_lordosis_T12_S1"]
    tk, tk_sd = YUKAWA_2018_DEG["thoracic_kyphosis_T1_T12"]
    ss, ss_sd = YUKAWA_2018_DEG["sacral_slope"]
    model = -(thoracic_fwd - lumbar_fwd)
    mean, sd = 0.5 * (ll - tk), 0.5 * math.hypot(ll_sd, tk_sd)
    z = (model - mean) / sd
    return {
        "reference": YUKAWA_REFERENCE,
        "model": "uniform-curvature regions; a regional joint is its chord",
        "thoracolumbar_chord_extension_deg": {
            "model_deg": model,
            "reference_mean_deg": mean,
            "reference_sd_deg": sd,
            "z_score": z,
            "within_2sd": bool(abs(z) <= 2.0),
        },
        "lumbar_chord_forward_tilt_deg": {
            "model_deg": lumbar_fwd,
            "reference_lab_mean_deg": ss - 0.5 * ll,
            "reference_lab_sd_deg": math.hypot(ss_sd, 0.5 * ll_sd),
            "comparable": False,
            "reason": (
                "the pelvis frame is the waist-marker plane, not the sacral "
                "endplate, and q = 0 is a modelling neutral, not a measured "
                "standing posture"
            ),
        },
        "frontal_tilt_deg": {"lumbar": lumbar_left, "thoracic": thoracic_left},
    }


def landmark_trunk_locations(
    v1: Mapping[str, Any], *, tolerance_m: float = 1e-9
) -> TrunkJointLocations:
    """HRG-D2 landmark joint centres of the v2 trunk (see module docstring).

    Postcondition: ``evidence`` records the C7/T1 point, the L5/S1 source,
    the regional length fraction and :func:`neutral_curvature_report`.
    """
    l5s1, source = l5s1_landmark(v1, tolerance_m=tolerance_m)
    c7t1 = c7t1_landmark(v1)
    t12l1 = l5s1 + LUMBAR_LENGTH_FRACTION * (c7t1 - l5s1)
    evidence = {
        "l5s1_source": source,
        "c7t1_in_pelvis_m": c7t1.tolist(),
        "c7t1_source": "neck joint centre at the de Leva cervicale",
        "lumbar_length_fraction": LUMBAR_LENGTH_FRACTION,
        "regional_lengths_reference": GRAY_REFERENCE,
        "neutral_curvature": neutral_curvature_report(l5s1, t12l1, c7t1),
    }
    return TrunkJointLocations(
        tuple(l5s1.tolist()),
        tuple(t12l1.tolist()),
        "HRG-D2 landmarks: L5/S1 from the HRG-C3 HJC-midpoint design offset; "
        "T12/L1 on the L5/S1-C7/T1 line at the Gray's Anatomy lumbar share "
        "18/(18+28) of the thoracolumbar column",
        evidence,
    )


def derive_landmark_v2_spec(
    v1: Mapping[str, Any], *, provenance: str
) -> dict[str, Any]:
    """``derive_full_body_v2_spec`` at :func:`landmark_trunk_locations`."""
    return derive_full_body_v2_spec(
        v1, landmark_trunk_locations(v1), provenance=provenance
    )


@dataclass(frozen=True)
class MassProperties:
    """Mass (kg), world COM (m) and world inertia about the COM (kg m^2)."""

    mass_kg: float
    com_m: Array
    inertia_com: Array


def _solid_world(solid: Mapping[str, Any], pose: Array) -> tuple[float, Array, Array]:
    placed = pose @ np.asarray(solid["placement"], dtype=float)
    rot = placed[:3, :3]
    com = rot @ np.asarray(solid["com_m"], dtype=float) + placed[:3, 3]
    inertia = rot @ np.asarray(solid["inertia_com_kg_m2"], dtype=float) @ rot.T
    return float(solid["mass_kg"]), com, inertia


def spec_mass_properties(
    spec: Mapping[str, Any],
    q: Mapping[str, float] | None = None,
    *,
    exclude_bodies: Sequence[str] = (),
) -> MassProperties:
    """Locked aggregate mass properties of a spec at coordinates ``q``.

    Every joint is locked at ``q`` (missing coordinates are zero); bodies in
    ``exclude_bodies`` are left out. Raises ``TypeError`` for a non-Mapping
    spec and ``ValueError`` for an unknown excluded body or no mass.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("spec_mass_properties needs a spec Mapping")
    names = {b["name"] for b in spec["bodies"]}
    unknown = set(exclude_bodies) - names
    if unknown:
        raise ValueError(f"exclude_bodies names unknown bodies {sorted(unknown)}")
    poses = spec_body_poses(spec, q)
    parts = [
        _solid_world(s, poses[b["name"]])
        for b in spec["bodies"]
        if b["name"] not in exclude_bodies
        for s in b["solids"]
    ]
    mass = sum(m for m, _, _ in parts)
    if not mass > 0.0:
        raise ValueError("spec_mass_properties: no mass")
    com = sum(m * c for m, c, _ in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, i_c in parts:
        r = c - com
        inertia += i_c + m * (float(r @ r) * np.eye(3) - np.outer(r, r))
    return MassProperties(float(mass), np.asarray(com), inertia)


_EXCESS_SOLIDS = {
    "clavicle_links_not_in_de_leva": (("clavicle",), ()),
    "hands_native_mass": (("LHand", "RHand"), ("hand", "hand")),
    "hand_standoffs": (("LHandStandoff", "RHandStandoff"), ()),
    "lower_limbs_rajagopal": (
        tuple(f"{s}_{side}" for side in "rl" for s in ("femur", "tibia", "talus"))
        + tuple(f"{s}_{side}" for side in "rl" for s in ("calcn", "toes")),
        ("thigh", "shank", "foot") * 2,
    ),
}


def human_mass_ledger(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Human mass (total minus the club) against the subject mass, explained.

    ``excess_kg`` attributes the difference to solids the de Leva table does
    not have or sizes differently; ``other`` is the remainder (zero when the
    attribution is complete). Raises ``KeyError`` for a spec without the
    anthropometric ``subject``, ``anthropometry`` and ``club`` blocks.
    """
    subject = float(spec["subject"]["mass_kg"])
    fractions = spec["anthropometry"]["segments"]
    by_suffix: dict[str, float] = {}
    for body in spec["bodies"]:
        for solid in body["solids"]:
            suffix = solid["name"].rsplit("/", 1)[-1]
            by_suffix[suffix] = by_suffix.get(suffix, 0.0) + float(solid["mass_kg"])
    total = sum(by_suffix.values())
    human = total - float(spec["club"]["total_mass_kg"])
    excess = {
        key: sum(by_suffix.get(s, 0.0) for s in solids)
        - sum(subject * fractions[seg]["mass_fraction"] for seg in segments)
        for key, (solids, segments) in _EXCESS_SOLIDS.items()
    }
    excess["other"] = human - subject - sum(excess.values())
    return {
        "subject_mass_kg": subject,
        "total_mass_kg": total,
        "human_mass_kg": human,
        "excess_kg": excess,
    }


def _marker_function(
    doc: Mapping[str, Any], base: Mapping[str, float], markers: Sequence[str]
):
    names = V2_TRUNK.coordinates

    def observe(values: Array) -> Array:
        q = {**base, **dict(zip(names, (float(v) for v in values), strict=True))}
        found = compute_spec_marker_positions(doc, q)
        return np.concatenate([found[m] for m in markers])

    return observe


def _check_markers(doc: Mapping[str, Any], markers: Sequence[str]) -> None:
    attached = doc.get("marker_attachments", {})
    missing = [
        m for m in markers if m not in attached or attached[m].get("offset_m") is None
    ]
    if not markers or missing:
        raise ValueError(f"Observability needs attached marker offsets: {missing}")


def _coordinate_status(
    info: Array, posterior: Array, null: Array
) -> tuple[dict[str, str], dict[str, float]]:
    data_cov = np.linalg.pinv(info)
    status, weight = {}, {}
    for k, name in enumerate(V2_TRUNK.coordinates):
        if null.size and float(np.max(np.abs(null[k]))) > _NULL_SHARE:
            status[name], weight[name] = "degenerate", 1.0
            continue
        weight[name] = float(1.0 - posterior[k, k] / data_cov[k, k])
        dominated = weight[name] > PRIOR_DOMINATED_WEIGHT
        status[name] = "prior_dominated" if dominated else "identified"
    return status, weight


def _regional_sum_sd_deg(info: Array, null: Array) -> dict[str, float]:
    """Data-only SD of the summed lumbar + thoracic ISB angle per axis."""
    data_cov = np.linalg.pinv(info)
    out = {}
    for joint_axis, axis in enumerate(V2_TRUNK.joints[0].isb_axes):
        c = np.zeros(len(V2_TRUNK.coordinates))
        for offset, joint in ((0, V2_TRUNK.joints[0]), (3, V2_TRUNK.joints[1])):
            c[offset + joint_axis] = joint.isb_signs[joint_axis]
        hidden = null.size and float(np.max(np.abs(c @ null))) > _NULL_SHARE
        out[axis] = math.inf if hidden else math.degrees(math.sqrt(c @ data_cov @ c))
    return out


def _thorax_rotation_jacobian(
    doc: Mapping[str, Any], base: Mapping[str, float], step: float = 1e-6
) -> Array:
    """3 x 6 Jacobian of the thorax-in-pelvis rotation (rad/rad, pelvis axes).

    Central differences of ``R0^T R`` with the skew-part (vee) readout, which
    is exact to O(step^2) for the small rotation it differentiates.
    """
    lumbar, thoracic = V2_TRUNK.locate(doc)
    pelvis, thorax = lumbar["parent"], thoracic["child"]

    def rotation(q: Mapping[str, float]) -> Array:
        poses = spec_body_poses(doc, q)
        return (np.linalg.inv(poses[pelvis]) @ poses[thorax])[:3, :3]

    r0 = rotation(base)
    cols = []
    for name in V2_TRUNK.coordinates:
        x = float(base.get(name, 0.0))
        d = r0.T @ (
            rotation({**base, name: x + step}) - rotation({**base, name: x - step})
        )
        skew = (d - d.T) / (4.0 * step)
        cols.append(r0 @ np.array([skew[2, 1], skew[0, 2], skew[1, 0]]))
    return np.column_stack(cols)


def _thorax_orientation_sd_deg(jac: Array, info: Array, null: Array) -> dict:
    """Data-only SD of the thorax orientation about the pelvis axes (deg).

    ``inf`` on an axis the data null space reaches (``|J n| > _NULL_SHARE``).
    """
    cov = jac @ np.linalg.pinv(info) @ jac.T
    hidden = np.max(np.abs(jac @ null), axis=1) if null.size else np.zeros(3)
    return {
        axis: math.inf
        if hidden[k] > _NULL_SHARE
        else math.degrees(math.sqrt(cov[k, k]))
        for k, axis in enumerate("xyz")
    }


def trunk_observability(
    doc: Mapping[str, Any],
    *,
    markers: Sequence[str] = BACK_MARKERS,
    noise_m: float = DEFAULT_MARKER_NOISE_M,
    prior_sd_rad: float = DEFAULT_PRIOR_SD_RAD,
    q: Mapping[str, float] | None = None,
    prior: RegionalPrior | None = None,
) -> dict[str, Any]:
    """Rank and identifiability of the six v2 trunk coordinates from markers.

    Linearised at ``q`` (default the neutral pose) with the pelvis held. The
    data information is ``J^T J / noise_m^2``; the prior adds
    ``D^T D / prior_sd_rad^2`` (``D`` from ``prior.split_matrix``). Per
    coordinate the status is ``degenerate`` (touched by the data null space),
    ``prior_dominated`` (the prior removes more than half of the data-only
    variance) or ``identified``. ``split_status`` is ``regularised`` unless all
    six are identified. Raises ``ValueError`` for a non-v2 document, unknown
    or unattached markers, or a non-positive noise or prior scale.
    """
    require_full_body_schema(
        doc, consumer="trunk_observability", accepted=("full-body-v2",)
    )
    for label, value in (("noise_m", noise_m), ("prior_sd_rad", prior_sd_rad)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{label} must be positive and finite")
    _check_markers(doc, markers)
    prior = prior or RegionalPrior.from_coupling_law()
    base = dict(q or {})
    x0 = np.array([float(base.get(c, 0.0)) for c in V2_TRUNK.coordinates])
    probe = probe_synthetic_chain_identifiability(
        V2_TRUNK.coordinates,
        _marker_function(doc, base, tuple(markers)),
        x0,
        relative_tolerance=_RANK_RTOL,
    )
    s, v = probe.singular_values, probe.right_singular_vectors
    null = np.asarray(v[:, probe.rank :], dtype=np.float64)
    info = (v * (s**2)) @ v.T / noise_m**2
    split = prior.split_matrix(V2_TRUNK)
    posterior = np.linalg.inv(info + split.T @ split / prior_sd_rad**2)
    status, weight = _coordinate_status(info, posterior, null)
    regularised = any(v != "identified" for v in status.values())
    return {
        "markers": list(markers),
        "pelvis": "held (fixed by the waist markers)",
        "noise_m": noise_m,
        "prior_sd_rad": prior_sd_rad,
        "singular_values_m_per_rad": s.tolist(),
        "rank": probe.rank,
        "null_space_dimension": len(V2_TRUNK.coordinates) - probe.rank,
        "null_directions": null.T.tolist(),
        "condition_number": probe.condition_number,
        "coordinate_status": status,
        "prior_weight": weight,
        "posterior_sd_deg": {
            n: math.degrees(math.sqrt(posterior[k, k]))
            for k, n in enumerate(V2_TRUNK.coordinates)
        },
        "regional_sum_data_sd_deg": _regional_sum_sd_deg(info, null),
        "null_share": {
            n: float(np.max(np.abs(null[k]))) if null.size else 0.0
            for k, n in enumerate(V2_TRUNK.coordinates)
        },
        "thorax_orientation_data_sd_deg": _thorax_orientation_sd_deg(
            _thorax_rotation_jacobian(doc, base), info, null
        ),
        "split_status": "regularised" if regularised else "identified",
        "distribution_source": prior.source(),
    }
