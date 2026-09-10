"""Calibrated anatomical joint poses and orientation representations.

World rotations map local column vectors into a right-handed world frame.
Positions are metres, rotation vectors are radians, and Euler units are explicit.
Representational conversion cannot supply missing anatomical calibration.
SciPy's shared SO(3) implementation supplies all six Cardan and six proper Euler
sequences, both intrinsic (uppercase) and extrinsic (lowercase).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RotationConvention:
    """Explicit Euler sequence; letter case selects moving or fixed axes."""

    sequence: str = "XYZ"
    degrees: bool = False

    def __post_init__(self) -> None:
        s = self.sequence
        if not isinstance(s, str) or len(s) != 3:
            raise ValueError("Euler sequence must contain exactly three axes")
        if not (s.isupper() or s.islower()) or set(s.lower()) - set("xyz"):
            raise ValueError(
                "Use uniformly uppercase intrinsic or lowercase extrinsic axes"
            )
        if s[0] == s[1] or s[1] == s[2]:
            raise ValueError("Adjacent Euler axes must differ")
        if not isinstance(self.degrees, bool):
            raise ValueError("degrees must be boolean")


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim < len(shape) or a.shape[-len(shape) :] != shape:
        raise ValueError(f"{name} must have trailing shape {shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite; exclude missing frames explicitly")
    return a


def _rotation_matrix(value: object) -> np.ndarray:
    a = _array(value, (3, 3), "rotation")
    if not np.allclose(np.swapaxes(a, -1, -2) @ a, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("rotation must be orthonormal")
    if not np.allclose(np.linalg.det(a), 1.0, atol=1e-8, rtol=0):
        raise ValueError("rotation must be proper and right-handed")
    return a


@dataclass(frozen=True)
class OrientationResult:
    """Converted coordinates with Euler non-uniqueness diagnostics.

    At a singularity coordinates reconstruct the rotation, but the first and last
    Euler components are not individually identifiable. Do not differentiate them.
    """

    values: np.ndarray
    singular: np.ndarray
    representation: str
    convention: RotationConvention


def orientations_to_matrix(
    values: object,
    representation: str,
    convention: RotationConvention = RotationConvention(),
) -> np.ndarray:
    """Convert arbitrary batches without silently repairing invalid matrices.

    Quaternion order must be named. Nonzero finite quaternions are normalized;
    rotation vectors use axis times angle in radians, independent of Euler units.
    """
    if representation == "matrix":
        return _rotation_matrix(values).copy()
    width = 4 if representation.startswith("quaternion_") else 3
    a = _array(values, (width,), representation)
    batch = a.shape[:-1]
    flat = a.reshape(-1, width)
    if representation in ("quaternion_xyzw", "quaternion_wxyz"):
        norms = np.linalg.norm(flat, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError("quaternions must be nonzero")
        flat = flat / norms[:, None]
        if representation == "quaternion_wxyz":
            flat = flat[:, [1, 2, 3, 0]]
        rotation = Rotation.from_quat(flat)
    elif representation == "euler":
        rotation = Rotation.from_euler(
            convention.sequence, flat, degrees=convention.degrees
        )
    elif representation == "rotation_vector":
        rotation = Rotation.from_rotvec(flat)
    else:
        raise ValueError(f"Unknown orientation representation: {representation}")
    return rotation.as_matrix().reshape(batch + (3, 3))


def matrix_to_orientations(
    matrices: object,
    representation: str,
    convention: RotationConvention = RotationConvention(),
) -> OrientationResult:
    """Convert validated rotations, reporting each Euler singular frame."""
    matrices_arr = cast(np.ndarray, _rotation_matrix(matrices))
    batch = matrices_arr.shape[:-2]
    rotations = Rotation.from_matrix(matrices_arr.reshape(-1, 3, 3))
    singular = np.zeros(batch, dtype=bool)
    if representation == "matrix":
        values = matrices_arr.copy()
    elif representation == "euler":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Gimbal lock detected.*", category=UserWarning
            )
            angles = rotations.as_euler(cast(Any, convention.sequence), degrees=False)
        proper_euler = convention.sequence[0] == convention.sequence[2]
        measure = np.sin(angles[:, 1]) if proper_euler else np.cos(angles[:, 1])
        singular = (np.abs(measure) <= 1e-7).reshape(batch)
        values = (np.rad2deg(angles) if convention.degrees else angles).reshape(
            batch + (3,)
        )
    elif representation in ("quaternion_xyzw", "quaternion_wxyz"):
        values = rotations.as_quat(canonical=True)
        if representation == "quaternion_wxyz":
            values = values[:, [3, 0, 1, 2]]
        values = values.reshape(batch + (4,))
    elif representation == "rotation_vector":
        values = rotations.as_rotvec().reshape(batch + (3,))
    else:
        raise ValueError(f"Unknown orientation representation: {representation}")
    return OrientationResult(values, singular, representation, convention)


def convert_orientations(
    values: object,
    source_representation: str,
    target_representation: str,
    source_convention: RotationConvention = RotationConvention(),
    target_convention: RotationConvention = RotationConvention(),
) -> OrientationResult:
    """Convert through SO(3), never by permuting Euler components."""
    return matrix_to_orientations(
        orientations_to_matrix(values, source_representation, source_convention),
        target_representation,
        target_convention,
    )


@dataclass(frozen=True)
class AnatomicalFrame:
    """World pose of a calibrated segment, not an unqualified engine body.

    ``calibration_id`` identifies the actual landmark/digitization/model mapping.
    Batch shapes must match; missing frames must be removed before construction.
    """

    rotation_world: np.ndarray
    origin_world: np.ndarray
    calibration_id: str

    def __post_init__(self) -> None:
        r = _rotation_matrix(self.rotation_world).copy()
        p = _array(self.origin_world, (3,), "origin_world").copy()
        if r.shape[:-2] != p.shape[:-1]:
            raise ValueError("Rotation and position batch shapes must match")
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("An anatomical calibration identifier is required")
        r.setflags(write=False)
        p.setflags(write=False)
        object.__setattr__(self, "rotation_world", r)
        object.__setattr__(self, "origin_world", p)

    @classmethod
    def from_body_pose(
        cls,
        body_rotation_world: object,
        body_origin_world: object,
        anatomical_rotation_body: object,
        anatomical_origin_body: object,
        calibration_id: str,
    ) -> AnatomicalFrame:
        """Apply a fixed measured anatomical-to-body calibration transform."""
        body_r = _rotation_matrix(body_rotation_world)
        body_p = _array(body_origin_world, (3,), "body_origin_world")
        calibration_r = _rotation_matrix(anatomical_rotation_body)
        calibration_p = _array(anatomical_origin_body, (3,), "anatomical_origin_body")
        if calibration_r.shape != (3, 3) or calibration_p.shape != (3,):
            raise ValueError("Calibration must be a single fixed transform")
        return cls(
            body_r @ calibration_r,
            body_p + np.einsum("...ij,j->...i", body_r, calibration_p),
            calibration_id,
        )


@dataclass(frozen=True)
class JointKinematics:
    """Distal anatomical pose expressed in proximal anatomical coordinates."""

    rotation_relative: np.ndarray
    position_proximal: np.ndarray
    angles: np.ndarray
    singular: np.ndarray
    convention: RotationConvention
    provenance: tuple[str, str]


def joint_kinematics(
    proximal: AnatomicalFrame,
    distal: AnatomicalFrame,
    convention: RotationConvention = RotationConvention(),
) -> JointKinematics:
    """Compute relative orientation and origin separation for matching frames."""
    if proximal.rotation_world.shape != distal.rotation_world.shape:
        raise ValueError("Proximal and distal frame shapes must match")
    inverse = np.swapaxes(proximal.rotation_world, -1, -2)
    relative = inverse @ distal.rotation_world
    position = np.einsum(
        "...ij,...j->...i", inverse, distal.origin_world - proximal.origin_world
    )
    angles = matrix_to_orientations(relative, "euler", convention)
    return JointKinematics(
        relative,
        position,
        angles.values,
        angles.singular,
        convention,
        (proximal.calibration_id, distal.calibration_id),
    )


@dataclass(frozen=True)
class AnatomicalProfile:
    """Sourced axes for ALREADY calibrated anatomical coordinate systems.

    Angle names describe positive directions where the publication specifies them;
    otherwise paired names retain right-handed mathematical signs. Left/right bone
    frames must follow the cited calibration, not a post-hoc angle sign reversal.
    """

    sequence: str
    angle_names: tuple[str, str, str]
    calibration: str
    source_url: str
    source_section: str
    negative_middle: bool = False


_ISB_I = "https://media.isbweb.org/images/documents/standards/isb_jcs_part_i.pdf"
_ISB_II = "https://media.isbweb.org/images/documents/standards/isb_jcs_part_ii.pdf"
ANATOMICAL_PROFILES = MappingProxyType(
    {
        "isb_hip": AnatomicalProfile(
            "ZXY",
            ("flexion_extension", "adduction_abduction", "internal_external_rotation"),
            "Pelvis: hip-center origin, Z right along ASISs, X anterior in ASIS/PSIS plane. Femur: Y toward hip center from epicondyle midpoint; Z right in epicondyle/hip plane.",
            _ISB_I,
            "4.3-4.5",
        ),
        "isb_ankle_complex": AnatomicalProfile(
            "ZXY",
            ("dorsiflexion", "inversion", "internal_rotation"),
            "Tibia: intermalleolar origin, Z right along malleoli, X anterior normal to torsional plane. Calcaneus: neutral shank longitudinal Y cranial, X anterior; preserve neutral calibration. Entire ankle complex only.",
            _ISB_I,
            "3.3-3.5",
        ),
        "isb_spine": AnatomicalProfile(
            "ZXY",
            ("flexion_extension", "lateral_bending", "axial_rotation"),
            "Adjacent vertebrae: Y cephalad through endplate centers, Z right along pedicle landmarks, X anterior; common neutral origin from longitudinal-axis intersection or endplate midpoint.",
            _ISB_I,
            "5.2-5.3",
        ),
        "isb_thorax": AnatomicalProfile(
            "ZXY",
            ("extension", "right_lateral_flexion", "left_axial_rotation"),
            "Global proximal ISB frame. Thorax: IJ origin; Y upward between PX/T8 and IJ/C7 midpoints, Z right normal to landmark plane, X anterior.",
            _ISB_II,
            "2.3.1, 2.4.1",
        ),
        "isb_glenohumeral": AnatomicalProfile(
            "YXY",
            ("plane_of_elevation", "negative_elevation", "internal_rotation"),
            "Scapula: AA origin, Z TS-to-AA, X anterior normal to AI/AA/TS plane. Humerus: GH origin, Y proximal from epicondyle midpoint; use documented humerus option 1 or calibrated option 2.",
            _ISB_II,
            "2.3.3-2.3.5, 2.4.4",
            True,
        ),
        "isb_thoracohumeral": AnatomicalProfile(
            "YXY",
            ("plane_of_elevation", "negative_elevation", "internal_rotation"),
            "Thorax IJ/C7/PX/T8 frame and humerus GH/epicondyle frame. Plane zero is abduction; 90 degrees is forward flexion. Humerus option must be recorded.",
            _ISB_II,
            "2.4.7",
            True,
        ),
        "isb_sternoclavicular": AnatomicalProfile(
            "YXZ",
            ("protraction", "depression", "posterior_axial_rotation"),
            "Thorax frame; clavicle SC origin, Z toward AC, X anterior perpendicular to clavicle Z and thorax Y.",
            _ISB_II,
            "2.3.2, 2.4.2",
        ),
        "isb_acromioclavicular": AnatomicalProfile(
            "YXZ",
            ("protraction", "medial_rotation", "posterior_tilt"),
            "Calibrated clavicle and scapula frames; anatomical resting alignment need not have zero angles.",
            _ISB_II,
            "2.4.3",
        ),
        "isb_scapulothoracic": AnatomicalProfile(
            "YXZ",
            ("protraction", "medial_rotation", "posterior_tilt"),
            "Thorax IJ/C7/PX/T8 frame and scapula AA/TS/AI frame; axes are anatomically calibrated.",
            _ISB_II,
            "2.4.6",
        ),
        "isb_elbow": AnatomicalProfile(
            "ZXY",
            ("flexion", "carrying_angle", "pronation"),
            "Humerus preferably option 2 calibrated at 90-degree elbow flexion, neutral forearm. Forearm origin US, Y proximal to epicondyle midpoint, X anterior normal to US/RS/epicondyle plane.",
            _ISB_II,
            "2.3.5-2.3.6, 3.4.1",
        ),
        "isb_wrist": AnatomicalProfile(
            "ZXY",
            ("flexion", "ulnar_deviation", "pronation"),
            "Radius and third metacarpal bone frames per section 4.3 (not elbow forearm frames): right X volar, Y proximal, Z radial; left X dorsal, Y distal, Z ulnar. Neutral metacarpal/radius long axes parallel.",
            _ISB_II,
            "4.3-4.4.1",
        ),
        "grood_suntay_knee": AnatomicalProfile(
            "XYZ",
            ("femoral_x_rotation", "floating_axis_rotation", "tibial_z_rotation"),
            "Femoral X flexion axis, tibial Z longitudinal axis; calibrated right-handed reference frames. Clinical sign and neutral reference must be documented for the model.",
            "https://pubmed.ncbi.nlm.nih.gov/6865355/",
            "Grood and Suntay 1983; user-calibrated axes",
        ),
    }
)


def anatomical_joint_angles(
    matrices: object, profile: str, *, degrees: bool = False
) -> OrientationResult:
    """Extract a named anatomical profile after landmark frame calibration.

    The result retains the published ISB negative shoulder elevation branch.
    Input is distal-to-proximal rotation in the documented anatomical axes.
    """
    try:
        definition = ANATOMICAL_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown anatomical profile: {profile}") from exc
    convention = RotationConvention(definition.sequence, degrees=False)
    result = matrix_to_orientations(matrices, "euler", convention)
    angles = result.values.copy()
    if definition.negative_middle:
        angles[..., 0] += np.pi
        angles[..., 1] *= -1
        angles[..., 2] += np.pi
        angles[..., [0, 2]] = (angles[..., [0, 2]] + np.pi) % (2 * np.pi) - np.pi
    return OrientationResult(
        np.rad2deg(angles) if degrees else angles,
        result.singular,
        "euler",
        RotationConvention(definition.sequence, degrees),
    )


@dataclass(frozen=True)
class FloatingAxisResult:
    """Three JCS rotations plus the instantaneous floating axis in world space."""

    angles: np.ndarray
    floating_axis_world: np.ndarray
    singular: np.ndarray
    convention: RotationConvention
    provenance: tuple[str, str]


def grood_suntay(
    proximal: AnatomicalFrame,
    distal: AnatomicalFrame,
    *,
    proximal_axis: str = "Z",
    distal_axis: str = "Y",
    degrees: bool = False,
) -> FloatingAxisResult:
    """Compute a body-fixed/floating/body-fixed JCS for distinct calibrated axes.

    The equivalent intrinsic Cardan decomposition preserves the right-handed axis
    signs. Floating-axis direction agrees with its sequence's middle axis at neutral.
    Coincident first/third axes (shoulder proper Euler) require the separate profile
    method. Singular floating axes are NaN, never invented anatomical directions.
    """
    if (
        proximal_axis not in ("X", "Y", "Z")
        or distal_axis not in ("X", "Y", "Z")
        or proximal_axis == distal_axis
    ):
        raise ValueError("JCS requires distinct uppercase proximal/distal axes")
    middle = next(a for a in "XYZ" if a not in (proximal_axis, distal_axis))
    convention = RotationConvention(proximal_axis + middle + distal_axis, degrees)
    result = joint_kinematics(proximal, distal, convention)
    i, j, k = ["XYZ".index(a) for a in convention.sequence]
    e1, e3 = proximal.rotation_world[..., :, i], distal.rotation_world[..., :, k]
    parity = np.dot(np.cross(np.eye(3)[k], np.eye(3)[i]), np.eye(3)[j])
    cross = parity * np.cross(e3, e1)
    norm = np.linalg.norm(cross, axis=-1)
    axis = np.full_like(cross, np.nan)
    np.divide(cross, norm[..., None], out=axis, where=norm[..., None] > 1e-7)
    return FloatingAxisResult(
        result.angles, axis, result.singular, convention, result.provenance
    )


@dataclass(frozen=True)
class OrientationSeriesResult:
    """Trajectory conversion with explicit missing-frame and singular masks."""

    values: np.ndarray
    valid: np.ndarray
    singular: np.ndarray


def convert_orientation_series(
    values: object,
    source_representation: str,
    target_representation: str,
    source_convention: RotationConvention = RotationConvention(),
    target_convention: RotationConvention = RotationConvention(),
) -> OrientationSeriesResult:
    """Preserve all-NaN frames; reject partially missing frames and infinities."""
    a = np.asarray(values, dtype=float)
    tail = (
        (3, 3)
        if source_representation == "matrix"
        else (4,)
        if source_representation.startswith("quaternion_")
        else (3,)
    )
    if a.ndim != len(tail) + 1 or a.shape[1:] != tail:
        raise ValueError(
            "Orientation series must have one time dimension and the representation shape"
        )
    flat = a.reshape(len(a), -1)
    valid = np.isfinite(flat).all(axis=1)
    missing = np.isnan(flat).all(axis=1)
    if not np.all(valid | missing):
        raise ValueError(
            "Missing orientation frames must be entirely NaN; infinities are invalid"
        )
    # Validate target even for entirely missing trajectories.
    template = matrix_to_orientations(
        np.eye(3), target_representation, target_convention
    )
    orientations_to_matrix(
        np.zeros((1,) + tail)
        if source_representation not in ("matrix", "quaternion_xyzw", "quaternion_wxyz")
        else np.eye(3)
        if source_representation == "matrix"
        else [0, 0, 0, 1]
        if source_representation == "quaternion_xyzw"
        else [1, 0, 0, 0],
        source_representation,
        source_convention,
    )
    output = np.full((len(a),) + template.values.shape, np.nan)
    singular = np.zeros(len(a), dtype=bool)
    if np.any(valid):
        result = convert_orientations(
            a[valid],
            source_representation,
            target_representation,
            source_convention,
            target_convention,
        )
        output[valid], singular[valid] = result.values, result.singular
    return OrientationSeriesResult(output, valid, singular)
