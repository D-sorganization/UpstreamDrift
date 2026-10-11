"""Builders for OpenCap session fixtures in the layout OpenCap really writes.

The layout and names are pinned from the upstream OpenCap sources (Apache-2.0),
not from this repository, so these fixtures catch drift between UpstreamDrift's
reader and real OpenCap output (#11402, #11403):

- Session layout: ``opencap-processing/utils.py`` (``get_motion_data``,
  ``get_model``, ``get_metadata``).
- Augmented marker names: ``opencap-core/opensimPipeline/Models/
  LaiUhlrich2022_markers_augmenter.xml``.
- Detector keypoints written alongside them: ``opencap-core/utils.py``
  (``getOpenPoseMarkers_lowerExtremity2``,
  ``getMarkers_upperExtremity_noPelvis2``).

Values are synthetic. They qualify the parser contract only, never accuracy.
"""

from __future__ import annotations

from pathlib import Path

# Copied literally from LaiUhlrich2022_markers_augmenter.xml, in file order.
OPENCAP_AUGMENTED_MARKERS: tuple[str, ...] = (
    "r.ASIS_study",
    "L.ASIS_study",
    "r.PSIS_study",
    "L.PSIS_study",
    "r_knee_study",
    "r_mknee_study",
    "r_ankle_study",
    "r_mankle_study",
    "r_toe_study",
    "r_5meta_study",
    "r_calc_study",
    "L_knee_study",
    "L_mknee_study",
    "L_ankle_study",
    "L_mankle_study",
    "L_toe_study",
    "L_calc_study",
    "L_5meta_study",
    "r_shoulder_study",
    "L_shoulder_study",
    "C7_study",
    "r_lelbow_study",
    "r_melbow_study",
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lelbow_study",
    "L_melbow_study",
    "L_lwrist_study",
    "L_mwrist_study",
    "r_thigh1_study",
    "r_thigh2_study",
    "r_thigh3_study",
    "L_thigh1_study",
    "L_thigh2_study",
    "L_thigh3_study",
    "r_sh1_study",
    "r_sh2_study",
    "r_sh3_study",
    "L_sh1_study",
    "L_sh2_study",
    "L_sh3_study",
    "RHJC_study",
    "LHJC_study",
)

# Detector keypoints the augmenter consumes; OpenCap keeps them in the TRC.
OPENCAP_DETECTOR_KEYPOINTS: tuple[str, ...] = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RHip",
    "LHip",
    "RKnee",
    "LKnee",
    "RAnkle",
    "LAnkle",
)

SESSION_METADATA_YAML = """\
calibrationSettings:
  overwriteDeployedIntrinsics: false
  saveSessionIntrinsics: false
gender_mf: m
height_m: 1.82
iphoneModel:
  Cam0: iphone13,3
  Cam1: iphone13,3
markerAugmentationSettings:
  markerAugmenterModel: LSTM
mass_kg: 79.5
openSimModel: LaiUhlrich2022
subjectID: subject-01
"""


def write_trc(
    path: Path,
    marker_names: tuple[str, ...] | list[str],
    *,
    n_frames: int = 3,
    rate_hz: float = 60.0,
    units: str = "m",
) -> Path:
    """Write a TRC file with OpenCap's header layout and synthetic values.

    Marker ``i`` at frame ``f`` sits at ``(i + f/100, 1 + i, 2 + i)`` in the
    given units, so a test can recover which column landed where.
    """
    n_markers = len(marker_names)
    name_row = "Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\t\t\t"
    axis_row = "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, n_markers + 1))
    rows = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        (
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
            "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
        ),
        f"{rate_hz}\t{rate_hz}\t{n_frames}\t{n_markers}\t{units}\t"
        f"{rate_hz}\t1\t{n_frames}",
        name_row,
        axis_row,
    ]
    for frame in range(n_frames):
        coords = "\t".join(
            f"{i + frame / 100:.4f}\t{1 + i:.4f}\t{2 + i:.4f}" for i in range(n_markers)
        )
        rows.append(f"{frame + 1}\t{frame / rate_hz:.6f}\t{coords}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _transform_axis(axis: str, coordinate: str = "") -> str:
    return (
        f'<TransformAxis name="{axis}"><coordinates>{coordinate}</coordinates>'
        "<axis>1 0 0</axis></TransformAxis>"
    )


def _custom_joint(
    name: str,
    coordinates: tuple[str, ...],
    rotations: tuple[str, str, str],
    translations: tuple[str, str, str],
) -> str:
    coords = "".join(f'<Coordinate name="{c}" />' for c in coordinates)
    axes = "".join(
        _transform_axis(f"rotation{i + 1}", c) for i, c in enumerate(rotations)
    ) + "".join(
        _transform_axis(f"translation{i + 1}", c) for i, c in enumerate(translations)
    )
    return (
        f'<CustomJoint name="{name}"><coordinates>{coords}</coordinates>'
        f"<SpatialTransform>{axes}</SpatialTransform></CustomJoint>"
    )


# The structure mirrors LaiUhlrich2022.osim: the pelvis free-flyer is a
# CustomJoint with translation axes, and the knee couples its translations to
# the knee *angle* (so knee_angle_r is rotational despite sitting on
# translation axes). A SliderJoint and a PinJoint cover the other joint types.
SCALED_MODEL_OSIM = (
    '<?xml version="1.0" encoding="UTF-8" ?>\n'
    '<OpenSimDocument Version="40000"><Model name="LaiUhlrich2022_scaled">'
    "<JointSet><objects>"
    + _custom_joint(
        "ground_pelvis",
        ("pelvis_tilt", "pelvis_tx", "pelvis_ty"),
        ("pelvis_tilt", "", ""),
        ("pelvis_tx", "pelvis_ty", ""),
    )
    + _custom_joint(
        "hip_r", ("hip_flexion_r",), ("hip_flexion_r", "", ""), ("", "", "")
    )
    + _custom_joint(
        "walker_knee_r",
        ("knee_angle_r",),
        ("knee_angle_r", "knee_angle_r", ""),
        ("knee_angle_r", "knee_angle_r", ""),
    )
    + '<PinJoint name="mtp_r"><coordinates><Coordinate name="mtp_angle_r" />'
    "</coordinates></PinJoint>"
    + '<SliderJoint name="sled"><coordinates><Coordinate name="sled_x" />'
    "</coordinates></SliderJoint>" + "</objects></JointSet></Model></OpenSimDocument>\n"
)

KINEMATICS_COLUMNS = (
    "pelvis_tilt",
    "pelvis_tx",
    "pelvis_ty",
    "hip_flexion_r",
    "knee_angle_r",
    "mtp_angle_r",
)


def write_scaled_model(session: Path, name: str = "LaiUhlrich2022") -> Path:
    path = session / "OpenSimData" / "Model" / f"{name}_scaled.osim"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SCALED_MODEL_OSIM, encoding="utf-8")
    return path


def write_kinematics(
    session: Path,
    trial: str,
    columns: tuple[str, ...] = KINEMATICS_COLUMNS,
    rows: tuple[tuple[float, ...], ...] = (
        (0.0, 90.0, 0.5, 0.9, 30.0, 45.0, 10.0),
        (1 / 60, 90.0, 0.6, 0.9, 31.0, 46.0, 10.0),
    ),
) -> Path:
    """Write an OpenCap-style IK ``.mot`` (``inDegrees=yes``)."""
    path = session / "OpenSimData" / "Kinematics" / f"{trial}.mot"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Coordinates",
        "version=1",
        f"nRows={len(rows)}",
        f"nColumns={len(columns) + 1}",
        "inDegrees=yes",
        "",
        "Units are S.I. units (second, meters, Newtons, ...)",
        "endheader",
        "time\t" + "\t".join(columns),
    ]
    body = ["\t".join(f"{v:.8f}" for v in row) for row in rows]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return path


def write_opencap_session(
    root: Path,
    trials: tuple[str, ...] = ("swing1",),
    *,
    markers: tuple[str, ...] = OPENCAP_DETECTOR_KEYPOINTS + OPENCAP_AUGMENTED_MARKERS,
    with_metadata: bool = True,
) -> Path:
    """Create ``root/<session>`` with one augmented TRC per trial."""
    session = root / "OpenCapData_session"
    for trial in trials:
        write_trc(session / "MarkerData" / f"{trial}.trc", markers)
    if with_metadata:
        (session / "sessionMetadata.yaml").write_text(
            SESSION_METADATA_YAML, encoding="utf-8"
        )
    return session
