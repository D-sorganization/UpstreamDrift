"""
Canonical marker set mappings for anthropometric scaling.

Part of issue #4565. Provides standardized marker-to-segment mappings
for common marker sets used in motion capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MarkerSet:
    """
    A marker set definition with segment mappings.
    
    Attributes:
        name: Name of the marker set (e.g., "Plug-in-Gait")
        markers: List of marker names in this set
        marker_to_segment: Mapping from marker name to body segment
        segment_pairs: List of marker pairs used for segment length estimation
    """
    
    name: str
    markers: list[str]
    marker_to_segment: dict[str, str]
    segment_pairs: list[tuple[str, str]]  # (proximal_marker, distal_marker)


# Plug-in-Gait (Vicon) marker set
PLUG_IN_GAIT = MarkerSet(
    name="Plug-in-Gait",
    markers=[
        # Head/Neck
        "HEAD",
        "C7",
        "CLAV",
        "STRN",
        # Torso
        "T10",
        # Pelvis
        "RASI",
        "LASI",
        "RPSI",
        "LPSI",
        # Right Arm
        "RUPA",
        "RUPB",
        "RELB",
        "RFRM",
        "RWRA",
        "RWRB",
        # Right Hand
        "RFIN",
        "RTHB",
        # Right Leg
        "RTHI",
        "RKNE",
        "RTIB",
        "RANK",
        # Right Foot
        "RHEE",
        "RTOE",
        # Left Arm
        "LUPA",
        "LUPB",
        "LELB",
        "LFRM",
        "LWRA",
        "LWRB",
        # Left Hand
        "LFIN",
        "LTHB",
        # Left Leg
        "LTHI",
        "LKNE",
        "LTIB",
        "LANK",
        # Left Foot
        "LHEE",
        "LTOE",
    ],
    marker_to_segment={
        # Head/Neck
        "HEAD": "head",
        "C7": "neck",
        "CLAV": "torso",
        "STRN": "torso",
        # Torso
        "T10": "torso",
        # Pelvis
        "RASI": "pelvis",
        "LASI": "pelvis",
        "RPSI": "pelvis",
        "LPSI": "pelvis",
        # Right Arm
        "RUPA": "right_upper_arm",
        "RUPB": "right_upper_arm",
        "RELB": "right_elbow",
        "RFRM": "right_forearm",
        "RWRA": "right_wrist",
        "RWRB": "right_wrist",
        # Right Hand
        "RFIN": "right_hand",
        "RTHB": "right_hand",
        # Right Leg
        "RTHI": "right_thigh",
        "RKNE": "right_knee",
        "RTIB": "right_shank",
        "RANK": "right_ankle",
        # Right Foot
        "RHEE": "right_foot",
        "RTOE": "right_foot",
        # Left Arm
        "LUPA": "left_upper_arm",
        "LUPB": "left_upper_arm",
        "LELB": "left_elbow",
        "LFRM": "left_forearm",
        "LWRA": "left_wrist",
        "LWRB": "left_wrist",
        # Left Hand
        "LFIN": "left_hand",
        "LTHB": "left_hand",
        # Left Leg
        "LTHI": "left_thigh",
        "LKNE": "left_knee",
        "LTIB": "left_shank",
        "LANK": "left_ankle",
        # Left Foot
        "LHEE": "left_foot",
        "LTOE": "left_foot",
    },
    segment_pairs=[
        # Pelvis width
        ("RASI", "LASI"),
        ("RPSI", "LPSI"),
        # Right thigh
        ("RTHI", "RKNE"),
        # Right shank
        ("RKNE", "RANK"),
        # Left thigh
        ("LTHI", "LKNE"),
        # Left shank
        ("LKNE", "LANK"),
        # Right upper arm
        ("RUPA", "RELB"),
        # Right forearm
        ("RELB", "RWRA"),
        # Left upper arm
        ("LUPA", "LELB"),
        # Left forearm
        ("LELB", "LWRA"),
        # Torso
        ("CLAV", "T10"),
        # Shoulders
        ("RUPA", "LUPA"),
    ]
)

# IOR (Institute for Orthopaedic Research) marker set
IOR = MarkerSet(
    name="IOR",
    markers=[
        # Pelvis
        "R_ASIAS",
        "L_ASIAS",
        "R_AISPS",
        "L_AISPS",
        # Right Leg
        "R_THIGH1",
        "R_THIGH2",
        "R_THIGH3",
        "R_KNEE",
        "R_SHANK1",
        "R_SHANK2",
        "R_SHANK3",
        "R_ANKLE",
        # Right Foot
        "R_HEEL",
        "R_TOE",
        # Left Leg
        "L_THIGH1",
        "L_THIGH2",
        "L_THIGH3",
        "L_KNEE",
        "L_SHANK1",
        "L_SHANK2",
        "L_SHANK3",
        "L_ANKLE",
        # Left Foot
        "L_HEEL",
        "L_TOE",
        # Torso
        "C7",
        "T8",
        "T12",
        "CLAV_R",
        "CLAV_L",
        "STER",
    ],
    marker_to_segment={
        # Pelvis
        "R_ASIAS": "pelvis",
        "L_ASIAS": "pelvis",
        "R_AISPS": "pelvis",
        "L_AISPS": "pelvis",
        # Right Leg
        "R_THIGH1": "right_thigh",
        "R_THIGH2": "right_thigh",
        "R_THIGH3": "right_thigh",
        "R_KNEE": "right_knee",
        "R_SHANK1": "right_shank",
        "R_SHANK2": "right_shank",
        "R_SHANK3": "right_shank",
        "R_ANKLE": "right_ankle",
        # Right Foot
        "R_HEEL": "right_foot",
        "R_TOE": "right_foot",
        # Left Leg
        "L_THIGH1": "left_thigh",
        "L_THIGH2": "left_thigh",
        "L_THIGH3": "left_thigh",
        "L_KNEE": "left_knee",
        "L_SHANK1": "left_shank",
        "L_SHANK2": "left_shank",
        "L_SHANK3": "left_shank",
        "L_ANKLE": "left_ankle",
        # Left Foot
        "L_HEEL": "left_foot",
        "L_TOE": "left_foot",
        # Torso
        "C7": "neck",
        "T8": "torso",
        "T12": "torso",
        "CLAV_R": "torso",
        "CLAV_L": "torso",
        "STER": "torso",
    },
    segment_pairs=[
        # Pelvis
        ("R_ASIAS", "L_ASIAS"),
        ("R_AISPS", "L_AISPS"),
        # Right thigh
        ("R_THIGH2", "R_KNEE"),
        # Right shank
        ("R_KNEE", "R_ANKLE"),
        # Left thigh
        ("L_THIGH2", "L_KNEE"),
        # Left shank
        ("L_KNEE", "L_ANKLE"),
        # Torso
        ("T8", "T12"),
    ]
)

# Theia marker set
THEIA = MarkerSet(
    name="Theia",
    markers=[
        # Head
        "Head_Top",
        "Head_Back",
        # Torso
        "Sternum",
        "Spine_7",
        "Spine_1",
        # Pelvis
        "Pelvis_Front",
        "Pelvis_Back",
        "Pelvis_Right",
        "Pelvis_Left",
        # Right Arm
        "Right_Shoulder",
        "Right_Elbow_Lat",
        "Right_Elbow_Med",
        "Right_Wrist_Rad",
        "Right_Wrist_Uln",
        # Right Hand
        "Right_Hand",
        # Right Leg
        "Right_Hip_Prox",
        "Right_Knee_Prox",
        "Right_Knee_Dist",
        "Right_Ankle_Prox",
        "Right_Ankle_Dist",
        # Right Foot
        "Right_Heel",
        "Right_Toe",
        # Left Arm
        "Left_Shoulder",
        "Left_Elbow_Lat",
        "Left_Elbow_Med",
        "Left_Wrist_Rad",
        "Left_Wrist_Uln",
        # Left Hand
        "Left_Hand",
        # Left Leg
        "Left_Hip_Prox",
        "Left_Knee_Prox",
        "Left_Knee_Dist",
        "Left_Ankle_Prox",
        "Left_Ankle_Dist",
        # Left Foot
        "Left_Heel",
        "Left_Toe",
    ],
    marker_to_segment={
        # Head
        "Head_Top": "head",
        "Head_Back": "head",
        # Torso
        "Sternum": "torso",
        "Spine_7": "neck",
        "Spine_1": "pelvis",
        # Pelvis
        "Pelvis_Front": "pelvis",
        "Pelvis_Back": "pelvis",
        "Pelvis_Right": "pelvis",
        "Pelvis_Left": "pelvis",
        # Right Arm
        "Right_Shoulder": "right_upper_arm",
        "Right_Elbow_Lat": "right_elbow",
        "Right_Elbow_Med": "right_elbow",
        "Right_Wrist_Rad": "right_wrist",
        "Right_Wrist_Uln": "right_wrist",
        # Right Hand
        "Right_Hand": "right_hand",
        # Right Leg
        "Right_Hip_Prox": "right_thigh",
        "Right_Knee_Prox": "right_knee",
        "Right_Knee_Dist": "right_knee",
        "Right_Ankle_Prox": "right_shank",
        "Right_Ankle_Dist": "right_ankle",
        # Right Foot
        "Right_Heel": "right_foot",
        "Right_Toe": "right_foot",
        # Left Arm
        "Left_Shoulder": "left_upper_arm",
        "Left_Elbow_Lat": "left_elbow",
        "Left_Elbow_Med": "left_elbow",
        "Left_Wrist_Rad": "left_wrist",
        "Left_Wrist_Uln": "left_wrist",
        # Left Hand
        "Left_Hand": "left_hand",
        # Left Leg
        "Left_Hip_Prox": "left_thigh",
        "Left_Knee_Prox": "left_knee",
        "Left_Knee_Dist": "left_knee",
        "Left_Ankle_Prox": "left_shank",
        "Left_Ankle_Dist": "left_ankle",
        # Left Foot
        "Left_Heel": "left_foot",
        "Left_Toe": "left_foot",
    },
    segment_pairs=[
        # Pelvis
        ("Pelvis_Right", "Pelvis_Left"),
        ("Pelvis_Front", "Pelvis_Back"),
        # Right thigh
        ("Right_Hip_Prox", "Right_Knee_Prox"),
        # Right shank
        ("Right_Knee_Dist", "Right_Ankle_Prox"),
        # Left thigh
        ("Left_Hip_Prox", "Left_Knee_Prox"),
        # Left shank
        ("Left_Knee_Dist", "Left_Ankle_Prox"),
        # Right upper arm
        ("Right_Shoulder", "Right_Elbow_Lat"),
        # Right forearm
        ("Right_Elbow_Lat", "Right_Wrist_Rad"),
        # Left upper arm
        ("Left_Shoulder", "Left_Elbow_Lat"),
        # Left forearm
        ("Left_Elbow_Lat", "Left_Wrist_Rad"),
        # Torso
        ("Sternum", "Spine_1"),
    ]
)

# Vicon Full Body marker set
VICON_FULL_BODY = MarkerSet(
    name="Vicon-Full-Body",
    markers=[
        # Head
        "FHD",
        "BHD",
        # Torso
        "C7",
        "CLAV",
        "STRN",
        "T10",
        # Pelvis
        "R.ASIS",
        "L.ASIS",
        "R.PSIS",
        "L.PSIS",
        # Right Arm
        "RUPA",
        "RELB",
        "RFRM",
        "RWRA",
        "RWRB",
        # Right Hand
        "RFIN",
        "RTHB",
        # Right Leg
        "RTHI",
        "RKNE",
        "RTIB",
        "RANK",
        # Right Foot
        "RHEE",
        "RTOE",
        # Left Arm
        "LUPA",
        "LELB",
        "LFRM",
        "LWRA",
        "LWRB",
        # Left Hand
        "LFIN",
        "LTHB",
        # Left Leg
        "LTHI",
        "LKNE",
        "LTIB",
        "LANK",
        # Left Foot
        "LHEE",
        "LTOE",
    ],
    marker_to_segment={
        # Head
        "FHD": "head",
        "BHD": "head",
        # Torso
        "C7": "neck",
        "CLAV": "torso",
        "STRN": "torso",
        "T10": "torso",
        # Pelvis
        "R.ASIS": "pelvis",
        "L.ASIS": "pelvis",
        "R.PSIS": "pelvis",
        "L.PSIS": "pelvis",
        # Right Arm
        "RUPA": "right_upper_arm",
        "RELB": "right_elbow",
        "RFRM": "right_forearm",
        "RWRA": "right_wrist",
        "RWRB": "right_wrist",
        # Right Hand
        "RFIN": "right_hand",
        "RTHB": "right_hand",
        # Right Leg
        "RTHI": "right_thigh",
        "RKNE": "right_knee",
        "RTIB": "right_shank",
        "RANK": "right_ankle",
        # Right Foot
        "RHEE": "right_foot",
        "RTOE": "right_foot",
        # Left Arm
        "LUPA": "left_upper_arm",
        "LELB": "left_elbow",
        "LFRM": "left_forearm",
        "LWRA": "left_wrist",
        "LWRB": "left_wrist",
        # Left Hand
        "LFIN": "left_hand",
        "LTHB": "left_hand",
        # Left Leg
        "LTHI": "left_thigh",
        "LKNE": "left_knee",
        "LTIB": "left_shank",
        "LANK": "left_ankle",
        # Left Foot
        "LHEE": "left_foot",
        "LTOE": "left_foot",
    },
    segment_pairs=[
        # Pelvis
        ("R.ASIS", "L.ASIS"),
        ("R.PSIS", "L.PSIS"),
        # Right thigh
        ("RTHI", "RKNE"),
        # Right shank
        ("RKNE", "RANK"),
        # Left thigh
        ("LTHI", "LKNE"),
        # Left shank
        ("LKNE", "LANK"),
        # Right upper arm
        ("RUPA", "RELB"),
        # Right forearm
        ("RELB", "RWRA"),
        # Left upper arm
        ("LUPA", "LELB"),
        # Left forearm
        ("LELB", "LWRA"),
        # Torso
        ("CLAV", "T10"),
        # Shoulders
        ("RUPA", "LUPA"),
    ]
)

# Registry of all marker sets
MARKER_SETS: dict[str, MarkerSet] = {
    "plug-in-gait": PLUG_IN_GAIT,
    "plugingait": PLUG_IN_GAIT,
    "pig": PLUG_IN_GAIT,
    "ior": IOR,
    "theia": THEIA,
    "vicon": VICON_FULL_BODY,
    "vicon-full-body": VICON_FULL_BODY,
}


def get_marker_set(name: str) -> MarkerSet:
    """
    Get a marker set by name.
    
    Args:
        name: Name of the marker set (case-insensitive)
    
    Returns:
        MarkerSet object
    
    Raises:
        ValueError: If marker set not found
    """
    name_lower = name.lower()
    if name_lower not in MARKER_SETS:
        available = ", ".join(MARKER_SETS.keys())
        raise ValueError(
            f"Unknown marker set: {name}. Available: {available}"
        )
    return MARKER_SETS[name_lower]