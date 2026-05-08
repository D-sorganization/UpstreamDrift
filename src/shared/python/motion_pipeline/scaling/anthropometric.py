"""
Anthropometric scaling for motion capture pipeline.

Part of issue #4565. Scales a generic SkeletonRig to match subject-specific
marker data using segment length estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..contracts import (
    JointDef,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)

logger = logging.getLogger(__name__)


@dataclass
class MarkerMap:
    """
    Mapping from marker names to skeleton segments.
    
    Attributes:
        marker_to_segment: Dict mapping marker name to segment name
        segment_pairs: List of (proximal, distal) marker pairs for segment length
    """
    
    marker_to_segment: dict[str, str] = field(default_factory=dict)
    segment_pairs: list[tuple[str, str]] = field(default_factory=list)


def _compute_segment_length(
    markers: MarkerFrame,
    proximal_marker: str,
    distal_marker: str,
) -> float | None:
    """
    Compute the distance between two markers.
    
    Args:
        markers: Marker frame containing the markers
        proximal_marker: Name of proximal marker
        distal_marker: Name of distal marker
    
    Returns:
        Distance in meters, or None if markers not found
    """
    if proximal_marker not in markers.markers:
        return None
    if distal_marker not in markers.markers:
        return None
    
    prox = markers.markers[proximal_marker]
    dist = markers.markers[distal_marker]
    
    # Compute Euclidean distance
    diff = np.array([dist.x - prox.x, dist.y - prox.y, dist.z - prox.z])
    return float(np.linalg.norm(diff))


def _compute_average_segment_lengths(
    trajectory: MarkerTrajectory,
    segment_pairs: list[tuple[str, str]],
) -> dict[str, float]:
    """
    Compute average segment lengths across all frames.
    
    Args:
        trajectory: Marker trajectory
        segment_pairs: List of (proximal, distal) marker pairs
    
    Returns:
        Dict mapping pair name to average length
    """
    lengths: dict[str, list[float]] = {f"{p}-{d}": [] for p, d in segment_pairs}
    
    for frame in trajectory.frames:
        for proximal, distal in segment_pairs:
            length = _compute_segment_length(frame, proximal, distal)
            if length is not None:
                lengths[f"{proximal}-{distal}"].append(length)
    
    # Compute averages
    averages = {}
    for key, values in lengths.items():
        if values:
            averages[key] = float(np.mean(values))
        else:
            averages[key] = 0.0
    
    return averages


def _get_reference_segment_lengths(
    rig: SkeletonRig,
    marker_map: MarkerMap,
) -> dict[str, float]:
    """
    Get reference segment lengths from the generic skeleton rig.
    
    Uses Dempster / de Leva anthropometric regression based on
    total height and segment proportions.
    
    Args:
        rig: Generic skeleton rig
        marker_map: Marker to segment mapping
    
    Returns:
        Dict mapping segment pair to reference length
    """
    # Default anthropometric proportions (de Leva 1996)
    # These are normalized to total height
    SEGMENT_PROPORTIONS = {
        "pelvis": 0.15,  # ASIS width
        "right_thigh": 0.245,
        "left_thigh": 0.245,
        "right_shank": 0.246,
        "left_shank": 0.246,
        "right_upper_arm": 0.186,
        "left_upper_arm": 0.186,
        "right_forearm": 0.152,
        "left_forearm": 0.152,
        "torso": 0.35,
        "neck": 0.04,
        "head": 0.13,
    }
    
    # Assume reference height of 1.75m
    reference_height = 1.75
    
    reference_lengths = {}
    for pair_name, _ in marker_map.segment_pairs:
        # Extract segment name from pair
        segment = pair_name.split("-")[0].lower()
        
        # Map common marker pair names to segment names
        segment_mapping = {
            "rasi": "pelvis",
            "lasi": "pelvis",
            "rthi": "right_thigh",
            "lkne": "left_knee",
            "rthigh": "right_thigh",
            "lthigh": "left_thigh",
            "rkne": "right_knee",
            "lknee": "left_knee",
            "rank": "right_ankle",
            "lank": "left_ankle",
            "rupa": "right_upper_arm",
            "lupa": "left_upper_arm",
            "relb": "right_elbow",
            "lelb": "left_elbow",
            "rwra": "right_wrist",
            "lwra": "left_wrist",
        }
        
        # Try to find segment proportion
        proportion = SEGMENT_PROPORTIONS.get(segment, 0.15)
        reference_lengths[pair_name] = proportion * reference_height
    
    return reference_lengths


def scale_skeleton(
    rig: SkeletonRig,
    calibration_markers: MarkerFrame | MarkerTrajectory,
    marker_to_segment: dict[str, str] | None = None,
    segment_pairs: list[tuple[str, str]] | None = None,
) -> SkeletonRig:
    """
    Scale a skeleton rig to match subject-specific marker data.
    
    Uses segment length estimation from marker pairs to compute
    scale factors for each body segment.
    
    Args:
        rig: Generic skeleton rig to scale
        calibration_markers: Static marker frame or trajectory for calibration
        marker_to_segment: Optional mapping from marker names to segments.
                          If None, uses markers from calibration data.
        segment_pairs: Optional list of (proximal, distal) marker pairs.
                      If None, uses default pairs.
    
    Returns:
        Scaled SkeletonRig with adjusted segment lengths
    
    Raises:
        ValueError: If insufficient markers for scaling
    """
    # Get marker data
    if isinstance(calibration_markers, MarkerTrajectory):
        # Use first frame for calibration
        if not calibration_markers.frames:
            raise ValueError("Empty marker trajectory")
        markers = calibration_markers.frames[0]
    else:
        markers = calibration_markers
    
    # Build marker map
    if marker_to_segment is None:
        marker_to_segment = {}
    if segment_pairs is None:
        segment_pairs = []
    
    marker_map = MarkerMap(
        marker_to_segment=marker_to_segment,
        segment_pairs=segment_pairs,
    )
    
    # Compute measured segment lengths from markers
    if segment_pairs:
        measured_lengths = _compute_average_segment_lengths(
            calibration_markers if isinstance(calibration_markers, MarkerTrajectory)
            else MarkerTrajectory(
                id="calibration",
                frames=[markers],
            ),
            segment_pairs,
        )
    else:
        measured_lengths = {}
    
    # Get reference lengths from generic rig
    reference_lengths = _get_reference_segment_lengths(rig, marker_map)
    
    # Compute scale factors
    scale_factors: dict[str, float] = {}
    for pair_name, ref_length in reference_lengths.items():
        measured = measured_lengths.get(pair_name, 0.0)
        if measured > 0 and ref_length > 0:
            scale_factors[pair_name] = measured / ref_length
        else:
            scale_factors[pair_name] = 1.0
    
    # Compute global scale factor (average of all segments)
    valid_scales = [s for s in scale_factors.values() if s > 0]
    global_scale = float(np.mean(valid_scales)) if valid_scales else 1.0
    
    # DbC postcondition: all scale factors should be positive and reasonable
    if global_scale <= 0:
        raise ValueError("Computed scale factor is non-positive")
    if global_scale < 0.5 or global_scale > 2.0:
        logger.warning(f"Unusual scale factor: {global_scale:.2f}")
    
    # Scale the skeleton
    scaled_joints: dict[str, JointDef] = {}
    for joint_name, joint in rig.joints.items():
        # Scale T-pose offset
        scaled_offset = [v * global_scale for v in joint.tpose_offset]
        
        scaled_joints[joint_name] = JointDef(
            name=joint.name,
            parent=joint.parent,
            children=joint.children,
            tpose_offset=scaled_offset,
            axes=joint.axes,
            limits=joint.limits,
            semantic_label=joint.semantic_label,
        )
    
    # Create scaled rig
    scaled_rig = SkeletonRig(
        id=f"{rig.id}-scaled",
        joints=scaled_joints,
        root_joint=rig.root_joint,
        up_axis=rig.up_axis,
        scale=global_scale,
        metadata={
            **rig.metadata,
            "scale_factor": global_scale,
            "scale_factors_by_segment": scale_factors,
            "reference_height": 1.75 * global_scale,
            "calibration_markers": len(markers.markers),
        }
    )
    
    logger.info(f"Scaled skeleton: global_scale={global_scale:.3f}")
    
    return scaled_rig


def estimate_subject_height(
    calibration_markers: MarkerFrame | MarkerTrajectory,
    marker_set: str = "plug-in-gait",
) -> float:
    """
    Estimate subject height from marker data.
    
    Uses standard anthropometric relationships:
    - ASIS height ≈ 0.53 × height
    - C7 height ≈ 0.82 × height (when standing)
    
    Args:
        calibration_markers: Marker frame or trajectory
        marker_set: Name of marker set used
    
    Returns:
        Estimated height in meters
    """
    # Get marker data
    if isinstance(calibration_markers, MarkerTrajectory):
        markers = calibration_markers.frames[0]
    else:
        markers = calibration_markers
    
    # Find lowest foot marker (heel or toe)
    foot_markers = ["RHEE", "LHEE", "RTOE", "LTOE", "R_ANKLE", "L_ANKLE"]
    foot_z = []
    for name in foot_markers:
        if name in markers.markers:
            foot_z.append(markers.markers[name].z)
    
    if not foot_z:
        # Default to 0 if no foot markers
        foot_z = [0.0]
    
    # Find highest head marker
    head_markers = ["HEAD", "Head_Top", "FHD", "C7"]
    head_z = []
    for name in head_markers:
        if name in markers.markers:
            head_z.append(markers.markers[name].z)
    
    if not head_z:
        # Fallback: use any marker
        head_z = [m.z for m in markers.markers.values()]
    
    # Compute height from vertical distance
    # Add ~10cm for head top above C7
    vertical_distance = max(head_z) - min(foot_z)
    estimated_height = vertical_distance / 0.82 + 0.10
    
    return float(estimated_height)