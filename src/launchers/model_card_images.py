"""Tile image mappings for launcher model cards."""

from __future__ import annotations

# Tile image file names
_IMG_SIMSCAPE = "simscape_multibody.png"
_IMG_MATLAB = "matlab_logo.png"

# Maps display names to tile image files in assets/
MODEL_IMAGES: dict[str, str] = {
    # Physics Engines - Current names from models.yaml
    "MuJoCo": "mujoco_humanoid.png",
    "Drake": "drake.png",
    "Pinocchio": "pinocchio.png",
    "OpenSim": "opensim.png",
    "MyoSuite": "myosim.png",
    # MATLAB/Simscape
    "Matlab Models": _IMG_MATLAB,
    # Tools
    "Motion Capture": "c3d_viewer_modern.png",
    "Model Explorer": "urdf_icon.png",
    "Putting Green": "putting_green_modern.png",
    "Video Analyzer": "video_analyzer_modern.png",
    "Data Explorer": "data_explorer_modern.png",
    "OpenPose": "openpose.png",
    "MediaPipe": "mediapipe.png",
    "Project Map": "project_map.png",
    "Movement Optimizer": "movement_optimizer.png",
    # Legacy names (backward compatibility)
    "MuJoCo Humanoid": "mujoco_humanoid.png",
    "MuJoCo Dashboard": "mujoco_hand.png",
    "Drake Dashboard": "drake.png",
    "Pinocchio Dashboard": "pinocchio.png",
    "Drake Golf Model": "drake.png",
    "Pinocchio Golf Model": "pinocchio.png",
    "OpenSim Golf": "opensim.png",
    "MyoSim Suite": "myosim.png",
    "OpenPose Analysis": "openpose.jpg",
    "Matlab Simscape": _IMG_MATLAB,
    "Matlab Simscape 2D": _IMG_MATLAB,
    "Matlab Simscape 3D": _IMG_MATLAB,
    "Dataset Generator GUI": _IMG_MATLAB,
    "Golf Swing Analysis GUI": _IMG_MATLAB,
    "MATLAB Code Analyzer": _IMG_MATLAB,
    "URDF Generator": "urdf_icon.png",
    "C3D Motion Viewer": "c3d_viewer_modern.png",
    "Shot Tracer": "golf_icon.png",
    # New launcher tiles
    "Cross Engine": "cross_engine.svg",
    "Exercise Dashboard": "exercise_dashboard.svg",
    "Swing Optimizer": "swing_optimizer.svg",
    "Injury Analysis": "injury_analysis.svg",
    "Terrain Engine": "putting_green_modern.png",
    "BunkerShot 3D": "bunkershot3d.svg",
    "Pendulum": "pendulum.svg",
    "Chat Assistant": "golf_logo.png",
    "Character Builder": "urdf_icon.png",
    "Pose Studio": "pose_studio.svg",
    "Dataset Generator": "data_explorer_modern.png",
    "Golf Simulation Suite": "golf_logo.png",
    "Motion-Match Preview": "motion_target_preview.svg",
    "Starting-Pose Matcher (legacy)": "motion_target_preview.svg",
    "Data Processor": "data_explorer_modern.png",
    "Video Processor": "video_analyzer_modern.png",
}
