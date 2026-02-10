"""Help Content Mappings for UpstreamDrift.

This module contains all help text and topic mappings for the help system.
It provides:

- UI_HELP_TOPICS: Mapping of UI component identifiers to help topics
- FEATURE_HELP: Detailed help text for major features
- QUICK_TIPS: Short tips for common operations
- get_component_help(): Function to retrieve help for a specific component

The help content is organized by feature area:
- Engine selection and management
- Simulation controls
- Motion capture import and processing
- Visualization settings
- Analysis tools and plotting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HelpTopic:
    """Represents a help topic with associated metadata."""

    id: str
    title: str
    short_description: str
    help_file: str | None = None
    manual_section: str | None = None
    related_topics: list[str] | None = None

    def __post_init__(self) -> None:
        if self.related_topics is None:
            self.related_topics = []


# ==============================================================================
# UI Component to Help Topic Mapping
# ==============================================================================

UI_HELP_TOPICS: dict[str, str] = {
    # Main launcher components
    "launcher_main": "getting_started",
    "launcher_grid": "engine_selection",
    "launcher_search": "getting_started",
    "launcher_docker": "docker_setup",
    "launcher_wsl": "wsl_setup",
    # Engine tiles
    "tile_mujoco": "engine_selection",
    "tile_drake": "engine_selection",
    "tile_pinocchio": "engine_selection",
    "tile_opensim": "engine_selection",
    "tile_myosuite": "engine_selection",
    "tile_matlab": "matlab_integration",
    # Simulation panels
    "simulation_controls": "simulation_controls",
    "simulation_parameters": "simulation_controls",
    "simulation_playback": "simulation_controls",
    "simulation_export": "data_export",
    # Motion capture
    "mocap_import": "motion_capture",
    "mocap_viewer": "motion_capture",
    "mocap_retarget": "motion_capture",
    "c3d_viewer": "motion_capture",
    # Visualization
    "viz_3d_view": "visualization",
    "viz_camera": "visualization",
    "viz_forces": "visualization",
    "viz_energy": "analysis_tools",
    # Analysis
    "analysis_plots": "analysis_tools",
    "analysis_phase": "analysis_tools",
    "analysis_energy": "analysis_tools",
    "analysis_jacobian": "analysis_tools",
    "analysis_kinematic": "analysis_tools",
    # Tools
    "urdf_generator": "urdf_generator",
    "model_explorer": "model_explorer",
    "shot_tracer": "ball_flight",
    "project_map": "project_map",
    # Settings
    "settings_general": "configuration",
    "settings_engines": "engine_selection",
    "settings_visualization": "visualization",
}


# ==============================================================================
# Feature Help Content
# ==============================================================================

FEATURE_HELP: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # Engine Selection
    # -------------------------------------------------------------------------
    "engine_selection": {
        "title": "Engine Selection",
        "short": "Choose the physics engine for your simulation",
        "description": """
UpstreamDrift supports multiple physics engines, each with different strengths:

**MuJoCo** (Recommended for beginners)
- Best for: General biomechanics, contact physics, muscle simulation
- Features: Fast, stable, excellent visualization
- Requirements: pip install mujoco

**Drake**
- Best for: Trajectory optimization, control design
- Features: Advanced optimization tools, model-based design
- Requirements: conda install -c conda-forge drake

**Pinocchio**
- Best for: Fast rigid body dynamics, research algorithms
- Features: ZTCF/ZVCF analysis, analytical Jacobians
- Requirements: conda install -c conda-forge pinocchio

**OpenSim**
- Best for: Musculoskeletal validation, clinical research
- Features: Gold-standard biomechanics models
- Requirements: conda install -c opensim-org opensim

**MyoSuite**
- Best for: Realistic muscle-driven simulation
- Features: 290-muscle models, Hill-type muscles
- Requirements: pip install myosuite (MuJoCo-based)

Select an engine based on your primary analysis goals.
""",
        "tips": [
            "MuJoCo is the easiest to install and get started with",
            "Drake excels at trajectory optimization",
            "Pinocchio is lightweight and great for prototyping",
            "MyoSuite requires MuJoCo and provides muscle simulation",
        ],
        "see_also": ["simulation_controls", "visualization"],
    },
    # -------------------------------------------------------------------------
    # Simulation Controls
    # -------------------------------------------------------------------------
    "simulation_controls": {
        "title": "Simulation Controls",
        "short": "Control simulation playback and parameters",
        "description": """
The simulation controls allow you to run and interact with physics simulations.

**Starting a Simulation**
1. Select a physics engine from the launcher
2. Choose or load a model
3. Set initial conditions (joint angles, velocities)
4. Click "Start Simulation" or press Enter

**Playback Controls**
- Play/Pause: Space bar
- Step Forward: Right arrow (single timestep)
- Step Back: Left arrow (if history available)
- Reset: R key or Reset button
- Speed: Adjust playback speed multiplier

**Parameter Adjustment**
- Timestep: Smaller = more accurate, slower
- Gravity: Enable/disable or adjust vector
- Contact: Enable/disable contact physics

**Recording**
- Enable recording to capture simulation data
- Export to CSV or JSON for analysis
- Save video of visualization

**Keyboard Shortcuts**
- Space: Play/Pause
- R: Reset simulation
- +/-: Adjust playback speed
- Ctrl+S: Save current state
- Ctrl+E: Export data
""",
        "tips": [
            "Use smaller timesteps (0.001s) for accurate dynamics",
            "Enable recording before running to capture all data",
            "Pause simulation to adjust parameters without reset",
        ],
        "see_also": ["engine_selection", "visualization", "analysis_tools"],
    },
    # -------------------------------------------------------------------------
    # Motion Capture
    # -------------------------------------------------------------------------
    "motion_capture": {
        "title": "Motion Capture Import",
        "short": "Import and process motion capture data",
        "description": """
UpstreamDrift supports various motion capture formats for swing analysis.

**Supported Formats**
- C3D: Standard biomechanics format (.c3d files)
- CSV: Custom column mapping for marker positions
- JSON: Hierarchical joint/marker data

**Pose Estimation Systems**
- OpenPose: 25-body keypoints from video
- MediaPipe: 33 landmarks, runs locally
- MoveNet: Lightning/Thunder models

**Importing Data**
1. Click "Import Motion Capture" or use File menu
2. Select your data file
3. Configure marker mapping (if needed)
4. Preview the motion data
5. Click "Import" to load

**Retargeting**
Motion capture data can be retargeted to different models:
1. Load motion capture data
2. Select target model
3. Configure marker-to-joint mapping
4. Run retargeting algorithm
5. Review and adjust results

**C3D Viewer**
The C3D Viewer provides:
- 3D visualization of marker trajectories
- Analog channel plotting
- Force platform data display
- Frame-by-frame navigation
""",
        "tips": [
            "Verify marker names match your expected skeleton",
            "Use C3D format for professional motion capture data",
            "Preview data before full import to catch issues",
        ],
        "see_also": ["visualization", "analysis_tools"],
    },
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    "visualization": {
        "title": "Visualization Settings",
        "short": "Configure 3D rendering and display options",
        "description": """
The visualization system provides real-time 3D rendering of simulations.

**Camera Controls**
- Left-click + drag: Rotate view
- Right-click + drag: Pan view
- Scroll wheel: Zoom in/out
- Middle-click: Reset view

**Preset Views**
- 1: Side view (golfer's right)
- 2: Front view (face-on)
- 3: Top view (bird's eye)
- 4: Down-the-line (behind golfer)
- 5: Follow mode (tracks clubhead)

**Display Options**
- Show/hide coordinate frames
- Toggle contact point visualization
- Enable/disable shadows
- Adjust rendering quality

**Force/Torque Vectors**
- Show force vectors at joints
- Show torque arrows
- Adjust scale factor
- Color by magnitude or type

**Grid and Ground**
- Toggle ground plane
- Show distance grid
- Display target line/zone

**Performance Options**
- Render every N frames
- Reduce model complexity
- Disable shadows for speed
""",
        "tips": [
            "Use preset views for consistent analysis",
            "Enable force vectors to visualize dynamics",
            "Reduce rendering frequency for complex simulations",
        ],
        "see_also": ["simulation_controls", "analysis_tools"],
    },
    # -------------------------------------------------------------------------
    # Analysis Tools
    # -------------------------------------------------------------------------
    "analysis_tools": {
        "title": "Analysis and Plotting",
        "short": "Analyze simulation results with plots and metrics",
        "description": """
UpstreamDrift provides comprehensive analysis tools for simulation data.

**Energy Analysis**
- Kinetic energy over time
- Potential energy over time
- Total energy conservation check
- Energy transfer between segments

**Phase Diagrams**
- Position vs. velocity plots
- Joint-space trajectories
- Limit cycle analysis
- Stability visualization

**Force/Torque Analysis**
- Joint torque profiles
- Ground reaction forces
- Contact forces
- Muscle forces (if applicable)

**Kinematic Sequence**
- Proximal-to-distal sequencing
- Peak angular velocities
- Timing analysis
- X-factor metrics

**Jacobian Analysis**
- Manipulability ellipsoids
- End-effector velocity mapping
- Singularity detection
- Task-space analysis

**Export Options**
- CSV: Raw numerical data
- JSON: Structured data with metadata
- PNG/PDF: Plot images
- Video: Animated visualizations
""",
        "tips": [
            "Check energy conservation to validate simulation",
            "Phase diagrams reveal dynamic stability",
            "Kinematic sequence is key for golf swing analysis",
        ],
        "see_also": ["simulation_controls", "visualization"],
    },
    # -------------------------------------------------------------------------
    # Project Map
    # -------------------------------------------------------------------------
    "project_map": {
        "title": "Project Map",
        "short": "Complete map of all features and modules in UpstreamDrift",
        "description": """
The Project Map (docs/PROJECT_MAP.md) is a comprehensive reference for every
feature, module, and tool in the UpstreamDrift Golf Modeling Suite.

**What it covers:**
- All 11 launcher tiles and their capabilities
- All 7 physics engines with detailed features
- Gait & locomotion system (walk, run, stand, trot, crawl, bound, gallop)
- Robotics module (contact dynamics, whole-body control, sensing, planning)
- Learning & AI (RL environments, imitation learning, sim-to-real, AI assistant)
- Research modules (MPC, differentiable physics, deformable objects, multi-robot)
- Deployment (digital twin, real-time control, safety, teleoperation)
- Unreal Engine integration (streaming, VR, mesh loading)
- Shared analysis library (biomechanics, signal processing, spatial algebra)
- Tools (model explorer, humanoid builder, model generation, video analyzer)
- Visualization & plotting (10+ renderer types)
- API & web UI reference
- Hidden features not yet exposed in the UI
- Deprecated/archived code inventory

**Access:**
- Help menu > Project Map (opens the document)
- Directly at: docs/PROJECT_MAP.md
""",
        "tips": [
            "Use the Project Map to discover hidden features not in the launcher",
            "The Hidden Features table shows what can be exposed next",
            "Check the Deprecated section before working on old code",
        ],
        "see_also": ["engine_selection", "analysis_tools"],
    },
}


# ==============================================================================
# Quick Tips for UI Elements
# ==============================================================================

QUICK_TIPS: dict[str, str] = {
    # Launcher
    "launcher_search": "Type to filter models. Press Ctrl+F to focus.",
    "launcher_layout": "Click 'Edit Mode' to drag and rearrange tiles.",
    "launcher_docker": "Enable Docker mode for containerized engines.",
    "launcher_wsl": "WSL mode provides full Linux engine support.",
    # Engine tiles
    "tile_double_click": "Double-click a tile to launch immediately.",
    "tile_single_click": "Single-click to select, then click Launch.",
    # Simulation
    "sim_timestep": "Smaller timestep = more accuracy, slower speed.",
    "sim_record": "Enable recording before starting to capture data.",
    "sim_reset": "Press R to reset simulation to initial state.",
    # Visualization
    "viz_rotate": "Left-click and drag to rotate the 3D view.",
    "viz_pan": "Right-click and drag to pan.",
    "viz_zoom": "Scroll wheel to zoom in/out.",
    "viz_preset": "Press 1-5 for preset camera views.",
    # Analysis
    "analysis_export": "Click Export to save data as CSV or JSON.",
    "analysis_plot": "Double-click a plot to expand it.",
    # Motion capture
    "mocap_c3d": "C3D is the standard format for lab motion capture.",
    "mocap_video": "Video pose estimation works with standard webcam footage.",
}


# ==============================================================================
# Help Topic Registry
# ==============================================================================

HELP_TOPICS: dict[str, HelpTopic] = {
    "getting_started": HelpTopic(
        id="getting_started",
        title="Getting Started",
        short_description="Introduction to UpstreamDrift",
        help_file="getting_started.md",
        manual_section="Getting Started",
        related_topics=["engine_selection", "simulation_controls"],
    ),
    "engine_selection": HelpTopic(
        id="engine_selection",
        title="Engine Selection Guide",
        short_description="Choosing the right physics engine",
        help_file="engine_selection.md",
        manual_section="Physics Engines Guide",
        related_topics=["simulation_controls", "visualization"],
    ),
    "simulation_controls": HelpTopic(
        id="simulation_controls",
        title="Simulation Controls",
        short_description="Running and controlling simulations",
        help_file="simulation_controls.md",
        manual_section="Core Features",
        related_topics=["engine_selection", "visualization", "analysis_tools"],
    ),
    "motion_capture": HelpTopic(
        id="motion_capture",
        title="Motion Capture Integration",
        short_description="Importing and processing motion data",
        help_file="motion_capture.md",
        manual_section="Motion Capture Integration",
        related_topics=["visualization", "analysis_tools"],
    ),
    "visualization": HelpTopic(
        id="visualization",
        title="Visualization Settings",
        short_description="3D rendering and display options",
        help_file="visualization.md",
        manual_section="Visualization and Analysis",
        related_topics=["simulation_controls", "analysis_tools"],
    ),
    "analysis_tools": HelpTopic(
        id="analysis_tools",
        title="Analysis Tools",
        short_description="Analyzing simulation results",
        help_file="analysis_tools.md",
        manual_section="Visualization and Analysis",
        related_topics=["simulation_controls", "visualization"],
    ),
}


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_component_help(component_id: str) -> dict[str, Any] | None:
    """Get help content for a specific UI component.

    Args:
        component_id: The identifier of the UI component.

    Returns:
        A dictionary with help content, or None if not found.
    """
    topic_id = UI_HELP_TOPICS.get(component_id)
    if topic_id and topic_id in FEATURE_HELP:
        return FEATURE_HELP[topic_id]
    return None


def get_quick_tip(tip_id: str) -> str | None:
    """Get a quick tip by ID.

    Args:
        tip_id: The identifier of the tip.

    Returns:
        The tip text, or None if not found.
    """
    return QUICK_TIPS.get(tip_id)


def get_help_topic(topic_id: str) -> HelpTopic | None:
    """Get a help topic by ID.

    Args:
        topic_id: The topic identifier.

    Returns:
        The HelpTopic, or None if not found.
    """
    return HELP_TOPICS.get(topic_id)


def get_all_topics() -> list[HelpTopic]:
    """Get all registered help topics.

    Returns:
        A list of all HelpTopic objects.
    """
    return list(HELP_TOPICS.values())


def get_feature_help(feature_id: str) -> dict[str, Any] | None:
    """Get detailed feature help.

    Args:
        feature_id: The feature identifier.

    Returns:
        The feature help dictionary, or None if not found.
    """
    return FEATURE_HELP.get(feature_id)


def get_related_topics(topic_id: str) -> list[HelpTopic]:
    """Get related topics for a given topic.

    Args:
        topic_id: The topic identifier.

    Returns:
        A list of related HelpTopic objects.
    """
    topic = HELP_TOPICS.get(topic_id)
    if not topic or not topic.related_topics:
        return []

    related = []
    for related_id in topic.related_topics:
        related_topic = HELP_TOPICS.get(related_id)
        if related_topic:
            related.append(related_topic)
    return related


def search_help(query: str) -> list[tuple[str, str, str]]:
    """Search help content for a query string.

    Args:
        query: The search query.

    Returns:
        A list of tuples (topic_id, title, snippet) of matching content.
    """
    results = []
    query_lower = query.lower()

    for topic_id, content in FEATURE_HELP.items():
        title = content.get("title", "")
        description = content.get("description", "")

        if query_lower in title.lower() or query_lower in description.lower():
            # Extract a snippet around the match
            snippet = (
                description[:200] + "..." if len(description) > 200 else description
            )
            results.append((topic_id, title, snippet.strip()))

    return results
