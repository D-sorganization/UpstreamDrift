/**
 * Help content data ported from Python help_content.py.
 *
 * Provides structured help topics, quick tips, and search functionality
 * for the React help panel.
 *
 * See issue #1205
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HelpTopic {
  id: string;
  title: string;
  shortDescription: string;
  category: HelpCategory;
  relatedTopics: string[];
}

export interface FeatureHelp {
  title: string;
  short: string;
  description: string;
  tips: string[];
  seeAlso: string[];
}

export type HelpCategory =
  | 'getting_started'
  | 'engines'
  | 'simulation'
  | 'motion_capture'
  | 'visualization'
  | 'analysis'
  | 'tools'
  | 'settings';

// ---------------------------------------------------------------------------
// UI Component -> Help Topic Mapping
// ---------------------------------------------------------------------------

export const UI_HELP_TOPICS: Record<string, string> = {
  // Main launcher
  launcher_main: 'getting_started',
  launcher_grid: 'engine_selection',
  launcher_search: 'getting_started',
  launcher_docker: 'docker_setup',
  launcher_wsl: 'wsl_setup',
  // Engine tiles
  tile_mujoco: 'engine_selection',
  tile_drake: 'engine_selection',
  tile_pinocchio: 'engine_selection',
  tile_opensim: 'engine_selection',
  tile_myosuite: 'engine_selection',
  tile_matlab: 'matlab_integration',
  // Simulation panels
  simulation_controls: 'simulation_controls',
  simulation_parameters: 'simulation_controls',
  simulation_playback: 'simulation_controls',
  simulation_export: 'data_export',
  // Motion capture
  mocap_import: 'motion_capture',
  mocap_viewer: 'motion_capture',
  mocap_retarget: 'motion_capture',
  c3d_viewer: 'motion_capture',
  // Visualization
  viz_3d_view: 'visualization',
  viz_camera: 'visualization',
  viz_forces: 'visualization',
  viz_energy: 'analysis_tools',
  // Analysis
  analysis_plots: 'analysis_tools',
  analysis_phase: 'analysis_tools',
  analysis_energy: 'analysis_tools',
  analysis_jacobian: 'analysis_tools',
  analysis_kinematic: 'analysis_tools',
  // Tools
  urdf_generator: 'urdf_generator',
  model_explorer: 'model_explorer',
  shot_tracer: 'ball_flight',
  project_map: 'project_map',
  // Settings
  settings_general: 'configuration',
  settings_engines: 'engine_selection',
  settings_visualization: 'visualization',
};

// ---------------------------------------------------------------------------
// Help Topics Registry
// ---------------------------------------------------------------------------

export const HELP_TOPICS: Record<string, HelpTopic> = {
  getting_started: {
    id: 'getting_started',
    title: 'Getting Started',
    shortDescription: 'Introduction to UpstreamDrift',
    category: 'getting_started',
    relatedTopics: ['engine_selection', 'simulation_controls'],
  },
  engine_selection: {
    id: 'engine_selection',
    title: 'Engine Selection Guide',
    shortDescription: 'Choosing the right physics engine',
    category: 'engines',
    relatedTopics: ['simulation_controls', 'visualization'],
  },
  simulation_controls: {
    id: 'simulation_controls',
    title: 'Simulation Controls',
    shortDescription: 'Running and controlling simulations',
    category: 'simulation',
    relatedTopics: ['engine_selection', 'visualization', 'analysis_tools'],
  },
  motion_capture: {
    id: 'motion_capture',
    title: 'Motion Capture Integration',
    shortDescription: 'Importing and processing motion data',
    category: 'motion_capture',
    relatedTopics: ['visualization', 'analysis_tools'],
  },
  visualization: {
    id: 'visualization',
    title: 'Visualization Settings',
    shortDescription: '3D rendering and display options',
    category: 'visualization',
    relatedTopics: ['simulation_controls', 'analysis_tools'],
  },
  analysis_tools: {
    id: 'analysis_tools',
    title: 'Analysis Tools',
    shortDescription: 'Analyzing simulation results',
    category: 'analysis',
    relatedTopics: ['simulation_controls', 'visualization'],
  },
  project_map: {
    id: 'project_map',
    title: 'Project Map',
    shortDescription: 'Complete map of all features and modules',
    category: 'tools',
    relatedTopics: ['engine_selection', 'analysis_tools'],
  },
  docker_setup: {
    id: 'docker_setup',
    title: 'Docker Setup',
    shortDescription: 'Run engines in containerized environments',
    category: 'tools',
    relatedTopics: ['engine_selection', 'getting_started'],
  },
  wsl_setup: {
    id: 'wsl_setup',
    title: 'WSL Configuration',
    shortDescription: 'Windows Subsystem for Linux setup',
    category: 'tools',
    relatedTopics: ['engine_selection', 'getting_started'],
  },
  matlab_integration: {
    id: 'matlab_integration',
    title: 'MATLAB Integration',
    shortDescription: 'Export and analyze data in MATLAB',
    category: 'tools',
    relatedTopics: ['data_export', 'analysis_tools'],
  },
  data_export: {
    id: 'data_export',
    title: 'Data Export',
    shortDescription: 'Export simulation results in multiple formats',
    category: 'tools',
    relatedTopics: ['analysis_tools', 'matlab_integration'],
  },
  urdf_generator: {
    id: 'urdf_generator',
    title: 'URDF Generator',
    shortDescription: 'Create and edit robot model files',
    category: 'tools',
    relatedTopics: ['model_explorer', 'engine_selection'],
  },
  model_explorer: {
    id: 'model_explorer',
    title: 'Model Explorer',
    shortDescription: 'Browse and manage physics models',
    category: 'tools',
    relatedTopics: ['urdf_generator', 'visualization'],
  },
  ball_flight: {
    id: 'ball_flight',
    title: 'Ball Flight Analysis',
    shortDescription: 'Trace and analyze golf ball trajectories',
    category: 'analysis',
    relatedTopics: ['analysis_tools', 'visualization'],
  },
  configuration: {
    id: 'configuration',
    title: 'Configuration Settings',
    shortDescription: 'Configure application preferences and defaults',
    category: 'settings',
    relatedTopics: ['engine_selection', 'visualization'],
  },
};

// ---------------------------------------------------------------------------
// Feature Help Content
// ---------------------------------------------------------------------------

export const FEATURE_HELP: Record<string, FeatureHelp> = {
  engine_selection: {
    title: 'Engine Selection',
    short: 'Choose the physics engine for your simulation',
    description: `UpstreamDrift supports multiple physics engines, each with different strengths:

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

Select an engine based on your primary analysis goals.`,
    tips: [
      'MuJoCo is the easiest to install and get started with',
      'Drake excels at trajectory optimization',
      'Pinocchio is lightweight and great for prototyping',
      'MyoSuite requires MuJoCo and provides muscle simulation',
    ],
    seeAlso: ['simulation_controls', 'visualization'],
  },

  simulation_controls: {
    title: 'Simulation Controls',
    short: 'Control simulation playback and parameters',
    description: `The simulation controls allow you to run and interact with physics simulations.

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

**Keyboard Shortcuts**
- Space: Play/Pause
- R: Reset simulation
- +/-: Adjust playback speed
- Ctrl+S: Save current state
- Ctrl+E: Export data`,
    tips: [
      'Use smaller timesteps (0.001s) for accurate dynamics',
      'Enable recording before running to capture all data',
      'Pause simulation to adjust parameters without reset',
    ],
    seeAlso: ['engine_selection', 'visualization', 'analysis_tools'],
  },

  motion_capture: {
    title: 'Motion Capture Import',
    short: 'Import and process motion capture data',
    description: `UpstreamDrift supports various motion capture formats for swing analysis.

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
5. Click "Import" to load`,
    tips: [
      'Verify marker names match your expected skeleton',
      'Use C3D format for professional motion capture data',
      'Preview data before full import to catch issues',
    ],
    seeAlso: ['visualization', 'analysis_tools'],
  },

  visualization: {
    title: 'Visualization Settings',
    short: 'Configure 3D rendering and display options',
    description: `The visualization system provides real-time 3D rendering of simulations.

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
- Adjust rendering quality`,
    tips: [
      'Use preset views for consistent analysis',
      'Enable force vectors to visualize dynamics',
      'Reduce rendering frequency for complex simulations',
    ],
    seeAlso: ['simulation_controls', 'analysis_tools'],
  },

  analysis_tools: {
    title: 'Analysis and Plotting',
    short: 'Analyze simulation results with plots and metrics',
    description: `UpstreamDrift provides comprehensive analysis tools for simulation data.

**Energy Analysis**
- Kinetic energy over time
- Potential energy over time
- Total energy conservation check

**Phase Diagrams**
- Position vs. velocity plots
- Joint-space trajectories
- Limit cycle analysis

**Kinematic Sequence**
- Proximal-to-distal sequencing
- Peak angular velocities
- Timing analysis
- X-factor metrics

**Export Options**
- CSV: Raw numerical data
- JSON: Structured data with metadata
- PNG/PDF: Plot images
- Video: Animated visualizations`,
    tips: [
      'Check energy conservation to validate simulation',
      'Phase diagrams reveal dynamic stability',
      'Kinematic sequence is key for golf swing analysis',
    ],
    seeAlso: ['simulation_controls', 'visualization'],
  },

  project_map: {
    title: 'Project Map',
    short: 'Complete map of all features and modules in UpstreamDrift',
    description: `The Project Map is a comprehensive reference for every feature, module, and tool in the UpstreamDrift Golf Modeling Suite.

**What it covers:**
- All 11 launcher tiles and their capabilities
- All 7 physics engines with detailed features
- Gait & locomotion system
- Robotics module
- Learning & AI
- Deployment
- Unreal Engine integration
- Tools (model explorer, humanoid builder, model generation, video analyzer)
- Visualization & plotting
- API & web UI reference`,
    tips: [
      'Use the Project Map to discover hidden features not in the launcher',
      'The Hidden Features table shows what can be exposed next',
      'Check the Deprecated section before working on old code',
    ],
    seeAlso: ['engine_selection', 'analysis_tools'],
  },

  getting_started: {
    title: 'Getting Started with UpstreamDrift',
    short: 'Your first steps with the golf flight simulator',
    description: `Welcome to UpstreamDrift, a comprehensive golf ball flight and physics modeling suite.

**What You Can Do**
- Simulate golf swings with multiple physics engines
- Import and analyze motion capture data
- Visualize 3D trajectories and forces in real-time
- Export results for analysis in MATLAB, Excel, or custom tools
- Compare different physics engines side-by-side

**Your First Simulation**
1. Click a physics engine tile (MuJoCo recommended for beginners)
2. Load a pre-configured model or create a custom one
3. Adjust initial conditions (club speed, impact angle, spin)
4. Click "Start Simulation" to watch the ball fly
5. Use the analysis tools to examine the results

**Key Resources**
- Engine Selection Guide: Choose the right physics model
- Simulation Controls: Learn playback and recording
- Analysis Tools: Extract insights from your simulations
- Project Map: Discover all available features`,
    tips: [
      'Start with MuJoCo—it\'s the easiest to install and runs fast',
      'Use preset camera views (press 1-5) for consistent analysis angles',
      'Enable recording before running to capture full simulation data',
      'Check the Project Map to discover hidden features beyond the launcher',
    ],
    seeAlso: ['engine_selection', 'docker_setup', 'simulation_controls'],
  },

  docker_setup: {
    title: 'Docker Setup',
    short: 'Run engines in isolated containerized environments',
    description: `Docker allows you to run UpstreamDrift engines in containerized environments without modifying your system.

**Benefits**
- Isolated dependencies: Each engine runs in its own container
- Version pinning: Use exact engine versions without conflicts
- Cross-platform: Same container runs on Windows, Mac, Linux
- Clean uninstall: Simply remove container, no system files left

**Quick Start**
1. Enable Docker Mode from Settings > Docker Setup
2. Select which engines to containerize (MuJoCo, Drake, etc.)
3. Click "Build Containers" to download and prepare images
4. Launch simulations normally—containers handle the rest

**Container Management**
- View running containers from the Docker panel
- Stop/restart individual containers
- Update container images from the Engines menu
- Check container logs for debugging

**System Requirements**
- Docker Desktop installed (3.1+)
- 4GB free disk space per container
- WSL 2 backend recommended on Windows`,
    tips: [
      'Start with a single containerized engine to test your setup',
      'Container images are reused across simulations—first build takes longer',
      'Stop containers from the Docker panel to free system resources',
    ],
    seeAlso: ['engine_selection', 'getting_started', 'wsl_setup'],
  },

  wsl_setup: {
    title: 'WSL Configuration',
    short: 'Windows Subsystem for Linux setup for Linux engines',
    description: `WSL (Windows Subsystem for Linux) enables native Linux engine support on Windows without dual-boot.

**Setting Up WSL**
1. Open PowerShell as Administrator
2. Run: wsl --install
3. Restart your computer
4. Open Ubuntu from Start menu
5. Set up your Linux environment

**Configuring Engines in WSL**
- Use Settings > WSL Setup to link your WSL instance
- Select Ubuntu version (20.04 LTS, 22.04 LTS recommended)
- Let the system detect available engines
- Engines run faster in WSL than in Docker

**Accessing Files from Windows**
- WSL files live in: \\\\wsl$\\Ubuntu\\home\\username
- Map Z: drive in Windows for easy access
- Simulations save to your Windows Documents folder automatically

**Troubleshooting**
- WSL not detected: Ensure version 2 is installed (wsl --list -v)
- Performance issues: Disable Hyper-V if not needed
- File sync delays: Wait 1-2 seconds after editing files`,
    tips: [
      'WSL 2 is significantly faster than WSL 1—use version 2',
      'Create a dedicated VS Code terminal linked to your WSL instance',
      'Pin the WSL Ubuntu app to your taskbar for quick access',
    ],
    seeAlso: ['docker_setup', 'engine_selection', 'getting_started'],
  },

  matlab_integration: {
    title: 'MATLAB Integration',
    short: 'Export simulation data for analysis in MATLAB',
    description: `Export your UpstreamDrift simulation results to MATLAB for advanced analysis and custom visualization.

**Export Formats**
- .mat files: Binary format preserving structure and types
- .csv files: Simple tabular data (single result sets)
- .json files: Structured hierarchical data with metadata

**Exporting from UpstreamDrift**
1. Run a simulation and pause at the desired frame
2. Click Export > MATLAB Format
3. Choose data to include: trajectories, forces, energies, joint angles
4. Select resolution (every nth frame for large datasets)
5. Save to your preferred location

**Working with .mat Files in MATLAB**
\`\`\`matlab
% Load exported data
data = load('simulation_export.mat');

% Access trajectories (position over time)
trajectory = data.ballPosition; % Nx3 array [x, y, z]

% Access forces
forces = data.contactForces; % Nx3 array [Fx, Fy, Fz]

% Plot trajectory
plot3(trajectory(:,1), trajectory(:,2), trajectory(:,3));
\`\`\`

**Batch Export**
- Use the CLI: python -m upstream_drift export --engine mujoco --format mat
- Creates separate .mat files for each simulation`,
    tips: [
      'Always verify data ranges in MATLAB match your simulation expectations',
      'Use .csv for sharing data with non-MATLAB collaborators',
      'Reduce resolution for large simulations to keep file size manageable',
    ],
    seeAlso: ['data_export', 'analysis_tools'],
  },

  data_export: {
    title: 'Data Export',
    short: 'Save simulation results in CSV, JSON, or MATLAB formats',
    description: `Export your simulation results for external analysis, sharing, and long-term archival.

**Export Formats**
- CSV: Spreadsheet-compatible tabular format
  - Best for: Simple analysis, Excel/Pandas
  - Contains: Time series for position, velocity, forces

- JSON: Structured hierarchical format
  - Best for: Web applications, API integration
  - Contains: Full metadata, nested joint data, contact points

- MATLAB: Binary .mat files
  - Best for: Advanced analysis, custom visualizations
  - Contains: All arrays preserved with exact precision

**What Gets Exported**
- Trajectory data: X, Y, Z position at each timestep
- Velocities: Linear and angular velocities
- Forces: Contact forces, applied forces
- Energy: Kinetic, potential, total energy
- Joint angles: Rotation and angular velocity per joint
- Metadata: Simulation parameters, engine version, timestamps

**Exporting**
1. Click Export in the simulation controls
2. Select format (CSV/JSON/MATLAB)
3. Choose data scope: Full simulation or custom time range
4. Adjust resolution: Every frame or sample at N Hz
5. Click Export and choose save location

**Batch Export**
- Export multiple simulations at once
- Use template to match column/field names across runs
- Schedule exports from the API`,
    tips: [
      'CSV exports are fastest for quick data inspection',
      'Use JSON for preserving nested structure (joint hierarchies)',
      'High-frequency data (sampling at 500+ Hz) creates large files—resample if needed',
    ],
    seeAlso: ['matlab_integration', 'analysis_tools'],
  },

  urdf_generator: {
    title: 'URDF Generator',
    short: 'Create and edit robot model description files',
    description: `URDF (Unified Robot Description Format) files define robot/model geometry, physics, and visual properties.

**Creating a New URDF**
1. Click URDF Generator from launcher
2. Choose a template: Empty, Human Body, Golf Club, etc.
3. Add links (rigid bodies) and joints
4. Attach visual meshes and collision shapes
5. Set physics properties: mass, inertia
6. Save and validate

**URDF Components**
- Links: Rigid bodies with mass, inertia, geometry
- Joints: Connections between links (revolute, prismatic, fixed)
- Visual: How the model appears in rendering
- Collision: How physics engines compute contact

**Editing an Existing URDF**
1. Click Model Explorer > Open URDF
2. Select a .urdf file
3. Modify properties in the editor
4. See changes in real-time 3D preview
5. Validate for physics engines before saving

**Common Tasks**
- Add a new joint: Right-click link, "Add Joint"
- Import mesh: Drag .stl/.obj file onto link
- Adjust mass: Select link, modify mass slider
- Test in engine: Click "Test in MuJoCo" to validate

**Tips for Good URDFs**
- Keep inertias realistic (sphere rule: I = 0.4*m*r²)
- Close kinematic chains carefully (can cause numerical issues)
- Validate in at least two engines before deployment`,
    tips: [
      'Start with templates—they have correct structure and valid defaults',
      'Use low-poly meshes (<10k triangles) for fast simulation',
      'Test URDFs in MuJoCo first—most forgiving with marginal definitions',
    ],
    seeAlso: ['model_explorer', 'engine_selection'],
  },

  model_explorer: {
    title: 'Model Explorer',
    short: 'Browse, manage, and test physics models',
    description: `The Model Explorer is your hub for managing URDF models, inspecting properties, and testing compatibility.

**Browsing Models**
1. Click Model Explorer from launcher
2. Browse built-in models (human, golf clubs, balls, etc.)
3. Filter by category: Anatomical, Sports, Robots
4. Search by property: "mass > 2kg", "has_hands", etc.

**Model Properties**
- Overview: Links, joints, total mass, degrees of freedom
- Geometry: Mesh files, collision shapes
- Physics: Mass distribution, inertia matrix
- Materials: Friction, restitution, damping
- Kinematics: Joint ranges, default positions

**Testing Compatibility**
- Select model > "Test in Engine"
- Choose physics engine (MuJoCo, Drake, Pinocchio, etc.)
- See validation result: Green (compatible), Yellow (warnings), Red (errors)
- Check detailed error log

**Common Issues**
- "Floating base not supported": Drake requires explicit base frame
- "Inertia matrix not positive definite": Fix with Physics > Validate
- "Mesh file not found": Re-link mesh from URDF Generator

**Organizing Models**
- Create custom folders in Documents/UpstreamDrift/models/
- Tag models with custom keywords
- Star frequently-used models for quick access
- Export model info as JSON for backup`,
    tips: [
      'Use the validation tool before running long simulations',
      'Create a backup copy before editing important models',
      'Filter by engine compatibility to find usable models quickly',
    ],
    seeAlso: ['urdf_generator', 'visualization', 'engine_selection'],
  },

  ball_flight: {
    title: 'Ball Flight Analysis',
    short: 'Trace and analyze golf ball trajectories',
    description: `The Ball Flight tool provides detailed trajectory analysis, aerodynamics, and comparison features.

**Launching Ball Flight Analysis**
1. Run a golf shot simulation (any engine)
2. Click the shot tracer icon during playback
3. Or load a saved shot from Results
4. The ball trajectory is automatically extracted

**Trajectory Display**
- 3D flight path: Shows full ball trajectory in 3D space
- Launch angle indicator: Initial angle in vertical and horizontal planes
- Spin rate: Ball rotation rate during flight (RPM)
- Carry distance: Distance traveled in air
- Roll distance: Additional distance after landing

**Aerodynamic Metrics**
- Drag coefficient: How air resistance affects flight
- Lift coefficient: Magnus effect on the ball
- Spin axis: Direction of primary spin rotation
- Peak height: Maximum altitude of the ball

**Analysis Features**
- Compare multiple shots side-by-side
- Create overlay view to visualize differences
- Mark waypoints to measure intermediate distances
- Export as trajectory file for external analysis

**Preset Views**
- Face-on (XY plane): Curvature left/right
- Down-the-line (ZY plane): Height and distance
- Top-down (XZ plane): Full flight path
- 3D: Interactive rotation

**Export Options**
- CSV: Frame-by-frame trajectory data
- JSON: Full aerodynamic analysis
- Video: Animated trajectory with metrics
- Image: Snapshot with measurements`,
    tips: [
      'Enable 1-frame display for clearest flight path visualization',
      'Use overlay comparison to tune swing parameters',
      'Check spin axis—should match your swing type (draw/fade spin)',
    ],
    seeAlso: ['analysis_tools', 'visualization'],
  },

  configuration: {
    title: 'Configuration Settings',
    short: 'Adjust application preferences and default behaviors',
    description: `Customize UpstreamDrift to match your workflow and system capabilities.

**General Settings**
- Default physics engine: What loads when you click a tile
- Default camera view: Which preset view opens first (side, face-on, etc.)
- Auto-save interval: Save state every N minutes (0 to disable)
- Theme: Light, dark, or system default

**Engine Preferences**
- Engine install paths: Custom locations for Drake, OpenSim, etc.
- Parallel processing: Enable multi-core simulation
- GPU acceleration: CUDA/Metal support (if available)
- Memory limit: Cap RAM usage per simulation

**Visualization**
- Rendering quality: Low (fast), Medium, High, Ultra
- Anti-aliasing: Off, FXAA, or MSAA
- Shadows: Off, blob, or ray-traced
- Frame rate limit: 30, 60, 120, or unlimited fps

**Performance**
- Data recording rate: 60, 120, 240 Hz (higher = larger files)
- Playback frame rate: Match recording or custom
- Mesh decimation: Simplify complex models for speed
- Background processes: Keep engines running or auto-close

**File Locations**
- Models directory: Where custom URDF files are stored
- Exports directory: Default save location for data files
- Cache directory: Temporary files, mesh cache
- Projects directory: Save and load simulation sets

**Advanced**
- Log level: Debug, Info, Warning, Error
- Network: Enable API server, set port
- Profiling: Collect performance metrics
- Reset to defaults: Restore factory settings`,
    tips: [
      'Lower rendering quality to 30 fps for smoother long simulations',
      'Set data recording to match your analysis needs (120 Hz is typical)',
      'Create a symbolic link to sync models across machines',
    ],
    seeAlso: ['engine_selection', 'visualization'],
  },
};

// ---------------------------------------------------------------------------
// Quick Tips
// ---------------------------------------------------------------------------

export const QUICK_TIPS: Record<string, string> = {
  launcher_search: 'Type to filter models. Press Ctrl+F to focus.',
  launcher_layout: "Click 'Edit Mode' to drag and rearrange tiles.",
  launcher_docker: 'Enable Docker mode for containerized engines.',
  launcher_wsl: 'WSL mode provides full Linux engine support.',
  tile_double_click: 'Double-click a tile to launch immediately.',
  tile_single_click: 'Single-click to select, then click Launch.',
  sim_timestep: 'Smaller timestep = more accuracy, slower speed.',
  sim_record: 'Enable recording before starting to capture data.',
  sim_reset: 'Press R to reset simulation to initial state.',
  viz_rotate: 'Left-click and drag to rotate the 3D view.',
  viz_pan: 'Right-click and drag to pan.',
  viz_zoom: 'Scroll wheel to zoom in/out.',
  viz_preset: 'Press 1-5 for preset camera views.',
  analysis_export: 'Click Export to save data as CSV or JSON.',
  analysis_plot: 'Double-click a plot to expand it.',
  mocap_c3d: 'C3D is the standard format for lab motion capture.',
  mocap_video: 'Video pose estimation works with standard webcam footage.',
};

// ---------------------------------------------------------------------------
// Category Labels
// ---------------------------------------------------------------------------

export const CATEGORY_LABELS: Record<HelpCategory, string> = {
  getting_started: 'Getting Started',
  engines: 'Physics Engines',
  simulation: 'Simulation',
  motion_capture: 'Motion Capture',
  visualization: 'Visualization',
  analysis: 'Analysis',
  tools: 'Tools',
  settings: 'Settings',
};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/**
 * Get help content for a specific UI component.
 */
export function getComponentHelp(componentId: string): FeatureHelp | null {
  const topicId = UI_HELP_TOPICS[componentId];
  if (topicId && FEATURE_HELP[topicId]) {
    return FEATURE_HELP[topicId];
  }
  return null;
}

/**
 * Get a quick tip by ID.
 */
export function getQuickTip(tipId: string): string | null {
  return QUICK_TIPS[tipId] ?? null;
}

/**
 * Search help content for a query string.
 * Returns matching topics sorted by relevance.
 */
export function searchHelp(query: string): Array<{
  topicId: string;
  title: string;
  snippet: string;
}> {
  const queryLower = query.toLowerCase();
  const results: Array<{ topicId: string; title: string; snippet: string }> = [];

  for (const [topicId, content] of Object.entries(FEATURE_HELP)) {
    const titleMatch = content.title.toLowerCase().includes(queryLower);
    const descMatch = content.description.toLowerCase().includes(queryLower);
    const tipsMatch = content.tips.some((t) => t.toLowerCase().includes(queryLower));

    if (titleMatch || descMatch || tipsMatch) {
      const snippet =
        content.description.length > 200
          ? content.description.substring(0, 200) + '...'
          : content.description;
      results.push({ topicId, title: content.title, snippet: snippet.trim() });
    }
  }

  return results;
}

/**
 * Get all topics grouped by category.
 */
export function getTopicsByCategory(): Record<HelpCategory, HelpTopic[]> {
  const grouped = {} as Record<HelpCategory, HelpTopic[]>;

  for (const cat of Object.keys(CATEGORY_LABELS) as HelpCategory[]) {
    grouped[cat] = [];
  }

  for (const topic of Object.values(HELP_TOPICS)) {
    if (grouped[topic.category]) {
      grouped[topic.category].push(topic);
    }
  }

  return grouped;
}

/**
 * Get related topics for a given topic ID.
 */
export function getRelatedTopics(topicId: string): HelpTopic[] {
  const topic = HELP_TOPICS[topicId];
  if (!topic) return [];

  return topic.relatedTopics
    .map((id) => HELP_TOPICS[id])
    .filter((t): t is HelpTopic => t !== undefined);
}
