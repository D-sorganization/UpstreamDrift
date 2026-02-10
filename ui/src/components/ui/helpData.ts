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
