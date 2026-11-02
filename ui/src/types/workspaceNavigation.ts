/**
 * Task-oriented workspace navigation definitions (ORG-06, #10516).
 *
 * Aligns React/Tauri desktop navigation with the authoritative Python catalog
 * (src/launchers/workspace_navigation.py).
 */

export interface WorkspaceMetadata {
  id: string;
  slug: string;
  title: string;
  description: string;
  iconName: string;
  route: string;
  memberToolIds: readonly string[];
  isPrimary: boolean;
}

export const PRIMARY_WORKSPACES: Record<string, WorkspaceMetadata> = {
  capture_analyze: {
    id: 'capture_analyze',
    slug: 'capture-analyze',
    title: 'Capture & Analyze',
    description: 'Motion capture, video analysis, pose tracking, and time-series data',
    iconName: 'camera',
    route: '/workspaces/capture-analyze',
    memberToolIds: [
      'motion_capture',
      'video_analyzer',
      'c3d_viewer',
      'openpose_analysis',
      'mediapipe_analysis',
      'data_explorer',
      'data_processor',
      'rate_of_closure',
      'capture_rig',
    ],
    isPrimary: true,
  },
  model_match: {
    id: 'model_match',
    slug: 'model-match',
    title: 'Model & Match',
    description: 'Physics engines, biomechanical models, model exploration, and tour matching',
    iconName: 'computer',
    route: '/workspaces/model-match',
    memberToolIds: [
      'mujoco_unified',
      'drake_golf',
      'pinocchio_golf',
      'opensim_golf',
      'myosim_suite',
      'model_explorer',
      'motion_target_preview',
      'tour_matching_viewer',
      'starting_pose_matcher',
    ],
    isPrimary: true,
  },
  shot_course_lab: {
    id: 'shot_course_lab',
    slug: 'shot-course-lab',
    title: 'Shot & Course Lab',
    description: 'Putting greens, launch monitors, bunker shots, impact, and course terrain',
    iconName: 'sports_golf',
    route: '/workspaces/shot-course-lab',
    memberToolIds: [
      'putting_green',
      'golf_simulator',
      'bunkershot3d',
      'impact_explorer',
      'terrain',
      'swingset',
      'putting_green_gui',
    ],
    isPrimary: true,
  },
  optimize_train: {
    id: 'optimize_train',
    slug: 'optimize-train',
    title: 'Optimize & Train',
    description: 'Trajectory optimization, objective exploration, neural retargeting, and control',
    iconName: 'build',
    route: '/workspaces/optimize-train',
    memberToolIds: [
      'tools_movement_optimizer',
      'movement_optimizer',
      'swing_objective_lab',
      'training',
      'pid_generator',
      'pendulum_simulator',
      'sg_optimizer',
    ],
    isPrimary: true,
  },
  results_compare: {
    id: 'results_compare',
    slug: 'results-compare',
    title: 'Results & Compare',
    description: 'Cross-engine comparative dashboards, canonical comparison, and synthetic datasets',
    iconName: 'assessment',
    route: '/workspaces/results-compare',
    memberToolIds: [
      'cross_engine_dashboard',
      'cross_engine',
      'analysis_tools_api',
      'dataset_generator',
      'matlab_suite',
      'matlab_unified',
    ],
    isPrimary: true,
  },
};

export const SECONDARY_WORKSPACES: Record<string, WorkspaceMetadata> = {
  all_tools: {
    id: 'all_tools',
    slug: 'all-tools',
    title: 'All Tools',
    description: 'Searchable catalog of all available tools and engines',
    iconName: 'grid_view',
    route: '/workspaces/all-tools',
    memberToolIds: [],
    isPrimary: false,
  },
  favorites: {
    id: 'favorites',
    slug: 'favorites',
    title: 'Favorites',
    description: 'User-starred favorites',
    iconName: 'star',
    route: '/workspaces/favorites',
    memberToolIds: [],
    isPrimary: false,
  },
  history: {
    id: 'history',
    slug: 'history',
    title: 'History',
    description: 'Recently and frequently launched tools',
    iconName: 'history',
    route: '/workspaces/history',
    memberToolIds: [],
    isPrimary: false,
  },
  dev_research: {
    id: 'dev_research',
    slug: 'dev-research',
    title: 'Developer & Research',
    description: 'Developer utilities, research models, and headless integration services',
    iconName: 'code',
    route: '/workspaces/dev-research',
    memberToolIds: [
      'character_builder',
      'aip',
      'realtime_ws',
      'actuator_controls',
      'robotics_module',
      'unreal_integration',
      'perturbation_analysis',
      'force_overlays',
      'motion_pipeline',
      // Planned tools (#14262): reachable here, never from primary workspaces or home.
      'canonical_core_comparison',
      'canonical_core_estimation',
    ],
    isPrimary: false,
  },
};

/** Hand-written card copy for a workspace member tool (#14261). */
export interface ToolCardCopy {
  displayName: string;
  summary: string;
}

const MOVEMENT_OPTIMIZER: ToolCardCopy = {
  displayName: 'Movement Optimizer',
  summary: 'Optimize barbell and swing trajectories across physics engines.',
};
const PUTTING_GREEN: ToolCardCopy = {
  displayName: 'Putting Green',
  summary: 'Author greens and practice putts with ball-rolling physics.',
};
const CROSS_ENGINE_DASHBOARD: ToolCardCopy = {
  displayName: 'Cross-Engine Dashboard',
  summary: 'Compare perturbation robustness across physics engines.',
};
const MATLAB_MODELS: ToolCardCopy = {
  displayName: 'MATLAB Models',
  summary: 'MATLAB and Simscape models, dataset generators and tooling.',
};

/**
 * Card copy for every workspace member, keyed by tool id. Names are written
 * as authored (C3D, OpenPose, MediaPipe) rather than title-cased from the id,
 * and legacy alias ids share their canonical tool's copy. A test requires an
 * entry for every `memberToolIds` value.
 */
export const TOOL_CARD_COPY: Record<string, ToolCardCopy> = {
  motion_capture: {
    displayName: 'Motion Capture',
    summary: 'Record and process markerless motion capture from the command line.',
  },
  video_analyzer: {
    displayName: 'Video Analyzer',
    summary: 'Extract poses and joint tracks from swing video.',
  },
  c3d_viewer: {
    displayName: 'C3D Viewer',
    summary: 'Visualize and analyze optical mocap data (.c3d).',
  },
  openpose_analysis: {
    displayName: 'OpenPose',
    summary: 'High-accuracy body pose estimation (academic license).',
  },
  mediapipe_analysis: {
    displayName: 'MediaPipe',
    summary: 'Fast pose estimation with Google MediaPipe.',
  },
  data_explorer: {
    displayName: 'Data Explorer',
    summary: 'Import, filter and visualize simulation datasets.',
  },
  data_processor: {
    displayName: 'Data Processor',
    summary: 'Filter, transform and summarize recorded signals.',
  },
  rate_of_closure: {
    displayName: 'Rate of Closure',
    summary: 'See how impact-point delivery differs from the tracked clubhead point.',
  },
  capture_rig: {
    displayName: 'Capture Rig',
    summary: 'Multi-camera capture, pose ingest and reconstruction.',
  },
  mujoco_unified: {
    displayName: 'MuJoCo',
    summary: 'Humanoid golf swing simulation and analysis.',
  },
  drake_golf: {
    displayName: 'Drake',
    summary: 'Drake robotics dynamics with MeshCat visualization.',
  },
  pinocchio_golf: {
    displayName: 'Pinocchio',
    summary: 'Rigid-body dynamics with analytical derivatives.',
  },
  opensim_golf: {
    displayName: 'OpenSim',
    summary: 'Musculoskeletal modeling for biomechanics.',
  },
  myosim_suite: {
    displayName: 'MyoSuite',
    summary: 'Muscle-actuated engine status; simulation needs the optional myosuite package.',
  },
  model_explorer: {
    displayName: 'Model Explorer',
    summary: 'Browse models and generate URDF and MJCF files.',
  },
  motion_target_preview: {
    displayName: 'Motion-Match Preview',
    summary: 'Preview motion-capture targets on a shared time grid.',
  },
  tour_matching_viewer: {
    displayName: 'Tour Matching Viewer',
    summary: 'Replay full-body matching candidates against tour capture in 3D.',
  },
  starting_pose_matcher: {
    displayName: 'Starting-Pose Matcher',
    summary: 'Legacy entry for Motion-Match Preview.',
  },
  putting_green: PUTTING_GREEN,
  putting_green_gui: PUTTING_GREEN,
  golf_simulator: {
    displayName: 'Golf Simulator Console',
    summary: 'Destination controls and replay submission for a connected simulator.',
  },
  bunkershot3d: {
    displayName: 'BunkerShot3D',
    summary: 'Design wedge soles against a dynamic bunker sand model.',
  },
  impact_explorer: {
    displayName: 'Impact Explorer',
    summary: 'Explore club-ball impact and the resulting ball flight.',
  },
  terrain: {
    displayName: 'Terrain',
    summary: 'Terrain presets, materials and the active course surface.',
  },
  swingset: {
    displayName: 'SwingSet',
    summary: 'Listed in this workspace but not yet registered as a launcher tool.',
  },
  tools_movement_optimizer: MOVEMENT_OPTIMIZER,
  movement_optimizer: MOVEMENT_OPTIMIZER,
  swing_objective_lab: {
    displayName: 'Swing Objective Lab',
    summary: 'Compare downswings optimized for speed against other objectives.',
  },
  training: {
    displayName: 'Training Dashboard',
    summary: 'Training-controller dashboard; the desktop view is still in progress.',
  },
  pendulum_simulator: {
    displayName: 'Pendulum Simulator',
    summary: 'Interactive double, triple and 8-DOF golfer pendulum dynamics.',
  },
  sg_optimizer: {
    displayName: 'Strokes Gained Optimizer',
    summary: 'Strokes-gained library and CLI; the desktop UI is planned.',
  },
  cross_engine_dashboard: CROSS_ENGINE_DASHBOARD,
  cross_engine: CROSS_ENGINE_DASHBOARD,
  analysis_tools_api: {
    displayName: 'Analysis Tools',
    summary: 'REST access to swing-metric and biomechanics analysis endpoints.',
  },
  dataset_generator: {
    displayName: 'Dataset Generator',
    summary: 'Generate, import and export swing datasets.',
  },
  matlab_suite: MATLAB_MODELS,
  matlab_unified: MATLAB_MODELS,
  character_builder: {
    displayName: 'Character Builder',
    summary: 'Build a golfer from stature, mass and proportions and export models.',
  },
  aip: {
    displayName: 'AI Protocol (AIP)',
    summary: 'Structured method dispatch for AI-driven simulation control.',
  },
  realtime_ws: {
    displayName: 'Realtime WebSocket',
    summary: 'Live simulation data stream over WebSocket pub-sub.',
  },
  actuator_controls: {
    displayName: 'Actuator Controls',
    summary: 'Tune actuator parameters and edit control schemes live.',
  },
  robotics_module: {
    displayName: 'Robotics Module',
    summary: 'Locomotion planning, contact mechanics and humanoid control.',
  },
  pid_generator: {
    displayName: 'P&ID Generator',
    summary: 'Create ISA-5.1 piping and instrumentation diagrams.',
  },
  unreal_integration: {
    displayName: 'Unreal Integration',
    summary: 'Unreal Engine streaming and VR visualization.',
  },
  perturbation_analysis: {
    displayName: 'Perturbation Analysis',
    summary: 'Cross-engine robustness analysis through perturbation testing.',
  },
  force_overlays: {
    displayName: 'Force Overlays',
    summary: 'Joint forces and torques overlaid on the 3D model.',
  },
  motion_pipeline: {
    displayName: 'Motion Pipeline',
    summary: 'Markerless mocap to forward dynamics: video in, motion out.',
  },
  canonical_core_comparison: {
    displayName: 'Canonical-Core Comparison',
    summary: 'Planned cross-engine canonical-state comparison; no service yet.',
  },
  canonical_core_estimation: {
    displayName: 'Canonical-Core Estimation',
    summary: 'Planned canonical-state estimation; no service yet.',
  },
};

export const ALL_WORKSPACES: Record<string, WorkspaceMetadata> = {
  ...PRIMARY_WORKSPACES,
  ...SECONDARY_WORKSPACES,
};

export function isPrimaryWorkspace(workspaceId: string): boolean {
  return workspaceId in PRIMARY_WORKSPACES;
}

export function getWorkspaceById(workspaceId: string): WorkspaceMetadata | undefined {
  return ALL_WORKSPACES[workspaceId];
}

export function getWorkspaceBySlug(slug: string): WorkspaceMetadata | undefined {
  return Object.values(ALL_WORKSPACES).find((w) => w.slug === slug);
}

export function getWorkspaceForTool(toolId: string): string | null {
  for (const [id, ws] of Object.entries(PRIMARY_WORKSPACES)) {
    if (ws.memberToolIds.includes(toolId)) {
      return id;
    }
  }
  if (SECONDARY_WORKSPACES.dev_research.memberToolIds.includes(toolId)) {
    return 'dev_research';
  }
  return null;
}
