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
      'canonical_core_comparison',
      'canonical_core_estimation',
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
    ],
    isPrimary: false,
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
