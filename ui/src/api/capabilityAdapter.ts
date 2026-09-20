/**
 * Route-to-capability adapter for React and Tauri (ORG-06, #10516).
 *
 * Resolves tool identities into concrete, honest launch actions depending
 * on runtime environment (Tauri desktop app vs Web browser).
 */

import { canLaunchNativeWindow } from './webLaunch';

export type WorkspaceToolAction =
  | { kind: 'navigate'; route: string; title: string }
  | { kind: 'native-launch'; title: string; explanation: string }
  | {
      kind: 'native-only-explanation';
      title: string;
      badge: string;
      explanation: string;
      alternativeRoute?: string;
    };

/**
 * Known web route map for tools that have in-app React pages.
 */
export const TOOL_WEB_ROUTES: Record<string, string> = {
  video_analyzer: '/tools/video-analyzer',
  data_explorer: '/tools/data-explorer',
  putting_green: '/tools/putting-green',
  putting_green_gui: '/tools/putting-green',
  model_explorer: '/tools/model-explorer',
  impact_explorer: '/tools/impact-explorer',
  motion_capture: '/tools/motion-capture',
  terrain: '/tools/terrain',
  dataset_generator: '/tools/dataset',
  analysis_tools_api: '/tools/analysis',
  character_builder: '/tools/character-builder',
  canonical_core_estimation: '/tools/canonical-core/estimation',
  canonical_core_comparison: '/tools/canonical-core/comparison',
  swing_objective_lab: '/tools/swing-objective-lab',
  golf_simulator: '/tools/golf-simulator',
  rate_of_closure: '/ball-flight',
};

/**
 * Format a canonical tool ID to a readable title.
 */
export function formatToolTitle(toolId: string): string {
  return toolId
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export interface ResolveActionOptions {
  isTauri?: boolean;
}

/**
 * Resolve the action for launching or viewing a tool within a workspace.
 */
export function resolveWorkspaceToolAction(
  toolId: string,
  options?: ResolveActionOptions,
): WorkspaceToolAction {
  const isTauriEnv = options?.isTauri ?? canLaunchNativeWindow();
  const title = formatToolTitle(toolId);

  // Check if tool has an embedded in-app web route
  const route = TOOL_WEB_ROUTES[toolId];
  if (route) {
    return {
      kind: 'navigate',
      route,
      title,
    };
  }

  // Desktop native-only tool (e.g. MuJoCo, Drake, Pinocchio, OpenSim, MATLAB)
  if (isTauriEnv) {
    return {
      kind: 'native-launch',
      title,
      explanation: `Launches ${title} in a native desktop window via Tauri/local host.`,
    };
  }

  // Web browser fallback
  let alternativeRoute: string | undefined = '/simulation';
  if (toolId.includes('course') || toolId.includes('terrain')) {
    alternativeRoute = '/tools/terrain';
  } else if (toolId.includes('optimizer') || toolId.includes('training')) {
    alternativeRoute = '/tools/swing-objective-lab';
  }

  return {
    kind: 'native-only-explanation',
    title,
    badge: 'Desktop app only',
    explanation: `${title} runs as a native desktop application with C++ physics bindings. Run via the Tauri desktop app or connect to a local workstation backend.`,
    alternativeRoute,
  };
}
