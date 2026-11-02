/**
 * Authored workspace tool labels (UI review #14261).
 *
 * When present, overrides {@link formatToolTitle} for card headings and summaries.
 * Python parity can land later in `workspace_navigation.py`; this slice unblocks
 * the web workspace cards first.
 */

export type WorkspaceToolPresentation = {
  displayName: string;
  summary: string;
};

/** Primary-workspace tools with hand-written card copy. */
export const WORKSPACE_TOOL_PRESENTATION: Record<string, WorkspaceToolPresentation> =
  {
    c3d_viewer: {
      displayName: "C3D Viewer",
      summary: "Inspect motion-capture trials and marker trajectories from C3D files.",
    },
    openpose_analysis: {
      displayName: "OpenPose Analysis",
      summary: "Run 2D pose estimation workflows and review joint tracks from video.",
    },
    mediapipe_analysis: {
      displayName: "MediaPipe Analysis",
      summary: "Track body landmarks with MediaPipe and export time-series pose data.",
    },
    rate_of_closure: {
      displayName: "Rate of Closure",
      summary:
        "Open the impact and ball-flight explorer when the web bundle is built.",
    },
    motion_capture: {
      displayName: "Motion Capture",
      summary: "Configure capture pipelines and monitor live mocap sessions.",
    },
    video_analyzer: {
      displayName: "Video Analyzer",
      summary: "Review swing video with synchronized overlays and measurements.",
    },
    data_explorer: {
      displayName: "Data Explorer",
      summary: "Browse datasets, plots, and exports from prior analyses.",
    },
    model_explorer: {
      displayName: "Model Explorer",
      summary: "Inspect biomechanical models, parameters, and simulation setup.",
    },
    putting_green: {
      displayName: "Putting Green",
      summary: "Simulate putts on configurable greens and read break lines.",
    },
    golf_simulator: {
      displayName: "Golf Simulator",
      summary: "Run full-swing ball-flight scenarios from saved inputs.",
    },
    impact_explorer: {
      displayName: "Impact Explorer",
      summary: "Study impact mechanics, club delivery, and flight outcomes in the web app.",
    },
    analysis_tools_api: {
      displayName: "Analysis Tools",
      summary: "Launch comparative analysis dashboards backed by the local API.",
    },
    cross_engine_dashboard: {
      displayName: "Cross-Engine Dashboard",
      summary: "Compare engine outputs side by side with shared acceptance metrics.",
    },
  };

export function getWorkspaceToolPresentation(
  toolId: string,
): WorkspaceToolPresentation {
  const authored = WORKSPACE_TOOL_PRESENTATION[toolId];
  if (authored) {
    return authored;
  }
  const words = toolId.split("_").filter(Boolean);
  const displayName = words
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
  return {
    displayName,
    summary: `Open ${displayName} from this workspace.`,
  };
}
