import { describe, expect, it } from "vitest";

import { PRIMARY_WORKSPACES } from "@/types/workspaceNavigation";
import {
  getWorkspaceToolPresentation,
  WORKSPACE_TOOL_PRESENTATION,
} from "@/config/workspaceToolPresentation";

describe("workspaceToolPresentation (#14261)", () => {
  it("uses authored names for reviewed capture tools", () => {
    expect(getWorkspaceToolPresentation("c3d_viewer").displayName).toBe(
      "C3D Viewer",
    );
    expect(getWorkspaceToolPresentation("openpose_analysis").displayName).toBe(
      "OpenPose Analysis",
    );
    expect(getWorkspaceToolPresentation("mediapipe_analysis").displayName).toBe(
      "MediaPipe Analysis",
    );
  });

  it("does not emit the legacy template sentence", () => {
    for (const toolId of Object.keys(WORKSPACE_TOOL_PRESENTATION)) {
      const summary = getWorkspaceToolPresentation(toolId).summary;
      expect(summary).not.toMatch(/Access and configure/i);
      expect(summary).not.toMatch(/task space/i);
    }
  });

  it("covers every primary-workspace member with non-empty copy", () => {
    const memberIds = new Set<string>();
    for (const ws of Object.values(PRIMARY_WORKSPACES)) {
      for (const toolId of ws.memberToolIds) {
        memberIds.add(toolId);
      }
    }
    for (const toolId of memberIds) {
      const { displayName, summary } = getWorkspaceToolPresentation(toolId);
      expect(displayName.trim().length).toBeGreaterThan(0);
      expect(summary.trim().length).toBeGreaterThan(0);
    }
  });
});
