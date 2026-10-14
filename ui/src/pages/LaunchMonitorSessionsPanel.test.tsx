/**
 * Tests for the Launch Monitor Analytics Sessions panel (#11987).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorSessionsPanel } from "./LaunchMonitorSessionsPanel";
import type {
  ImportedSessionResponse,
  ImportPreviewResponse,
} from "@/api/launchMonitorAnalytics";

const SAMPLE_PREVIEW: ImportPreviewResponse = {
  filename: "trackman_session.csv",
  headers: ["Club Speed", "Ball Speed", "Launch Angle"],
  detected_profile_id: "trackman",
  detection_confidence: 0.95,
  profiles: [
    { profile_id: "trackman", vendor: "TrackMan" },
    { profile_id: "flightscope", vendor: "FlightScope" },
  ],
  auto_mappings: {},
  targets: ["club_speed", "ball_speed", "launch_angle"],
  measurement_statuses: ["reported", "calculated"],
};

const SAMPLE_IMPORTED: ImportedSessionResponse = {
  session_id: "sess_12345",
  name: "trackman_session",
  manifest: {
    vendor: "TrackMan",
    profile_id: "trackman",
    source_path: "trackman_session.csv",
  },
  columns: ["club_speed", "ball_speed", "launch_angle"],
  records: [
    { club_speed: 105.2, ball_speed: 155.4, launch_angle: 12.1 },
    { club_speed: 104.8, ball_speed: 154.9, launch_angle: 11.8 },
  ],
  row_count: 2,
};

const { previewImportV2Mock, importSessionV2Mock } = vi.hoisted(() => ({
  previewImportV2Mock: vi.fn(),
  importSessionV2Mock: vi.fn(),
}));

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    previewImportV2: previewImportV2Mock,
    importSessionV2: importSessionV2Mock,
  };
});

describe("LaunchMonitorSessionsPanel", () => {
  beforeEach(() => {
    previewImportV2Mock.mockReset();
    importSessionV2Mock.mockReset();
  });

  it("disables the import button when no file is selected", () => {
    render(<LaunchMonitorSessionsPanel />);

    expect(screen.getByTestId("lms-import-btn")).toBeDisabled();
    expect(screen.getByTestId("lms-status")).toHaveTextContent(
      "Select a launch-monitor export file to preview or import a session.",
    );
  });

  it("selecting a file calls previewImportV2 and populates the detected profile", async () => {
    previewImportV2Mock.mockResolvedValue(SAMPLE_PREVIEW);
    const user = userEvent.setup();
    render(<LaunchMonitorSessionsPanel />);

    const file = new File(
      ["Club Speed,Ball Speed\n100,150"],
      "trackman_session.csv",
      { type: "text/csv" },
    );
    const fileInput = screen.getByTestId("lms-file-input");
    await user.upload(fileInput, file);

    await waitFor(() => {
      expect(previewImportV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(previewImportV2Mock).toHaveBeenCalledWith(file);

    const select = screen.getByTestId("lms-profile-select") as HTMLSelectElement;
    await waitFor(() => {
      expect(select.value).toBe("trackman");
    });
    expect(screen.getByTestId("lms-status")).toHaveTextContent("Preview ready.");
  });

  it('clicking "Import Session" calls importSessionV2, displays the imported session in lms-sessions-list, and invokes onSessionImported', async () => {
    previewImportV2Mock.mockResolvedValue(SAMPLE_PREVIEW);
    importSessionV2Mock.mockResolvedValue(SAMPLE_IMPORTED);
    const onSessionImported = vi.fn();
    const user = userEvent.setup();

    render(<LaunchMonitorSessionsPanel onSessionImported={onSessionImported} />);

    const file = new File(
      ["Club Speed,Ball Speed\n100,150"],
      "trackman_session.csv",
      { type: "text/csv" },
    );
    await user.upload(screen.getByTestId("lms-file-input"), file);

    await waitFor(() => {
      expect(screen.getByTestId("lms-import-btn")).toBeEnabled();
    });

    await user.type(screen.getByTestId("lms-session-name"), "My Session");
    await user.type(screen.getByTestId("lms-player"), "Pro Golfer");

    await user.click(screen.getByTestId("lms-import-btn"));

    await waitFor(() => {
      expect(importSessionV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(importSessionV2Mock).toHaveBeenCalledWith(file, {
      profile_id: "trackman",
      session_name: "My Session",
      player: "Pro Golfer",
      monitor_model: null,
    });

    expect(onSessionImported).toHaveBeenCalledTimes(1);
    expect(onSessionImported).toHaveBeenCalledWith(SAMPLE_IMPORTED);

    const list = screen.getByTestId("lms-sessions-list");
    expect(list).toHaveTextContent("trackman_session");
    expect(list).toHaveTextContent("sess_12345");
    expect(list).toHaveTextContent("2 shot(s)");
    expect(list).toHaveTextContent("TrackMan");

    expect(screen.getByTestId("lms-status")).toHaveTextContent("Session imported.");
  });

  it("clicking the remove button removes the session from lms-sessions-list", async () => {
    previewImportV2Mock.mockResolvedValue(SAMPLE_PREVIEW);
    importSessionV2Mock.mockResolvedValue(SAMPLE_IMPORTED);
    const user = userEvent.setup();

    render(<LaunchMonitorSessionsPanel />);

    const file = new File(["dummy"], "trackman_session.csv", { type: "text/csv" });
    await user.upload(screen.getByTestId("lms-file-input"), file);

    await waitFor(() => {
      expect(screen.getByTestId("lms-import-btn")).toBeEnabled();
    });
    await user.click(screen.getByTestId("lms-import-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("lms-remove-sess_12345")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("lms-remove-sess_12345"));

    expect(screen.queryByTestId("lms-remove-sess_12345")).not.toBeInTheDocument();
    expect(screen.getByTestId("lms-sessions-list")).not.toHaveTextContent("sess_12345");
  });

  it("renders API errors cleanly in lms-status", async () => {
    previewImportV2Mock.mockRejectedValue(new Error("Corrupt export file"));
    const user = userEvent.setup();

    render(<LaunchMonitorSessionsPanel />);

    const file = new File(["bad data"], "corrupt.csv", { type: "text/csv" });
    await user.upload(screen.getByTestId("lms-file-input"), file);

    await waitFor(() => {
      expect(screen.getByTestId("lms-status")).toHaveTextContent(
        "Session import failed: Corrupt export file",
      );
    });

    previewImportV2Mock.mockResolvedValue(SAMPLE_PREVIEW);
    importSessionV2Mock.mockRejectedValue(new Error("Validation failed"));

    const validFile = new File(["ok"], "ok.csv", { type: "text/csv" });
    await user.upload(screen.getByTestId("lms-file-input"), validFile);

    await waitFor(() => {
      expect(screen.getByTestId("lms-import-btn")).toBeEnabled();
    });
    await user.click(screen.getByTestId("lms-import-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("lms-status")).toHaveTextContent(
        "Session import failed: Validation failed",
      );
    });
  });
});
