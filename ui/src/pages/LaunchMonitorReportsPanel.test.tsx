/**
 * Tests for the Launch Monitor Analytics Reports panel (#11987, slice 7b).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorReportsPanel } from "./LaunchMonitorReportsPanel";
import type {
  ExportReportResponse,
  ReportResponse,
} from "@/api/launchMonitorAnalytics";
import type { CsvValue } from "./LaunchMonitorAnalytics";

const RECORDS: Record<string, CsvValue>[] = Array.from(
  { length: 5 },
  (_, index) => ({
    ball_speed: 145 + index,
    club_speed: 95 + index,
    session_id: index % 2 === 0 ? "s1" : "s2",
  }),
);

const SAMPLE_REPORT: ReportResponse = {
  report_text: "Launch Monitor Study Report\n============\n5 shots analyzed.",
  project_name: "Untitled Launch Monitor Study",
  session_count: 2,
  shot_count: 5,
  canonical_metrics: ["ball_speed", "club_speed"],
  treatment_action_count: 0,
};

const SAMPLE_EXPORT: ExportReportResponse = {
  csv: "ball_speed,club_speed,session_id\n145,95,s1\n146,96,s2\n",
  data_export: { export_id: "abc123", exported_at: "2026-01-01T00:00:00Z" },
  manifest: { project_name: "Untitled Launch Monitor Study", sessions: [] },
};

const { buildReportV2Mock, exportReportV2Mock } = vi.hoisted(() => ({
  buildReportV2Mock: vi.fn(),
  exportReportV2Mock: vi.fn(),
}));

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    buildReportV2: buildReportV2Mock,
    exportReportV2: exportReportV2Mock,
  };
});

describe("LaunchMonitorReportsPanel", () => {
  beforeEach(() => {
    buildReportV2Mock.mockReset();
    exportReportV2Mock.mockReset();
  });

  it("disables both buttons when there are no records", () => {
    render(<LaunchMonitorReportsPanel records={[]} />);

    expect(screen.getByTestId("lmr-generate-btn")).toBeDisabled();
    expect(screen.getByTestId("lmr-export-btn")).toBeDisabled();
  });

  it("enables both buttons once records are loaded", () => {
    render(<LaunchMonitorReportsPanel records={RECORDS} />);

    expect(screen.getByTestId("lmr-generate-btn")).toBeEnabled();
    expect(screen.getByTestId("lmr-export-btn")).toBeEnabled();
  });

  it("calls buildReportV2 with the loaded records and project name, and displays the report text", async () => {
    buildReportV2Mock.mockResolvedValue(SAMPLE_REPORT);
    const user = userEvent.setup();
    render(<LaunchMonitorReportsPanel records={RECORDS} />);

    const projectNameInput = screen.getByTestId("lmr-project-name");
    await user.clear(projectNameInput);
    await user.type(projectNameInput, "My Study");

    await user.click(screen.getByTestId("lmr-generate-btn"));

    await waitFor(() => {
      expect(buildReportV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(buildReportV2Mock).toHaveBeenCalledWith(RECORDS, "My Study", []);

    expect(await screen.findByTestId("lmr-report-text")).toHaveTextContent(
      "Launch Monitor Study Report",
    );
  });

  it("calls exportReportV2 and displays both the CSV and manifest", async () => {
    exportReportV2Mock.mockResolvedValue(SAMPLE_EXPORT);
    const user = userEvent.setup();
    render(<LaunchMonitorReportsPanel records={RECORDS} />);

    await user.click(screen.getByTestId("lmr-export-btn"));

    await waitFor(() => {
      expect(exportReportV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(exportReportV2Mock).toHaveBeenCalledWith(
      RECORDS,
      "Untitled Launch Monitor Study",
      [],
    );

    expect(await screen.findByTestId("lmr-export-csv")).toHaveTextContent(
      "ball_speed,club_speed,session_id",
    );
    expect(screen.getByTestId("lmr-export-manifest")).toHaveTextContent(
      "Untitled Launch Monitor Study",
    );
  });

  it("renders API errors cleanly in the status message", async () => {
    buildReportV2Mock.mockRejectedValue(new Error("Report generation failed"));
    const user = userEvent.setup();
    render(<LaunchMonitorReportsPanel records={RECORDS} />);

    await user.click(screen.getByTestId("lmr-generate-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("lmr-status")).toHaveTextContent(
        "Report could not be generated: Report generation failed",
      );
    });
  });

  it("renders export API errors cleanly in the status message", async () => {
    exportReportV2Mock.mockRejectedValue(new Error("Export failed badly"));
    const user = userEvent.setup();
    render(<LaunchMonitorReportsPanel records={RECORDS} />);

    await user.click(screen.getByTestId("lmr-export-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("lmr-status")).toHaveTextContent(
        "Export could not run: Export failed badly",
      );
    });
  });
});
