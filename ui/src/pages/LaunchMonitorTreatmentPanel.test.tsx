/**
 * Tests for the Launch Monitor Analytics Data Treatment panel (#11987,
 * slice 6b).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorTreatmentPanel } from "./LaunchMonitorTreatmentPanel";
import type { TreatmentResponse } from "@/api/launchMonitorAnalytics";
import type { CsvValue } from "./LaunchMonitorAnalytics";

const COLUMNS = ["ball_speed", "club_speed", "shot_id"];
const RECORDS: Record<string, CsvValue>[] = [
  { ball_speed: 150, club_speed: 100, shot_id: "a" },
  { ball_speed: 152, club_speed: 101, shot_id: "b" },
  { ball_speed: 999, club_speed: 102, shot_id: "c" },
];

const SAMPLE_RESULT: TreatmentResponse = {
  data: [
    { ball_speed: 150, club_speed: 100, shot_id: "a" },
    { ball_speed: 152, club_speed: 101, shot_id: "b" },
  ],
  flags: [
    { row_index: 2, flag_type: "robust_outlier", metric: "ball_speed" },
    { row_index: 2, flag_type: "duplicate", metric: null },
  ],
  audit_log: [
    {
      action: "flag",
      flag_type: "robust_outlier",
      row_index: 2,
      metric: "ball_speed",
    },
  ],
  shot_count: 2,
  flag_count: 2,
};

const { applyTreatmentV2Mock } = vi.hoisted(() => ({
  applyTreatmentV2Mock: vi.fn(),
}));

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    applyTreatmentV2: applyTreatmentV2Mock,
  };
});

describe("LaunchMonitorTreatmentPanel", () => {
  beforeEach(() => {
    applyTreatmentV2Mock.mockReset();
  });

  it("renders the desktop-default threshold and exclude-flagged checkbox", () => {
    render(
      <LaunchMonitorTreatmentPanel
        columns={COLUMNS}
        records={RECORDS}
        onTreated={vi.fn()}
      />,
    );

    expect(screen.getByLabelText("Modified Z Threshold")).toHaveValue(4.5);
    expect(
      screen.getByLabelText("Exclude Flagged Rows from Analysis View"),
    ).not.toBeChecked();
  });

  it("Apply sends the default empty metric lists and filters", async () => {
    applyTreatmentV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(
      <LaunchMonitorTreatmentPanel
        columns={COLUMNS}
        records={RECORDS}
        onTreated={vi.fn()}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: /apply reproducible treatment/i }),
    );

    await waitFor(() => {
      expect(applyTreatmentV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(applyTreatmentV2Mock).toHaveBeenCalledWith(RECORDS, {
      requiredMetrics: [],
      outlierMetrics: [],
      robustZThreshold: 4.5,
      excludeFlagged: false,
      filters: [],
    });
  });

  it("drops blank comma-separated metric entries and sends the added filter row", async () => {
    applyTreatmentV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(
      <LaunchMonitorTreatmentPanel
        columns={COLUMNS}
        records={RECORDS}
        onTreated={vi.fn()}
      />,
    );

    await user.type(
      screen.getByLabelText("Required Metrics"),
      "ball_speed, , club_speed",
    );
    await user.type(
      screen.getByLabelText("Robust-Outlier Metrics"),
      "ball_speed",
    );
    fireEvent.change(screen.getByLabelText("Modified Z Threshold"), {
      target: { value: "5" },
    });
    await user.click(
      screen.getByLabelText("Exclude Flagged Rows from Analysis View"),
    );
    await user.click(screen.getByRole("button", { name: /add filter/i }));
    await user.selectOptions(
      screen.getByLabelText("Filter Column"),
      "shot_id",
    );
    await user.selectOptions(screen.getByLabelText("Filter Operator"), "eq");
    await user.type(screen.getByLabelText("Filter Value"), "a");

    await user.click(
      screen.getByRole("button", { name: /apply reproducible treatment/i }),
    );

    await waitFor(() => {
      expect(applyTreatmentV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(applyTreatmentV2Mock).toHaveBeenCalledWith(RECORDS, {
      requiredMetrics: ["ball_speed", "club_speed"],
      outlierMetrics: ["ball_speed"],
      robustZThreshold: 5,
      excludeFlagged: true,
      filters: [{ column: "shot_id", operator: "eq", value: "a" }],
    });
  });

  it("calls onTreated with the treated data and shows the flag/shot status line", async () => {
    applyTreatmentV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const onTreated = vi.fn();
    const user = userEvent.setup();
    render(
      <LaunchMonitorTreatmentPanel
        columns={COLUMNS}
        records={RECORDS}
        onTreated={onTreated}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: /apply reproducible treatment/i }),
    );

    await waitFor(() => {
      expect(onTreated).toHaveBeenCalledWith(SAMPLE_RESULT.data);
    });
    expect(await screen.findByTestId("lmt-status")).toHaveTextContent(
      "2 flags; 2 shots in the analysis view.",
    );

    const flagsTable = screen.getByTestId("lmt-flags-table");
    expect(flagsTable).toHaveTextContent("robust_outlier");
    // A flag with no associated metric renders as a dash, never blank/"0".
    expect(flagsTable).toHaveTextContent("—");
  });

  it("Reset Treatment calls onTreated(null)", async () => {
    const onTreated = vi.fn();
    const user = userEvent.setup();
    render(
      <LaunchMonitorTreatmentPanel
        columns={COLUMNS}
        records={RECORDS}
        onTreated={onTreated}
      />,
    );

    await user.click(screen.getByRole("button", { name: /reset treatment/i }));

    expect(onTreated).toHaveBeenCalledWith(null);
  });

  it("surfaces the API error message on failure", async () => {
    applyTreatmentV2Mock.mockRejectedValue(
      new Error("Outlier metric not present: missing_metric"),
    );
    const user = userEvent.setup();
    render(
      <LaunchMonitorTreatmentPanel
        columns={COLUMNS}
        records={RECORDS}
        onTreated={vi.fn()}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: /apply reproducible treatment/i }),
    );

    expect(
      await screen.findByText(
        /Data treatment could not run: Outlier metric not present/,
      ),
    ).toBeInTheDocument();
  });
});
