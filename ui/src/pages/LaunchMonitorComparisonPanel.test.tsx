/**
 * Tests for the Launch Monitor Analytics Monitor Comparison panel (#11987,
 * slice 5b).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorComparisonPanel } from "./LaunchMonitorComparisonPanel";
import type { ComparisonResponse } from "@/api/launchMonitorAnalytics";
import type { CsvValue } from "./LaunchMonitorAnalytics";

const COLUMNS = ["captured_at", "ball_speed", "monitor_vendor", "shot_id"];
const RECORDS: Record<string, CsvValue>[] = [
  { captured_at: "t1", ball_speed: 150, monitor_vendor: "trackman", shot_id: "a" },
  { captured_at: "t2", ball_speed: 152, monitor_vendor: "trackman", shot_id: "b" },
  { captured_at: "t3", ball_speed: 148, monitor_vendor: "gcquad", shot_id: "a" },
  { captured_at: "t4", ball_speed: 155, monitor_vendor: "gcquad", shot_id: "b" },
  { captured_at: "t5", ball_speed: 151, monitor_vendor: "gcquad", shot_id: "c" },
];

const SAMPLE_RESULT: ComparisonResponse = {
  metric: "ball_speed",
  match_column: null,
  reference_monitor: null,
  summaries: [
    {
      monitor: "gcquad",
      sample_count: 3,
      mean: 151.3,
      standard_deviation: 3.5,
      median: 151,
    },
    {
      monitor: "trackman",
      sample_count: 2,
      mean: 151,
      standard_deviation: 1.4,
      median: 151,
    },
  ],
  pairwise: [
    {
      reference: "gcquad",
      comparator: "trackman",
      matched: false,
      sample_count: 2,
      mean_bias: -0.3,
      standard_deviation_bias: null,
      lower_limit: null,
      upper_limit: null,
      slope: null,
      intercept: null,
      correlation: null,
      warning:
        "Unmatched comparison is descriptive and may be confounded by player, club, environment, and session composition.",
    },
  ],
};

const { compareMonitorsV2Mock } = vi.hoisted(() => ({
  compareMonitorsV2Mock: vi.fn(),
}));

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    compareMonitorsV2: compareMonitorsV2Mock,
  };
});

describe("LaunchMonitorComparisonPanel", () => {
  beforeEach(() => {
    compareMonitorsV2Mock.mockReset();
  });

  it("offers the dataset's numeric columns as Metric options and defaults to unmatched/blank", () => {
    render(
      <LaunchMonitorComparisonPanel columns={COLUMNS} records={RECORDS} />,
    );

    const metricSelect = screen.getByLabelText("Metric");
    const metricOptions = Array.from(
      metricSelect.querySelectorAll("option"),
    ).map((o) => o.textContent);
    expect(metricOptions).toContain("ball_speed");

    expect(screen.getByLabelText("Matched-Shot Column")).toHaveValue(
      "(unmatched)",
    );
    expect(screen.getByLabelText("Reference Monitor")).toHaveValue(
      "(default: first monitor)",
    );
  });

  it("disables Run until a metric is selected", async () => {
    const user = userEvent.setup();
    render(
      <LaunchMonitorComparisonPanel columns={COLUMNS} records={RECORDS} />,
    );

    const runButton = screen.getByRole("button", {
      name: /compare monitor behavior/i,
    });
    expect(runButton).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Metric"), "ball_speed");
    expect(runButton).toBeEnabled();
  });

  it("sends null for unmatched/default-reference by default", async () => {
    compareMonitorsV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(
      <LaunchMonitorComparisonPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Metric"), "ball_speed");
    await user.click(
      screen.getByRole("button", { name: /compare monitor behavior/i }),
    );

    await waitFor(() => {
      expect(compareMonitorsV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(compareMonitorsV2Mock).toHaveBeenCalledWith(
      RECORDS,
      "ball_speed",
      null,
      null,
    );
  });

  it("sends the selected matched-shot column and reference monitor", async () => {
    compareMonitorsV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(
      <LaunchMonitorComparisonPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Metric"), "ball_speed");
    await user.selectOptions(
      screen.getByLabelText("Matched-Shot Column"),
      "shot_id",
    );
    await user.selectOptions(
      screen.getByLabelText("Reference Monitor"),
      "gcquad",
    );
    await user.click(
      screen.getByRole("button", { name: /compare monitor behavior/i }),
    );

    await waitFor(() => {
      expect(compareMonitorsV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(compareMonitorsV2Mock).toHaveBeenCalledWith(
      RECORDS,
      "ball_speed",
      "shot_id",
      "gcquad",
    );
  });

  it("renders summaries and pairwise tables with null as a dash", async () => {
    compareMonitorsV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(
      <LaunchMonitorComparisonPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Metric"), "ball_speed");
    await user.click(
      screen.getByRole("button", { name: /compare monitor behavior/i }),
    );

    const summaries = await screen.findByTestId("lmc-summaries-table");
    expect(summaries).toHaveTextContent("gcquad");
    expect(summaries).toHaveTextContent("151.3000");

    const pairwise = await screen.findByTestId("lmc-pairwise-table");
    expect(pairwise).toHaveTextContent("—");
    expect(pairwise).toHaveTextContent(
      "Unmatched comparison is descriptive",
    );
  });

  it("surfaces the API error message on failure", async () => {
    compareMonitorsV2Mock.mockRejectedValue(
      new Error("At least two monitors are required"),
    );
    const user = userEvent.setup();
    render(
      <LaunchMonitorComparisonPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Metric"), "ball_speed");
    await user.click(
      screen.getByRole("button", { name: /compare monitor behavior/i }),
    );

    expect(
      await screen.findByText(
        /Monitor comparison could not run: At least two monitors are required/,
      ),
    ).toBeInTheDocument();
  });
});
