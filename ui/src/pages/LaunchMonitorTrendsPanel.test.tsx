/**
 * Tests for the Launch Monitor Analytics Trends panel (#11987, slice 2b).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorTrendsPanel } from "./LaunchMonitorTrendsPanel";
import type { TrendResponse } from "@/api/launchMonitorAnalytics";
import type { CsvValue } from "./LaunchMonitorAnalytics";

const COLUMNS = ["captured_at", "club_speed", "notes"];
const RECORDS: Record<string, CsvValue>[] = [
  { captured_at: "2024-01-01T00:00:00Z", club_speed: 90, notes: "a" },
  { captured_at: "2024-01-02T00:00:00Z", club_speed: 91, notes: "b" },
  { captured_at: "2024-01-03T00:00:00Z", club_speed: 92, notes: "c" },
  { captured_at: "2024-01-04T00:00:00Z", club_speed: 93, notes: "d" },
];

const SAMPLE_RESULT: TrendResponse = {
  metric: "club_speed",
  sample_count: 4,
  slope_per_day: 1.0,
  robust_slope_per_day: 1.0,
  p_value: 0.01,
  earliest_mean: 90.5,
  latest_mean: 92.5,
  rolling: [
    {
      captured_at: "2024-01-01T00:00:00Z",
      value: 90,
      rolling_mean: null,
      rolling_median: null,
      rolling_std: null,
      ewma: 90,
    },
    {
      captured_at: "2024-01-04T00:00:00Z",
      value: 93,
      rolling_mean: 91.5,
      rolling_median: 91.5,
      rolling_std: 1.29,
      ewma: 92.1,
    },
  ],
  change_candidates: [
    {
      captured_at: "2024-01-03T00:00:00Z",
      row_index: 2,
      before_mean: 90.5,
      after_mean: 92.5,
      effect_size: 1.5,
    },
  ],
};

const { postTrendMock } = vi.hoisted(() => ({
  postTrendMock: vi.fn(),
}));

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    postTrend: postTrendMock,
  };
});

describe("LaunchMonitorTrendsPanel", () => {
  beforeEach(() => {
    postTrendMock.mockReset();
  });

  it("disables Run until a metric and time column are selected", async () => {
    const user = userEvent.setup();
    render(<LaunchMonitorTrendsPanel columns={COLUMNS} records={RECORDS} />);

    const runButton = screen.getByRole("button", {
      name: /analyze longitudinal change/i,
    });
    // `captured_at` is the only time-column candidate and is auto-selected,
    // so only the metric remains unselected.
    expect(runButton).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Metric"), "club_speed");
    expect(runButton).toBeEnabled();
  });

  it("disables Run while the rolling window is outside [3, 500]", async () => {
    const user = userEvent.setup();
    render(<LaunchMonitorTrendsPanel columns={COLUMNS} records={RECORDS} />);
    await user.selectOptions(screen.getByLabelText("Metric"), "club_speed");
    const runButton = screen.getByRole("button", {
      name: /analyze longitudinal change/i,
    });
    const windowInput = screen.getByLabelText("Rolling Window");

    await user.clear(windowInput);
    await user.type(windowInput, "2");
    expect(runButton).toBeDisabled();

    await user.clear(windowInput);
    await user.type(windowInput, "100");
    expect(windowInput).toHaveValue(100);
    expect(runButton).toBeEnabled();

    await user.clear(windowInput);
    await user.type(windowInput, "501");
    expect(runButton).toBeDisabled();
  });

  it("sends the expected payload (rolling_window default 10) and renders results", async () => {
    postTrendMock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(<LaunchMonitorTrendsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Metric"), "club_speed");
    await user.click(
      screen.getByRole("button", { name: /analyze longitudinal change/i }),
    );

    await waitFor(() => {
      expect(postTrendMock).toHaveBeenCalledTimes(1);
    });
    expect(postTrendMock).toHaveBeenCalledWith(
      RECORDS,
      "club_speed",
      "captured_at",
      10,
    );

    expect(await screen.findByTestId("lmt-summary")).toHaveTextContent(
      "club_speed",
    );
    const summary = screen.getByTestId("lmt-summary");
    expect(summary).toHaveTextContent("4"); // sample_count
    expect(summary).toHaveTextContent("1.0000"); // slope_per_day via formatStat
  });

  it("renders a null response value as — never 0", async () => {
    postTrendMock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(<LaunchMonitorTrendsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Metric"), "club_speed");
    await user.click(
      screen.getByRole("button", { name: /analyze longitudinal change/i }),
    );

    const rollingTable = await screen.findByTestId("lmt-rolling-table");
    // The first rolling row's rolling_mean/rolling_std are null (pre-window)
    // and must render as "—", never a misleading "0".
    const cells = rollingTable.querySelectorAll("tbody tr")[0].querySelectorAll("td");
    expect(cells[2]).toHaveTextContent("—"); // rolling_mean
    expect(cells[4]).toHaveTextContent("—"); // rolling_std
  });

  it("surfaces the API error message on failure", async () => {
    postTrendMock.mockRejectedValue(new Error("Trend columns not present: ['x']"));
    const user = userEvent.setup();
    render(<LaunchMonitorTrendsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Metric"), "club_speed");
    await user.click(
      screen.getByRole("button", { name: /analyze longitudinal change/i }),
    );

    expect(
      await screen.findByText(
        /Trend analysis could not run: Trend columns not present/,
      ),
    ).toBeInTheDocument();
  });
});
