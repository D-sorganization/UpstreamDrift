/**
 * Tests for the Launch Monitor Analytics Dispersion panel (#11987, slice 3b).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorDispersionPanel } from "./LaunchMonitorDispersionPanel";
import type { DispersionResponse } from "@/api/launchMonitorAnalytics";
import type { CsvValue } from "./LaunchMonitorAnalytics";

const COLUMNS = ["captured_at", "carry_distance", "lateral_carry", "club", "notes"];
const RECORDS: Record<string, CsvValue>[] = [
  { captured_at: "2024-01-01T00:00:00Z", carry_distance: 220, lateral_carry: 5, club: "driver", notes: "a" },
  { captured_at: "2024-01-01T00:01:00Z", carry_distance: 215, lateral_carry: -3, club: "driver", notes: "b" },
  { captured_at: "2024-01-01T00:02:00Z", carry_distance: 150, lateral_carry: 2, club: "7i", notes: "c" },
  { captured_at: "2024-01-01T00:03:00Z", carry_distance: 148, lateral_carry: -1, club: "7i", notes: "d" },
  { captured_at: "2024-01-01T00:04:00Z", carry_distance: 152, lateral_carry: 1, club: "7i", notes: "e" },
];

const SAMPLE_RESULT: DispersionResponse = {
  forward: "carry_distance",
  lateral: "lateral_carry",
  group_column: "club",
  groups: [
    {
      group: "driver",
      sample_count: 2,
      center_forward: 217.5,
      center_lateral: 1.0,
      mean_forward: 217.5,
      mean_lateral: 1.0,
      ellipse_major: 10.0,
      ellipse_minor: 4.0,
      ellipse_angle_rad: 0.2,
      area_95: 31.4,
      radial_rmse: 3.6,
      radial_p50: 3.5,
      radial_p90: 3.6,
    },
    {
      group: "7i",
      sample_count: 3,
      center_forward: null,
      center_lateral: null,
      mean_forward: null,
      mean_lateral: null,
      ellipse_major: null,
      ellipse_minor: null,
      ellipse_angle_rad: null,
      area_95: null,
      radial_rmse: null,
      radial_p50: null,
      radial_p90: null,
    },
  ],
};

const { analyzeDispersionV2Mock } = vi.hoisted(() => ({
  analyzeDispersionV2Mock: vi.fn(),
}));

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    analyzeDispersionV2: analyzeDispersionV2Mock,
  };
});

describe("LaunchMonitorDispersionPanel", () => {
  beforeEach(() => {
    analyzeDispersionV2Mock.mockReset();
  });

  it("falls back to the first numeric column and (all shots) when chosen columns leave the CSV", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <LaunchMonitorDispersionPanel columns={COLUMNS} records={RECORDS} />,
    );
    await user.selectOptions(screen.getByLabelText("Group By"), "club");

    const otherColumns = ["ball_speed", "spin_rate"];
    const otherRecords: Record<string, CsvValue>[] = RECORDS.map((_, i) => ({
      ball_speed: 150 + i,
      spin_rate: 2500 + i,
    }));
    rerender(
      <LaunchMonitorDispersionPanel
        columns={otherColumns}
        records={otherRecords}
      />,
    );

    expect(screen.getByLabelText("Forward Coordinate")).toHaveValue(
      "ball_speed",
    );
    expect(screen.getByLabelText("Lateral Coordinate")).toHaveValue(
      "ball_speed",
    );
    expect(screen.getByLabelText("Group By")).toHaveValue("(all shots)");
  });

  it(
    "sends the desktop-default inputs, omitting group_column for " +
      '"(all shots)", and includes it once a group is selected',
    async () => {
      analyzeDispersionV2Mock.mockResolvedValue(SAMPLE_RESULT);
      const user = userEvent.setup();
      render(
        <LaunchMonitorDispersionPanel columns={COLUMNS} records={RECORDS} />,
      );

      // `carry_distance`/`lateral_carry` are auto-selected (desktop defaults)
      // and "(all shots)" is the Group By default, so Run is enabled already.
      expect(screen.getByLabelText("Forward Coordinate")).toHaveValue(
        "carry_distance",
      );
      expect(screen.getByLabelText("Lateral Coordinate")).toHaveValue(
        "lateral_carry",
      );
      const runButton = screen.getByRole("button", {
        name: /analyze dispersion/i,
      });
      expect(runButton).toBeEnabled();

      await user.click(runButton);
      await waitFor(() => {
        expect(analyzeDispersionV2Mock).toHaveBeenCalledTimes(1);
      });
      expect(analyzeDispersionV2Mock).toHaveBeenCalledWith(
        RECORDS,
        "carry_distance",
        "lateral_carry",
        null,
      );

      await user.selectOptions(screen.getByLabelText("Group By"), "club");
      await user.click(runButton);
      await waitFor(() => {
        expect(analyzeDispersionV2Mock).toHaveBeenCalledTimes(2);
      });
      expect(analyzeDispersionV2Mock).toHaveBeenLastCalledWith(
        RECORDS,
        "carry_distance",
        "lateral_carry",
        "club",
      );
    },
  );

  it("renders one results-table row per group, with null fields as unavailable", async () => {
    analyzeDispersionV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(
      <LaunchMonitorDispersionPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.click(
      screen.getByRole("button", { name: /analyze dispersion/i }),
    );

    const table = await screen.findByTestId("lmd-results-table");
    const rows = table.querySelectorAll("tbody tr");
    expect(rows).toHaveLength(2);

    expect(rows[0]).toHaveTextContent("driver");
    expect(rows[0]).toHaveTextContent("10.0000"); // ellipse_major

    // The "7i" group's fields are all null and must render as "unavailable",
    // never a misleading "0".
    expect(rows[1]).toHaveTextContent("7i");
    expect(rows[1]).not.toHaveTextContent("0.0000");
    const nullCells = rows[1].querySelectorAll("td");
    // group name + sample_count are not null; every statistic column is.
    for (let index = 2; index < nullCells.length; index += 1) {
      expect(nullCells[index]).toHaveTextContent("unavailable");
    }
  });

  it("renders an ellipse SVG for a group with finite axes, and not for null axes", async () => {
    analyzeDispersionV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(
      <LaunchMonitorDispersionPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.click(
      screen.getByRole("button", { name: /analyze dispersion/i }),
    );

    expect(await screen.findByTestId("lmd-ellipse-svg-driver")).toBeInTheDocument();
    expect(screen.queryByTestId("lmd-ellipse-svg-7i")).not.toBeInTheDocument();
    expect(
      screen.getByTestId("lmd-ellipse-unavailable-7i"),
    ).toBeInTheDocument();
  });

  it("surfaces the API error message on failure", async () => {
    analyzeDispersionV2Mock.mockRejectedValue(
      new Error("Dispersion columns not present: ['x']"),
    );
    const user = userEvent.setup();
    render(
      <LaunchMonitorDispersionPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.click(
      screen.getByRole("button", { name: /analyze dispersion/i }),
    );

    expect(
      await screen.findByText(
        /Dispersion analysis could not run: Dispersion columns not present/,
      ),
    ).toBeInTheDocument();
  });
});
