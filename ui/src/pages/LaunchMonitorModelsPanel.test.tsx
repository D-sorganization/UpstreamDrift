/**
 * Tests for the Launch Monitor Analytics Models panel (#11987, slice 5b).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorModelsPanel } from "./LaunchMonitorModelsPanel";
import type { ModelResponse } from "@/api/launchMonitorAnalytics";
import type { CsvValue } from "./LaunchMonitorAnalytics";

const COLUMNS = ["ball_speed", "club_speed", "spin_rate", "session_id"];
const RECORDS: Record<string, CsvValue>[] = Array.from(
  { length: 12 },
  (_, index) => ({
    ball_speed: 145 + index,
    club_speed: 95 + index,
    spin_rate: 2400 + index * 10,
    session_id: index % 2 === 0 ? "s1" : "s2",
  }),
);

const SAMPLE_RESULT: ModelResponse = {
  model: "linear",
  target: "ball_speed",
  features: ["club_speed"],
  metrics: { r2: 0.91, mae: 1.1, rmse: 1.4 },
  coefficients: { club_speed: 1.5 },
  random_seed: 42,
  train_count: 9,
  test_count: 3,
  predictions: [
    { row_index: 0, actual: 145, predicted: 144.5, residual: 0.5 },
    { row_index: 1, actual: 146, predicted: null, residual: null },
  ],
};

const { fitModelV2Mock } = vi.hoisted(() => ({
  fitModelV2Mock: vi.fn(),
}));

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    fitModelV2: fitModelV2Mock,
  };
});

describe("LaunchMonitorModelsPanel", () => {
  beforeEach(() => {
    fitModelV2Mock.mockReset();
  });

  it("renders desktop-default model, seed, and grouped-holdout choices", () => {
    render(<LaunchMonitorModelsPanel columns={COLUMNS} records={RECORDS} />);

    expect(screen.getByLabelText("Model")).toHaveValue("linear");
    expect(screen.getByLabelText("Random Seed")).toHaveValue(42);
    expect(screen.getByLabelText("Grouped Holdout")).toHaveValue(
      "(random split)",
    );
  });

  it("disables Run until a target and at least one feature are selected", async () => {
    const user = userEvent.setup();
    render(<LaunchMonitorModelsPanel columns={COLUMNS} records={RECORDS} />);

    const runButton = screen.getByRole("button", {
      name: /fit and validate model/i,
    });
    expect(runButton).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Target"), "ball_speed");
    expect(runButton).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Features"), [
      "club_speed",
    ]);
    expect(runButton).toBeEnabled();
  });

  it("sends the desktop defaults when no grouped holdout is chosen", async () => {
    fitModelV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(<LaunchMonitorModelsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Target"), "ball_speed");
    await user.selectOptions(screen.getByLabelText("Features"), [
      "club_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /fit and validate model/i }),
    );

    await waitFor(() => {
      expect(fitModelV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(fitModelV2Mock).toHaveBeenCalledWith(
      RECORDS,
      "ball_speed",
      ["club_speed"],
      "linear",
      42,
      null,
    );
  });

  it("sends the selected model, seed, and grouped holdout", async () => {
    fitModelV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(<LaunchMonitorModelsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Target"), "ball_speed");
    await user.selectOptions(screen.getByLabelText("Features"), [
      "club_speed",
      "spin_rate",
    ]);
    await user.selectOptions(screen.getByLabelText("Model"), "ridge");
    await user.selectOptions(
      screen.getByLabelText("Grouped Holdout"),
      "session_id",
    );
    const seedInput = screen.getByLabelText("Random Seed");
    fireEvent.change(seedInput, { target: { value: "7" } });

    await user.click(
      screen.getByRole("button", { name: /fit and validate model/i }),
    );

    await waitFor(() => {
      expect(fitModelV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(fitModelV2Mock).toHaveBeenCalledWith(
      RECORDS,
      "ball_speed",
      ["club_speed", "spin_rate"],
      "ridge",
      7,
      "session_id",
    );
  });

  it("renders metrics, coefficients, and predictions with null as a dash", async () => {
    fitModelV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    render(<LaunchMonitorModelsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Target"), "ball_speed");
    await user.selectOptions(screen.getByLabelText("Features"), [
      "club_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /fit and validate model/i }),
    );

    const metrics = await screen.findByTestId("lmm-metrics-table");
    expect(metrics).toHaveTextContent("0.9100");

    const coefficients = await screen.findByTestId("lmm-coefficients-table");
    expect(coefficients).toHaveTextContent("club_speed");

    const predictions = await screen.findByTestId("lmm-predictions-table");
    expect(predictions).toHaveTextContent("—");

    expect(screen.getByTestId("lmm-split-counts")).toHaveTextContent(
      "Train n = 9",
    );
  });

  it("shows a note instead of a table when coefficients are unavailable (e.g. mlp)", async () => {
    fitModelV2Mock.mockResolvedValue({ ...SAMPLE_RESULT, coefficients: null });
    const user = userEvent.setup();
    render(<LaunchMonitorModelsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Target"), "ball_speed");
    await user.selectOptions(screen.getByLabelText("Features"), [
      "club_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /fit and validate model/i }),
    );

    expect(await screen.findByTestId("lmm-no-coefficients")).toHaveTextContent(
      "not available",
    );
  });

  it("surfaces a 503 'unavailable' message distinctly from a generic failure", async () => {
    fitModelV2Mock.mockRejectedValue(
      new Error("Model 'mlp' is unavailable: no module named 'sklearn'"),
    );
    const user = userEvent.setup();
    render(<LaunchMonitorModelsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Target"), "ball_speed");
    await user.selectOptions(screen.getByLabelText("Features"), [
      "club_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /fit and validate model/i }),
    );

    const status = await screen.findByTestId("lmm-status");
    await waitFor(() => {
      expect(status).toHaveTextContent(
        "Model unavailable: Model 'mlp' is unavailable: no module named 'sklearn'",
      );
    });
  });

  it("surfaces a generic failure message", async () => {
    fitModelV2Mock.mockRejectedValue(
      new Error("Insufficient complete rows for predictive modeling"),
    );
    const user = userEvent.setup();
    render(<LaunchMonitorModelsPanel columns={COLUMNS} records={RECORDS} />);

    await user.selectOptions(screen.getByLabelText("Target"), "ball_speed");
    await user.selectOptions(screen.getByLabelText("Features"), [
      "club_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /fit and validate model/i }),
    );

    expect(
      await screen.findByText(
        /Predictive model could not run: Insufficient complete rows/,
      ),
    ).toBeInTheDocument();
  });
});
