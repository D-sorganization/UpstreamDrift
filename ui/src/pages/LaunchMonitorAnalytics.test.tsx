/**
 * Tests for the Launch Monitor Analytics — Flexible Analysis page (#11987).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorAnalyticsPage } from "./LaunchMonitorAnalytics";
import type { LaunchMonitorAnalysisResultV2 } from "@/api/generated/types";

const CSV_TEXT = [
  "ball_speed,club_speed,notes",
  "100,80,a",
  "105,82,b",
  "110,85,c",
  "115,88,d",
].join("\n");

function csvFile(text: string, name = "shots.csv"): File {
  return new File([text], name, { type: "text/csv" });
}

const SAMPLE_RESULT: LaunchMonitorAnalysisResultV2 = {
  contract_version: "2.0.0",
  status: "available",
  analysis: {
    dataset: {
      row_count: 4,
      complete_row_count: 4,
      selected_columns: ["club_speed", "ball_speed"],
      monitor_vendors: [],
      session_ids: [],
      observation_kinds: ["shot"],
      fingerprint_sha256: "a".repeat(64),
    },
    correlations: [
      {
        predictor: "ball_speed",
        coefficient: 0.98,
        p_value: 0.001,
        adjusted_p_value: 0.001,
        ci_lower: 0.5,
        ci_upper: 0.99,
        sample_count: 4,
        method: "pearson",
        is_boolean_projected: false,
      },
    ],
    regression: {
      sample_count: 4,
      r_squared: 0.96,
      adjusted_r_squared: 0.94,
      coefficients: {
        intercept: {
          estimate: 1.2,
          standard_error: 0.3,
          t_statistic: 4.0,
          p_value: 0.02,
          ci_lower: 0.1,
          ci_upper: 2.3,
        },
        ball_speed: {
          estimate: 0.8,
          standard_error: 0.1,
          t_statistic: 8.0,
          p_value: 0.001,
          ci_lower: 0.6,
          ci_upper: 1.0,
        },
      },
      residual_diagnostics: {
        rmse: 0.5,
        mae: 0.4,
        residual_mean: 0.0,
        residual_std: 0.5,
        durbin_watson: 2.0,
        jarque_bera_p_value: 0.5,
        influential_count: 0,
      },
    },
    groups: [],
    units: {},
    warnings: [],
  },
  units: {},
  lineage: {
    dataset_fingerprint_sha256: "b".repeat(64),
    transformations: [],
    sources: [],
    backing_records: [],
  },
  missingness: {
    input_row_count: 4,
    complete_row_count: 4,
    missing_by_variable: {},
    non_numeric_by_variable: {},
    excluded_by_reason: {},
    policy: "pairwise",
  },
  availability: [],
  uncertainty: {
    confidence_level: 0.95,
    correlation_interval: "fisher-z",
    regression_interval: "student-t",
    multiplicity_adjustment: "benjamini-hochberg",
    assumptions: [],
  },
  player_identity: { trust_level: "not_provided" },
  vendor_provenance: [],
  model_provenance: [],
  warnings: [],
};

const {
  runFlexibleAnalysisV2Mock,
  fetchCapabilitiesMock,
  analyzeRelationshipsV2Mock,
  analyzeMultivariateV2Mock,
  applyTreatmentV2Mock,
} = vi.hoisted(() => ({
  runFlexibleAnalysisV2Mock: vi.fn(),
  fetchCapabilitiesMock: vi.fn(async () => ({
    analysis_modes: ["comprehensive", "correlation", "regression"],
    correlation_methods: ["pearson", "spearman", "kendall"],
    missing_policies: ["pairwise", "listwise", "fail"],
    maximum_inline_records: 20_000,
  })),
  analyzeRelationshipsV2Mock: vi.fn(),
  analyzeMultivariateV2Mock: vi.fn(),
  applyTreatmentV2Mock: vi.fn(),
}));

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    fetchLaunchMonitorAnalyticsCapabilities: fetchCapabilitiesMock,
    runFlexibleAnalysisV2: runFlexibleAnalysisV2Mock,
    analyzeRelationshipsV2: analyzeRelationshipsV2Mock,
    analyzeMultivariateV2: analyzeMultivariateV2Mock,
    applyTreatmentV2: applyTreatmentV2Mock,
  };
});

async function loadCsv() {
  render(<LaunchMonitorAnalyticsPage />);
  const input = screen.getByTestId("csv-file-input");
  fireEvent.change(input, { target: { files: [csvFile(CSV_TEXT)] } });
  await waitFor(() => {
    expect(screen.getByTestId("lma-row-col-count")).toHaveTextContent(
      /4 row\(s\)/,
    );
  });
}

describe("LaunchMonitorAnalyticsPage", () => {
  beforeEach(() => {
    runFlexibleAnalysisV2Mock.mockReset();
    fetchCapabilitiesMock.mockClear();
    analyzeRelationshipsV2Mock.mockReset();
    analyzeMultivariateV2Mock.mockReset();
    applyTreatmentV2Mock.mockReset();
  });

  it("parses a loaded CSV and shows row/column counts and numeric options", async () => {
    await loadCsv();

    expect(screen.getByTestId("lma-row-col-count")).toHaveTextContent(
      /3 column\(s\)/,
    );

    const outcomeSelect = screen.getByLabelText("Outcome");
    const options = Array.from(outcomeSelect.querySelectorAll("option")).map(
      (option) => option.textContent,
    );
    expect(options).toContain("ball_speed");
    expect(options).toContain("club_speed");
    // The free-text "notes" column never parses as numeric (#11987).
    expect(options).not.toContain("notes");
  });

  it("disables Run until an outcome and a predictor are selected", async () => {
    const user = userEvent.setup();
    await loadCsv();

    const runButton = screen.getByRole("button", {
      name: /run flexible analysis/i,
    });
    expect(runButton).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Outcome"), "club_speed");
    expect(runButton).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Predictors"), [
      "ball_speed",
    ]);
    expect(runButton).toBeEnabled();
  });

  it("runs the analysis and renders correlation and regression tables", async () => {
    runFlexibleAnalysisV2Mock.mockResolvedValue(SAMPLE_RESULT);
    const user = userEvent.setup();
    await loadCsv();

    await user.selectOptions(screen.getByLabelText("Outcome"), "club_speed");
    await user.selectOptions(screen.getByLabelText("Predictors"), [
      "ball_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /run flexible analysis/i }),
    );

    await waitFor(() => {
      expect(runFlexibleAnalysisV2Mock).toHaveBeenCalledTimes(1);
    });
    const [records, analysis] = runFlexibleAnalysisV2Mock.mock.calls[0];
    expect(records).toHaveLength(4);
    expect(analysis).toMatchObject({
      outcome: "club_speed",
      predictors: ["ball_speed"],
    });

    expect(
      await screen.findByText(/Analysis complete for 4 observation\(s\)\./),
    ).toBeInTheDocument();
    const correlationsSection = screen.getByTestId("lma-correlations-table");
    expect(correlationsSection).toHaveTextContent("ball_speed");
    const regressionSection = screen.getByTestId("lma-regression-table");
    expect(regressionSection).toHaveTextContent("intercept");
  });

  it("surfaces the API error message on failure", async () => {
    runFlexibleAnalysisV2Mock.mockRejectedValue(
      new Error("Columns not present: ['missing_column']"),
    );
    const user = userEvent.setup();
    await loadCsv();

    await user.selectOptions(screen.getByLabelText("Outcome"), "club_speed");
    await user.selectOptions(screen.getByLabelText("Predictors"), [
      "ball_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /run flexible analysis/i }),
    );

    expect(
      await screen.findByText(
        /Analysis could not run: Columns not present/,
      ),
    ).toBeInTheDocument();
  });

  it("feeds a successful treatment into every analysis panel as the treated view", async () => {
    applyTreatmentV2Mock.mockResolvedValue({
      data: [
        { ball_speed: 100, club_speed: 80, notes: "a" },
        { ball_speed: 105, club_speed: 82, notes: "b" },
      ],
      flags: [{ row_index: 2, flag_type: "robust_outlier", metric: "ball_speed" }],
      audit_log: [],
      shot_count: 2,
      flag_count: 1,
    });
    const user = userEvent.setup();
    await loadCsv();

    expect(screen.queryByTestId("lma-treated-indicator")).not.toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: /apply reproducible treatment/i }),
    );

    // The page passes the raw (not yet treated) records to the Treatment
    // panel itself, mirroring the desktop's `project.combined_shots()`.
    await waitFor(() => {
      expect(applyTreatmentV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(applyTreatmentV2Mock.mock.calls[0][0]).toHaveLength(4);

    // The treated view (2 of the original 4 rows) now drives the page's
    // indicator and every analysis panel's `analysisRecords`/`analysisColumns`.
    expect(await screen.findByTestId("lma-treated-indicator")).toHaveTextContent(
      "Treated view: 2 of 4 shots",
    );

    await user.click(screen.getByRole("button", { name: /reset treatment/i }));
    expect(
      screen.queryByTestId("lma-treated-indicator"),
    ).not.toBeInTheDocument();
  });
});
