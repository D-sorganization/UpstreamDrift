/**
 * Tests for the Launch Monitor Analytics Relationships panel (#11987, slice 4b).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LaunchMonitorRelationshipsPanel } from "./LaunchMonitorRelationshipsPanel";
import type {
  MultivariateResponse,
  RelationshipsResponse,
} from "@/api/launchMonitorAnalytics";
import type { CsvValue } from "./LaunchMonitorAnalytics";

const COLUMNS = [
  "captured_at",
  "ball_speed",
  "club_speed",
  "spin_rate",
  "club",
  "notes",
];
const RECORDS: Record<string, CsvValue>[] = [
  { captured_at: "2024-01-01T00:00:00Z", ball_speed: 150, club_speed: 100, spin_rate: 2500, club: "driver", notes: "a" },
  { captured_at: "2024-01-01T00:01:00Z", ball_speed: 152, club_speed: 101, spin_rate: 2550, club: "driver", notes: "b" },
  { captured_at: "2024-01-01T00:02:00Z", ball_speed: 148, club_speed: 99, spin_rate: 2450, club: "7i", notes: "c" },
  { captured_at: "2024-01-01T00:03:00Z", ball_speed: 155, club_speed: 103, spin_rate: 2600, club: "7i", notes: "d" },
  { captured_at: "2024-01-01T00:04:00Z", ball_speed: 151, club_speed: 100, spin_rate: 2500, club: "7i", notes: "e" },
];

const SAMPLE_RELATIONSHIPS: RelationshipsResponse = {
  method: "pearson",
  metrics: ["ball_speed", "club_speed", "spin_rate"],
  coefficients: [
    [1.0, 0.95, null],
    [0.95, 1.0, null],
    [null, null, 1.0],
  ],
  p_values: [
    [0.0, 0.001, null],
    [0.001, 0.0, null],
    [null, null, 0.0],
  ],
  adjusted_p_values: [
    [0.0, 0.002, null],
    [0.002, 0.0, null],
    [null, null, 0.0],
  ],
  pair_counts: [
    [5, 5, 0],
    [5, 5, 0],
    [0, 0, 5],
  ],
  partial_coefficients: null,
  derived_metrics: [],
  boolean_projected: [],
  edges: [
    {
      source: "ball_speed",
      target: "club_speed",
      coefficient: 0.95,
      p_value: 0.001,
      adjusted_p_value: 0.002,
      sample_count: 5,
      includes_derived_metric: false,
      includes_boolean_projection: false,
    },
  ],
};

const SAMPLE_MULTIVARIATE: MultivariateResponse = {
  pca: {
    metrics: ["ball_speed", "club_speed", "spin_rate"],
    component_names: ["PC1", "PC2", "PC3"],
    explained_variance_ratio: [0.7, 0.2, 0.1],
    loadings: [
      [0.8, 0.1, -0.2],
      [0.75, -0.1, 0.3],
      [0.6, 0.4, 0.1],
    ],
    scores: [[0.1, 0.2, 0.3]],
    sample_count: 5,
  },
  vif: {
    values: { ball_speed: 2.1, club_speed: null, spin_rate: 1.0 },
    sample_count: 5,
    warning_metrics: [],
  },
};

const { analyzeRelationshipsV2Mock, analyzeMultivariateV2Mock } = vi.hoisted(
  () => ({
    analyzeRelationshipsV2Mock: vi.fn(),
    analyzeMultivariateV2Mock: vi.fn(),
  }),
);

vi.mock("@/api/launchMonitorAnalytics", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/api/launchMonitorAnalytics")>();
  return {
    ...actual,
    analyzeRelationshipsV2: analyzeRelationshipsV2Mock,
    analyzeMultivariateV2: analyzeMultivariateV2Mock,
  };
});

describe("LaunchMonitorRelationshipsPanel", () => {
  beforeEach(() => {
    analyzeRelationshipsV2Mock.mockReset();
    analyzeMultivariateV2Mock.mockReset();
  });

  it("offers the dataset's numeric columns as Metrics and Controls options", () => {
    render(
      <LaunchMonitorRelationshipsPanel columns={COLUMNS} records={RECORDS} />,
    );

    for (const label of ["Metrics", "Partial-Correlation Controls"]) {
      const select = screen.getByLabelText(label);
      const options = Array.from(select.querySelectorAll("option")).map(
        (option) => option.textContent,
      );
      expect(options).toEqual(["ball_speed", "club_speed", "spin_rate"]);
    }
  });

  it("disables Run until at least two metrics are selected", async () => {
    const user = userEvent.setup();
    render(
      <LaunchMonitorRelationshipsPanel columns={COLUMNS} records={RECORDS} />,
    );

    const runButton = screen.getByRole("button", {
      name: /map interdependencies/i,
    });
    expect(runButton).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Metrics"), ["ball_speed"]);
    expect(runButton).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Metrics"), [
      "ball_speed",
      "club_speed",
    ]);
    expect(runButton).toBeEnabled();
  });

  it("Map Interdependencies sends the desktop-default method and threshold when no controls are selected", async () => {
    analyzeRelationshipsV2Mock.mockResolvedValue(SAMPLE_RELATIONSHIPS);
    const user = userEvent.setup();
    render(
      <LaunchMonitorRelationshipsPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Metrics"), [
      "ball_speed",
      "club_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /map interdependencies/i }),
    );

    await waitFor(() => {
      expect(analyzeRelationshipsV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(analyzeRelationshipsV2Mock).toHaveBeenCalledWith(
      RECORDS,
      ["ball_speed", "club_speed"],
      "pearson",
      [],
      0.3,
    );
    // PCA/VIF is a separate action, as on the desktop.
    expect(analyzeMultivariateV2Mock).not.toHaveBeenCalled();
  });

  it("Run PCA and VIF Diagnostics calls only the multivariate endpoint and reports like the desktop", async () => {
    analyzeMultivariateV2Mock.mockResolvedValue(SAMPLE_MULTIVARIATE);
    const user = userEvent.setup();
    render(
      <LaunchMonitorRelationshipsPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Metrics"), [
      "ball_speed",
      "club_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /run pca and vif diagnostics/i }),
    );

    await waitFor(() => {
      expect(analyzeMultivariateV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(analyzeMultivariateV2Mock).toHaveBeenCalledWith(RECORDS, [
      "ball_speed",
      "club_speed",
    ]);
    expect(analyzeRelationshipsV2Mock).not.toHaveBeenCalled();
    expect(await screen.findByTestId("lmr-status")).toHaveTextContent(
      "PCA/VIF complete for 5 complete shots. VIF >= 5: none.",
    );
  });

  it("sends the selected method, threshold, and controls with overlapping metrics dropped", async () => {
    analyzeRelationshipsV2Mock.mockResolvedValue(SAMPLE_RELATIONSHIPS);
    analyzeMultivariateV2Mock.mockResolvedValue(SAMPLE_MULTIVARIATE);
    const user = userEvent.setup();
    render(
      <LaunchMonitorRelationshipsPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Method"), "spearman");
    await user.selectOptions(screen.getByLabelText("Metrics"), [
      "ball_speed",
      "club_speed",
    ]);
    // "ball_speed" is also a selected metric and must be dropped from the
    // request's controls, mirroring `_read_relationship_params`.
    await user.selectOptions(
      screen.getByLabelText("Partial-Correlation Controls"),
      ["ball_speed", "spin_rate"],
    );
    const thresholdInput = screen.getByLabelText("Network Edge Threshold");
    fireEvent.change(thresholdInput, { target: { value: "0.5" } });

    await user.click(
      screen.getByRole("button", { name: /map interdependencies/i }),
    );

    await waitFor(() => {
      expect(analyzeRelationshipsV2Mock).toHaveBeenCalledTimes(1);
    });
    expect(analyzeRelationshipsV2Mock).toHaveBeenCalledWith(
      RECORDS,
      ["ball_speed", "club_speed"],
      "spearman",
      ["spin_rate"],
      0.5,
    );
  });

  it("renders the coefficient matrix, edges, PCA, and VIF tables with null as unavailable", async () => {
    analyzeRelationshipsV2Mock.mockResolvedValue(SAMPLE_RELATIONSHIPS);
    analyzeMultivariateV2Mock.mockResolvedValue(SAMPLE_MULTIVARIATE);
    const user = userEvent.setup();
    render(
      <LaunchMonitorRelationshipsPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Metrics"), [
      "ball_speed",
      "club_speed",
      "spin_rate",
    ]);
    await user.click(
      screen.getByRole("button", { name: /map interdependencies/i }),
    );
    await screen.findByTestId("lmr-coefficients-table");
    await user.click(
      screen.getByRole("button", { name: /run pca and vif diagnostics/i }),
    );

    const coefficientsTable = await screen.findByTestId(
      "lmr-coefficients-table",
    );
    expect(coefficientsTable).toHaveTextContent("0.9500");
    // ball_speed/spin_rate did not clear the pairwise floor and must render
    // as "unavailable", never a misleading "0".
    expect(coefficientsTable).toHaveTextContent("unavailable");

    const edgesTable = await screen.findByTestId("lmr-edges-table");
    expect(edgesTable).toHaveTextContent("ball_speed");
    expect(edgesTable).toHaveTextContent("club_speed");

    const pcaTable = await screen.findByTestId("lmr-pca-table");
    expect(pcaTable).toHaveTextContent("PC1");
    expect(pcaTable).toHaveTextContent("0.7000");

    const vifTable = await screen.findByTestId("lmr-vif-table");
    expect(vifTable).toHaveTextContent("2.1000");
    // An infinite VIF (perfectly collinear metrics) must render as
    // "unavailable", never "0".
    expect(vifTable).toHaveTextContent("unavailable");
  });

  it("surfaces the API error message on failure", async () => {
    analyzeRelationshipsV2Mock.mockRejectedValue(
      new Error("Columns not present: ['missing_metric']"),
    );
    analyzeMultivariateV2Mock.mockResolvedValue(SAMPLE_MULTIVARIATE);
    const user = userEvent.setup();
    render(
      <LaunchMonitorRelationshipsPanel columns={COLUMNS} records={RECORDS} />,
    );

    await user.selectOptions(screen.getByLabelText("Metrics"), [
      "ball_speed",
      "club_speed",
    ]);
    await user.click(
      screen.getByRole("button", { name: /map interdependencies/i }),
    );

    expect(
      await screen.findByText(
        /Relationship analysis could not run: Columns not present/,
      ),
    ).toBeInTheDocument();
  });

  it("clamps the edge threshold to [0, 1], matching the API contract", () => {
    render(
      <LaunchMonitorRelationshipsPanel columns={COLUMNS} records={RECORDS} />,
    );

    const thresholdInput = screen.getByLabelText("Network Edge Threshold");

    fireEvent.change(thresholdInput, { target: { value: "1.5" } });
    expect(thresholdInput).toHaveValue(1);

    fireEvent.change(thresholdInput, { target: { value: "-0.3" } });
    expect(thresholdInput).toHaveValue(0);
  });
});
