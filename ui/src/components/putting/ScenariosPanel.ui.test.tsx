import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  compareScenarios,
  exportSetup,
  importSetup,
  runScenario,
} from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import { defaultNoise } from "@/utils/dispersion";
import golden from "@/utils/puttingSetup.golden.json";
import { setupArgs, type Scenario } from "@/utils/puttingSetup";
import { ScenariosPanel } from "./ScenariosPanel";

vi.mock("@/api/puttingClient", () => ({
  runScenario: vi.fn(),
  compareScenarios: vi.fn(),
  exportSetup: vi.fn(),
  importSetup: vi.fn(),
}));

const [{ setup }] = golden.cases;
const scenario = (name: string, make_rate: number): Scenario => ({
  name,
  setup,
  result: { make_rate, mean_leave_ft: 1.5 },
});

function renderPanel(download = vi.fn()) {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <ScenariosPanel download={download} />
    </QueryClientProvider>,
  );
  return download;
}

async function save(name: string, make_rate: number) {
  vi.mocked(runScenario).mockResolvedValueOnce(scenario(name, make_rate));
  fireEvent.change(screen.getByLabelText("Scenario Name"), {
    target: { value: name },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save Scenario" }));
  await screen.findByText(name);
}

describe("ScenariosPanel (#14472)", () => {
  beforeEach(() => {
    usePuttingStore.getState().reset();
    usePuttingStore.getState().setUnitSystem("imperial");
    vi.mocked(runScenario).mockReset();
    vi.mocked(compareScenarios).mockReset();
    vi.mocked(exportSetup).mockReset();
    vi.mocked(importSetup).mockReset();
  });

  it("saves the page's putt as a core scenario and compares saved ones", async () => {
    renderPanel();
    await save("Flat 10 ft", 0.62);
    const { parameters, noise } = usePuttingStore.getState();
    expect(vi.mocked(runScenario).mock.calls[0][0]).toEqual({
      name: "Flat 10 ft",
      setup: setupArgs(parameters, noise),
      n: 100,
      seed: parameters.random_seed,
    });
    expect(screen.getByText("62.0%")).toBeInTheDocument();

    await save("Fast Green", 0.5);
    vi.mocked(compareScenarios).mockResolvedValue({
      baseline: "Flat 10 ft",
      rows: [
        { make_rate_delta: 0, mean_leave_delta_ft: 0 },
        { make_rate_delta: -0.12, mean_leave_delta_ft: 0.3 },
      ],
    });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    await waitFor(() =>
      expect(
        vi.mocked(compareScenarios).mock.calls[0][0].scenarios,
      ).toHaveLength(2),
    );
    const table = await screen.findByRole("table", {
      name: /Against Flat 10 ft/,
    });
    expect(table).toHaveTextContent("-12.0 pts");
    expect(table).toHaveTextContent("+0.30 ft");
  });

  it("exports the core's setup document as a JSON file", async () => {
    const document = {
      schema_version: golden.schema_version,
      suite: golden.suite,
    };
    vi.mocked(exportSetup).mockResolvedValue(document);
    const download = renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "Export JSON" }));
    await waitFor(() => expect(download).toHaveBeenCalledTimes(1));
    const [blob, name] = download.mock.calls[0];
    expect(name).toBe("putting-setup-v4.0.0.json");
    expect(JSON.parse(await (blob as Blob).text())).toEqual(document);
  });

  it("imports a studio file onto the page through the core", async () => {
    vi.mocked(importSetup).mockResolvedValue({
      setup,
      scenarios: [scenario("Saved", 0.4)],
      studio_scenarios: 2,
    });
    renderPanel();
    const file = new File(
      [JSON.stringify(golden.studio_import.document)],
      "s.json",
    );
    fireEvent.change(screen.getByLabelText("Import JSON"), {
      target: { files: [file] },
    });
    await screen.findByText("Saved");
    expect(vi.mocked(importSetup).mock.calls[0][0]).toEqual({
      document: golden.studio_import.document,
    });
    const { parameters, noise } = usePuttingStore.getState();
    expect(noise).toEqual(defaultNoise());
    expect(setupArgs(parameters, noise).distance_ft).toBeCloseTo(
      setup.distance_ft,
      10,
    );
    expect(
      screen.getByText(/2 studio scenario rows are text/),
    ).toBeInTheDocument();
  });

  it("reports a file that is not JSON", async () => {
    renderPanel();
    fireEvent.change(screen.getByLabelText("Import JSON"), {
      target: { files: [new File(["not json"], "bad.json")] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(/JSON/);
    expect(importSetup).not.toHaveBeenCalled();
  });
});
