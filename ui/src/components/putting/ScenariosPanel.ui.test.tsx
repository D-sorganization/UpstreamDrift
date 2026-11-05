import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { compareScenarios, runScenario } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import golden from "@/utils/puttingSetup.golden.json";
import { setupArgs, type Scenario } from "@/utils/puttingSetup";
import { ScenariosPanel } from "./ScenariosPanel";

vi.mock("@/api/puttingClient", () => ({
  runScenario: vi.fn(),
  compareScenarios: vi.fn(),
}));

const [{ setup }] = golden.cases;
const scenario = (name: string, make_rate: number): Scenario => ({
  name,
  setup,
  result: { make_rate, mean_leave_ft: 1.5 },
});

function renderPanel() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <ScenariosPanel />
    </QueryClientProvider>,
  );
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
});
