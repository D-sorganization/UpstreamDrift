import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { fetchPutterPresets } from "@/api/puttingClient";
import { usePuttingStore } from "@/stores/usePuttingStore";
import golden from "@/utils/putterHead.golden.json";
import { PutterHeadControl } from "./PutterHeadControl";

vi.mock("@/api/puttingClient", () => ({ fetchPutterPresets: vi.fn() }));

function renderControl() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <PutterHeadControl />
    </QueryClientProvider>,
  );
}

const [toeHang] = golden.presets;

describe("PutterHeadControl (#14462)", () => {
  beforeEach(() => {
    usePuttingStore.getState().reset();
    vi.mocked(fetchPutterPresets).mockResolvedValue({
      presets: golden.presets,
    });
  });

  it("edits the typical head's mass and MOI in SI", () => {
    renderControl();
    fireEvent.change(screen.getByLabelText("Head Mass (g)"), {
      target: { value: "400" },
    });
    fireEvent.change(screen.getByLabelText("Head MOI (g·cm²)"), {
      target: { value: "5000" },
    });
    const { parameters } = usePuttingStore.getState();
    expect(parameters.head_mass_kg).toBeCloseTo(0.4, 9);
    expect(parameters.head_moi_kg_m2).toBeCloseTo(5e-4, 9);
    expect(parameters.balance_type ?? null).toBeNull();
  });

  it("offers Typical plus every core preset and locks a preset's head", async () => {
    renderControl();
    await screen.findByRole("option", { name: toeHang.name });
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Typical",
      ...golden.presets.map((p) => p.name),
    ]);
    fireEvent.change(screen.getByLabelText("Balance Type"), {
      target: { value: toeHang.key },
    });
    const { parameters } = usePuttingStore.getState();
    expect(parameters).toEqual(
      expect.objectContaining({
        balance_type: toeHang.key,
        head_mass_kg: toeHang.head_mass_kg,
        hosel_toe_m: toeHang.hosel_toe_m,
      }),
    );
    expect(screen.getByLabelText("Head Mass (g)")).toBeDisabled();
    expect(screen.getByLabelText("Head Mass (g)")).toHaveValue(345);

    fireEvent.change(screen.getByLabelText("Balance Type"), {
      target: { value: "" },
    });
    expect(usePuttingStore.getState().parameters.balance_type).toBeNull();
    expect(screen.getByLabelText("Head Mass (g)")).toBeEnabled();
  });

  it("keeps the typical head usable when the presets cannot load", async () => {
    vi.mocked(fetchPutterPresets).mockRejectedValue(new Error("offline"));
    renderControl();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /presets unavailable/i,
    );
    expect(screen.getByLabelText("Head Mass (g)")).toBeEnabled();
  });
});
