import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import golden from "./fixtures/eventTableView.json";
import { AnatomicalEventTableFromApi } from "./AnatomicalEventTableFromApi";
import type { AnatomicalEventTableRequest } from "@/api/anatomicalEventTable";

const apiFetch = vi.hoisted(() => vi.fn());
vi.mock("@/api/fetch", () => ({ apiFetch }));

const frames = { rotation_world: [], origin_world: [], calibration_id: "cal-1" };
const request: AnatomicalEventTableRequest = {
  time_s: [0, 0.01],
  events: { address: 0 },
  handedness: "RH",
  event_source: "detector-v1",
  receipt_path: "receipts/swing.json",
  arms: { left: { humerus: frames, forearm: frames, hand: frames } },
};

afterEach(() => {
  cleanup();
  apiFetch.mockReset();
});

function bodyCells(): string[][] {
  const table = screen.getByTestId("anatomical-event-table");
  const [, body] = within(table).getAllByRole("rowgroup");
  return within(body)
    .getAllByRole("row")
    .map((row) =>
      within(row)
        .getAllByRole("cell")
        .map((cell) => cell.textContent ?? ""),
    );
}

describe("AnatomicalEventTableFromApi (#13175)", () => {
  it("fetches the document from the API and renders the golden cells", async () => {
    apiFetch.mockResolvedValue(golden.document);
    render(<AnatomicalEventTableFromApi request={request} />);
    expect(await screen.findByTestId("anatomical-event-table")).toBeInTheDocument();
    expect(apiFetch).toHaveBeenCalledWith(
      "/api/biomechanics/anatomical-event-table",
      expect.objectContaining({ method: "POST" }),
    );
    expect(
      JSON.parse(apiFetch.mock.calls[0][1].body as string).format,
    ).toBe("json");
    expect(bodyCells()).toEqual(golden.view.rows);
  });

  it("shows a loading status until the document arrives", async () => {
    apiFetch.mockReturnValue(new Promise(() => undefined));
    render(<AnatomicalEventTableFromApi request={request} />);
    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
    expect(screen.queryByTestId("anatomical-event-table")).not.toBeInTheDocument();
  });

  it("shows the refusal message when the request fails", async () => {
    apiFetch.mockRejectedValue(new Error("events must be within time_s"));
    render(<AnatomicalEventTableFromApi request={request} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /events must be within time_s/,
    );
    expect(screen.queryByTestId("anatomical-event-table")).not.toBeInTheDocument();
  });

  it("clears a previous error when the request changes and succeeds", async () => {
    apiFetch.mockRejectedValueOnce(new Error("bad bundle"));
    const { rerender } = render(<AnatomicalEventTableFromApi request={request} />);
    await screen.findByRole("alert");
    apiFetch.mockResolvedValue(golden.document);
    rerender(<AnatomicalEventTableFromApi request={{ ...request }} />);
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(await screen.findByTestId("anatomical-event-table")).toBeInTheDocument();
  });

  it("aborts the in-flight request on unmount", () => {
    apiFetch.mockReturnValue(new Promise(() => undefined));
    const { unmount } = render(<AnatomicalEventTableFromApi request={request} />);
    const signal = apiFetch.mock.calls[0][1].signal as AbortSignal;
    expect(signal.aborted).toBe(false);
    unmount();
    expect(signal.aborted).toBe(true);
  });
});
