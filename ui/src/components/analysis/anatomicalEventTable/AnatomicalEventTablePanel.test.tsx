import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { AnatomicalEventTablePanel } from "./AnatomicalEventTablePanel";
import golden from "./fixtures/eventTableView.json";

const apiFetch = vi.hoisted(() => vi.fn());
vi.mock("@/api/fetch", () => ({ apiFetch }));

afterEach(() => {
  cleanup();
  apiFetch.mockReset();
});

describe("AnatomicalEventTablePanel", () => {
  it("only sends calibrated frames after deliberate submission and renders the API document", async () => {
    apiFetch.mockResolvedValue(golden.document);
    render(<AnatomicalEventTablePanel />);
    expect(apiFetch).not.toHaveBeenCalled();
    const frames = {
      time_s: [0, 1],
      events: { address: 0 },
      arms: {},
      handedness: "RH",
      event_source: "manual",
      receipt_path: "receipt.json",
    };
    fireEvent.change(screen.getByLabelText("Calibrated Frames (JSON)"), {
      target: { value: JSON.stringify(frames) },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Compute Event Table" }),
    );
    expect(
      await screen.findByTestId("anatomical-event-table"),
    ).toBeInTheDocument();
    expect(JSON.parse(apiFetch.mock.calls[0][1].body)).toEqual({
      ...frames,
      format: "json",
    });
  });

  it.each(["{", "null", "[]", "42"])(
    "rejects invalid request %s without sending data",
    (value) => {
      render(<AnatomicalEventTablePanel />);
      fireEvent.change(screen.getByLabelText("Calibrated Frames (JSON)"), {
        target: { value },
      });
      fireEvent.click(
        screen.getByRole("button", { name: "Compute Event Table" }),
      );
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(apiFetch).not.toHaveBeenCalled();
    },
  );

  it("preserves the request and shows the server refusal for correction", async () => {
    apiFetch.mockRejectedValue(new Error("events must be within time_s"));
    render(<AnatomicalEventTablePanel />);
    const input = screen.getByLabelText("Calibrated Frames (JSON)");
    fireEvent.change(input, { target: { value: '{"time_s":[]}' } });
    fireEvent.click(
      screen.getByRole("button", { name: "Compute Event Table" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "events must be within time_s",
    );
    expect(input).toHaveValue('{"time_s":[]}');
  });
});
