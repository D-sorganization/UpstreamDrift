import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BunkerFlightView } from "./BunkerFlightView";

const record = {
  refused: false,
  carry: { carry_m: 8 },
  flight: {
    flight_m: [
      [0, 0, 0],
      [4, 1, 3],
      [8, 0, 0],
    ],
    rollout_m: [
      [8, 0, 0],
      [10, -1, 0],
    ],
    carry_m: 8,
    assumption: "Uncalibrated launch; flat 10 ft stimp green",
  },
};

describe("bunker flight display", () => {
  it("draws both projections with landing/rest markers and the model assumption", () => {
    const { container } = render(<BunkerFlightView record={record} />);
    expect(
      screen.getByRole("img", { name: "Ball Flight: Downrange vs Height (m)" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Green Landing: Downrange vs Left (m)" }),
    ).toBeInTheDocument();
    expect(container.querySelectorAll("polyline")).toHaveLength(4);
    expect(container.querySelectorAll("circle")).toHaveLength(4);
    expect(screen.getByText(record.flight.assumption)).toBeInTheDocument();
    expect(screen.getByText("Carry 8.00 m")).toBeInTheDocument();
  });

  it("converts coordinates and labels to yards without changing the trace", () => {
    const { container } = render(<BunkerFlightView record={record} />);
    const before = container.querySelector("polyline")?.getAttribute("points");
    fireEvent.change(screen.getByLabelText("Flight units"), { target: { value: "yd" } });
    expect(screen.getByText("Carry 8.75 yd")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Ball Flight: Downrange vs Height (yd)" }),
    ).toBeInTheDocument();
    expect(container.querySelector("polyline")?.getAttribute("points")).toBe(before);
  });

  it.each([
    { ...record, refused: true },
    { ...record, flight: undefined },
    { ...record, flight: null },
    { ...record, carry: { carry_m: Infinity } },
    {
      ...record,
      flight: {
        ...record.flight,
        flight_m: [
          [0, 0, NaN],
          [8, 0, 0],
        ],
      },
    },
    {
      ...record,
      flight: {
        ...record.flight,
        rollout_m: [
          [9, 0, 0],
          [10, 0, 0],
        ],
      },
    },
  ])("suppresses every trajectory for refused, missing or invalid data", (input) => {
    render(<BunkerFlightView record={input} />);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/REFUSED|unavailable/);
  });
});
