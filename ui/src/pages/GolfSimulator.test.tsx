/**
 * Tests for the GolfSimulator page.
 *
 * Covers destination selection, shot lifecycle, status badge changes,
 * replay transport controls, and delivery reconciliation.
 *
 * See issue #10196 (GS-07).
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const apiFetchMock = vi.fn();
vi.mock("@/api/fetch", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/fetch")>()),
  apiFetch: (...args: unknown[]) => apiFetchMock(...args),
}));

import { GolfSimulatorPage } from "./GolfSimulator";
import { ApiError } from "@/api/fetch";

describe("GolfSimulatorPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads available destinations on mount", async () => {
    apiFetchMock.mockResolvedValueOnce({
      destinations: [
        {
          destination_id: "local",
          name: "Local Reference Simulator",
          description: "Local physics",
          is_connected: true,
          capabilities: {
            shot_input: { state: "supported", evidence: "in-process" },
          },
        },
      ],
    });

    render(<GolfSimulatorPage />);

    await waitFor(() => {
      expect(apiFetchMock).toHaveBeenCalledWith("/tools/golf-simulator/destinations");
    });
    expect(screen.getByText("Local Reference Simulator")).toBeInTheDocument();
    expect(screen.getByText("Shot Input:")).toBeInTheDocument();
  });

  it("handles connect, prepare, arm, and submit workflow", async () => {
    // 1. Initial destinations
    apiFetchMock.mockResolvedValueOnce({ destinations: [] });

    render(<GolfSimulatorPage />);

    // 2. Connect
    apiFetchMock.mockResolvedValueOnce({ session_id: "web-session", state: "idle" });
    const connectBtn = screen.getByRole("button", { name: "Connect" });
    fireEvent.click(connectBtn);

    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("CONNECTED");
    });

    // 3. Prepare
    apiFetchMock.mockResolvedValueOnce({ prepared_shot_id: "prep-123" });
    const prepareBtn = screen.getByRole("button", { name: "Prepare Shot" });
    fireEvent.click(prepareBtn);

    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("PREPARED");
    });

    // 4. Arm
    apiFetchMock.mockResolvedValueOnce({ arm_token: "arm-token-abc", state: "armed" });
    const armBtn = screen.getByRole("button", { name: "Arm for Impact" });
    fireEvent.click(armBtn);

    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("ARMED");
    });

    // 5. Submit
    apiFetchMock.mockResolvedValueOnce({ state: "confirmed_accepted", shot_id: "shot-1" });
    const submitBtn = screen.getByRole("button", { name: "Trigger Impact Submit" });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("ACCEPTED");
    });
  });

  it("allows reconciliation when in uncertain state", async () => {
    apiFetchMock.mockResolvedValueOnce({ destinations: [] });
    render(<GolfSimulatorPage />);

    apiFetchMock.mockResolvedValueOnce({ session_id: "web-session", state: "idle" });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("CONNECTED");
    });

    apiFetchMock.mockResolvedValueOnce({ prepared_shot_id: "prep-123" });
    fireEvent.click(screen.getByRole("button", { name: "Prepare Shot" }));
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("PREPARED");
    });

    apiFetchMock.mockResolvedValueOnce({ arm_token: "arm-token-abc", state: "armed" });
    fireEvent.click(screen.getByRole("button", { name: "Arm for Impact" }));
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("ARMED");
    });

    apiFetchMock.mockResolvedValueOnce({ state: "unknown_ambiguous", shot_id: "shot-1" });
    fireEvent.click(screen.getByRole("button", { name: "Trigger Impact Submit" }));

    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("SENT_UNCONFIRMED");
    });

    // Fill operator evidence and reconcile
    const input = screen.getByLabelText("Operator Evidence");
    fireEvent.change(input, { target: { value: "Visual confirmation on screen" } });

    apiFetchMock.mockResolvedValueOnce({ state: "confirmed_accepted", shot_id: "shot-1" });
    const confirmBtn = screen.getByRole("button", { name: "Confirm Delivery" });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("VISUALLY_VERIFIED");
    });
  });
  it("keeps the console within the viewport at phone width (#14282)", async () => {
    apiFetchMock.mockResolvedValueOnce({ destinations: [] });
    render(<GolfSimulatorPage />);
    const heading = await screen.findByRole("heading", { name: "Golf Simulator Console" });
    expect(heading.parentElement).toHaveClass("w-full", "min-w-0", "max-w-4xl");
    for (const control of ["Connect", "Stop", "Confirm Delivery"]) {
      expect(screen.getByRole("button", { name: control }).parentElement).toHaveClass("flex-wrap");
    }
    expect(screen.getByLabelText("Operator Evidence")).toHaveClass("min-w-0");
  });

  it("shows plain-language copy with Retry when the API answers with HTML (#14256)", async () => {
    apiFetchMock.mockRejectedValueOnce(
      new ApiError("unexpected-response", 200, "/tools/golf-simulator/destinations", "html"),
    );

    render(<GolfSimulatorPage />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Simulator API not reachable");
    expect(alert).not.toHaveTextContent(/Unexpected token|not valid JSON/);

    apiFetchMock.mockResolvedValueOnce({ destinations: [] });
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    await waitFor(() => {
      expect(apiFetchMock).toHaveBeenCalledTimes(2);
    });
    await waitFor(() => {
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
  });

  it("explains why lifecycle and transport controls are disabled while disconnected (#14256)", async () => {
    apiFetchMock.mockResolvedValueOnce({ destinations: [] });

    render(<GolfSimulatorPage />);

    const reason = await screen.findByText(/Connect to a simulator to enable/);
    for (const name of ["Prepare Shot", "Arm for Impact", "Trigger Impact Submit", "Play", "Pause", "Stop"]) {
      const button = screen.getByRole("button", { name });
      expect(button).toBeDisabled();
      expect(button).toHaveAttribute("aria-describedby", reason.id);
    }

    apiFetchMock.mockResolvedValueOnce({ session_id: "web-session", state: "idle" });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    await waitFor(() => {
      expect(screen.queryByText(/Connect to a simulator to enable/)).not.toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: "Prepare Shot" })).not.toHaveAttribute("aria-describedby");
  });
});
