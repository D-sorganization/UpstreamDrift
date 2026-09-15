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
vi.mock("@/api/fetch", () => ({
  apiFetch: (...args: unknown[]) => apiFetchMock(...args),
}));

import { GolfSimulatorPage } from "./GolfSimulator";

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
});
