/**
 * Tests for the Impact Explorer page (launcher tile `rate_of_closure`).
 *
 * The page embeds the vendored Rate of Closure web bundle when the API has
 * mounted it, and otherwise states plainly how to build it — including the
 * `--base` flag, without which the embedded app would request its assets
 * from the host app's paths. Both states are pinned here so the page can
 * never silently render a dead frame.
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

import { ImpactExplorerPage, IMPACT_EXPLORER_APP_URL } from "./ImpactExplorer";

function mockFetchStatus(ok: boolean) {
  const fetchMock = vi.fn().mockResolvedValue({ ok } as Response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ImpactExplorerPage", () => {
  it("embeds the mounted bundle in an iframe when the probe succeeds", async () => {
    const fetchMock = mockFetchStatus(true);
    render(<ImpactExplorerPage />);

    const frame = await screen.findByTestId("impact-explorer-frame");
    expect(frame).toHaveAttribute("src", IMPACT_EXPLORER_APP_URL);
    expect(frame).toHaveAttribute("title", "Rate of Closure Impact Explorer");
    expect(fetchMock).toHaveBeenCalledWith(IMPACT_EXPLORER_APP_URL, {
      method: "HEAD",
    });
  });

  it("shows the build instructions, with the required --base flag, when the bundle is missing", async () => {
    mockFetchStatus(false);
    render(<ImpactExplorerPage />);

    await waitFor(() =>
      expect(
        screen.getByText("Impact Explorer Web Bundle Is Not Built"),
      ).toBeInTheDocument(),
    );
    expect(
      screen.getByText(/--base=\/impact-explorer-app\//),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("impact-explorer-frame")).toBeNull();
  });

  it("treats a network failure as missing rather than crashing", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    render(<ImpactExplorerPage />);

    await waitFor(() =>
      expect(
        screen.getByText("Impact Explorer Web Bundle Is Not Built"),
      ).toBeInTheDocument(),
    );
  });

  it("renders the neutral probe state first, never the fallback flash", () => {
    // A promise that never resolves keeps the page in `checking`.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockReturnValue(new Promise(() => undefined)),
    );
    render(<ImpactExplorerPage />);

    expect(screen.getByTestId("impact-explorer-checking")).toBeInTheDocument();
    expect(
      screen.queryByText("Impact Explorer Web Bundle Is Not Built"),
    ).toBeNull();
  });
});
