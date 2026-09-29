import { useEffect, useState } from "react";
import { usePageTitle } from "@/utils/usePageTitle";

/** Where the API mounts the vendored Impact Explorer web bundle. */
export const IMPACT_EXPLORER_APP_URL = "/impact-explorer-app/";

/**
 * Availability of the embedded bundle. `checking` renders nothing visual so
 * the page never flashes the fallback before the probe resolves.
 */
export type BundleState = "checking" | "available" | "missing";

/**
 * Probe whether the vendored bundle is mounted. The API mounts
 * `/impact-explorer-app` only when the build exists on disk
 * (`_mount_impact_explorer_directory`), so a 404/503 here means "not built",
 * never a transient error worth retrying in a loop.
 */
async function probeBundle(fetchImpl: typeof fetch): Promise<BundleState> {
  try {
    const res = await fetchImpl(IMPACT_EXPLORER_APP_URL, { method: "HEAD" });
    return res.ok ? "available" : "missing";
  } catch {
    return "missing";
  }
}

/**
 * Rate of Closure Impact Explorer (launcher tile `rate_of_closure`).
 *
 * The full product is the React app vendored at
 * `vendor/ud-tools/src/rate_of_closure/web`. When Tools' own build of that
 * app is present the API serves it at {@link IMPACT_EXPLORER_APP_URL} and
 * this page embeds it; otherwise the page states plainly how to get it —
 * either build the bundle or use the desktop launcher tile. It never
 * pretends: no dead buttons, no silent blank frame (issue tracked in the
 * launcher-manifest parity contract, #7461).
 */
export function ImpactExplorerPage() {
  usePageTitle("Impact Explorer");
  const [bundle, setBundle] = useState<BundleState>("checking");

  useEffect(() => {
    let cancelled = false;
    void probeBundle(fetch).then((state) => {
      if (!cancelled) setBundle(state);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  if (bundle === "available") {
    return (
      <iframe
        src={IMPACT_EXPLORER_APP_URL}
        title="Rate of Closure Impact Explorer"
        className="h-screen w-full border-0"
        data-testid="impact-explorer-frame"
      />
    );
  }

  if (bundle === "checking") {
    return (
      <div
        className="flex h-screen items-center justify-center bg-gray-900"
        data-testid="impact-explorer-checking"
      />
    );
  }

  return (
    <div className="flex h-screen items-center justify-center bg-gray-900 p-4">
      <div className="w-full max-w-xl rounded-md bg-gray-800 p-8 shadow-lg">
        <h1 className="text-xl font-semibold text-white">
          Impact Explorer Web Bundle Is Not Built
        </h1>
        <p className="mt-3 text-sm text-gray-300">
          The Rate of Closure Impact Explorer — the full swing, impact,
          ball-flight, and putting simulation suite — ships as a React app
          inside the vendored Tools tree. Build it once and this page serves the
          real product:
        </p>
        <pre className="mt-4 overflow-x-auto rounded bg-gray-950 p-3 text-xs text-gray-200">
          {`cd vendor/ud-tools/src/rate_of_closure/web
npm ci
npm run build -- --base=/impact-explorer-app/`}
        </pre>
        <p className="mt-3 text-xs text-gray-400">
          The <span className="font-mono">--base</span> flag is required: the
          bundle is served under{" "}
          <span className="font-mono">{IMPACT_EXPLORER_APP_URL}</span>, and a
          default-base build would request its assets from the wrong paths.
        </p>
        <p className="mt-4 text-sm text-gray-300">
          Prefer the desktop experience? The launcher tile{" "}
          <span className="font-semibold text-white">
            Rate of Closure Impact Explorer
          </span>{" "}
          (Simulation category) opens the native PyQt app with the same
          features.
        </p>
      </div>
    </div>
  );
}
