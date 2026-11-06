/**
 * One-command local run from a clean checkout (issue #14450).
 *
 * `npm run local` installs from the lockfile, builds, checks the bundle and
 * the tracked tree, then starts the API, which serves `ui/dist`.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  MIN_NODE_MAJOR,
  assertNodeVersion,
  buildSteps,
  launchCommand,
} from "../../scripts/localRun.mjs";

const pkg = JSON.parse(
  readFileSync(resolve(__dirname, "../../package.json"), "utf8"),
);

describe("local run (#14450)", () => {
  it("installs from the lockfile and then builds", () => {
    expect(buildSteps({ install: true })).toEqual([
      ["npm", ["ci"]],
      ["npm", ["run", "build"]],
    ]);
    expect(buildSteps({ install: false })).toEqual([["npm", ["run", "build"]]]);
  });

  it("starts the API launcher, which serves the built UI, on the given port", () => {
    expect(launchCommand(8123)).toEqual([
      "python3",
      ["launch_upstream_drift.py", "--port", "8123"],
    ]);
    expect(() => launchCommand(0)).toThrow(RangeError);
  });

  it("refuses a Node older than the CI toolchain", () => {
    expect(() => assertNodeVersion(`v${MIN_NODE_MAJOR - 1}.9.0`)).toThrow(
      /Node/,
    );
    expect(() => assertNodeVersion(`v${MIN_NODE_MAJOR}.0.0`)).not.toThrow();
  });

  it("exposes the one command and pins the Node engine", () => {
    expect(pkg.scripts.local).toBe("node scripts/localRun.mjs");
    expect(pkg.scripts["build:clean"]).toBe(
      "node scripts/localRun.mjs --build-only",
    );
    expect(pkg.engines.node).toBe(`>=${MIN_NODE_MAJOR}`);
  });
});
