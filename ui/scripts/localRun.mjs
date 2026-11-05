#!/usr/bin/env node
/**
 * Build and run the React web UI from a clean checkout in one command (#14450).
 *
 *   npm run local [-- --port 8000]         install, build, then serve through the API
 *   npm run build:clean [-- --skip-install] install and build only (CI clean-build check)
 *
 * The build must leave every tracked file unchanged (the lockfile above all)
 * and must produce dist/index.html, which the API serves at the given port.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const MIN_NODE_MAJOR = 22;

const UI = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = resolve(UI, "..");

/** Throw unless `version` (like process.version) is at least MIN_NODE_MAJOR. */
export function assertNodeVersion(version) {
  const major = Number(/^v(\d+)\./.exec(version)?.[1]);
  if (!(major >= MIN_NODE_MAJOR)) {
    throw new Error(
      `Node ${version} is too old; the UI needs Node >= ${MIN_NODE_MAJOR}`,
    );
  }
}

/** The commands that build the UI, run in ui/: a lockfile install, then the build. */
export function buildSteps({ install }) {
  return [...(install ? [["npm", ["ci"]]] : []), ["npm", ["run", "build"]]];
}

/** The API launcher command, run from the repository root; it serves ui/dist. */
export function launchCommand(port) {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new RangeError(`port must be an integer in 1-65535, got ${port}`);
  }
  return ["python3", ["launch_upstream_drift.py", "--port", String(port)]];
}

function run([command, args], cwd) {
  const result = spawnSync(command, args, {
    cwd,
    stdio: "inherit",
    shell: process.platform === "win32",
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(" ")} failed (exit ${result.status})`,
    );
  }
}

function trackedChanges() {
  const result = spawnSync(
    "git",
    ["status", "--porcelain", "--untracked-files=no", "--", "."],
    {
      cwd: UI,
      encoding: "utf8",
    },
  );
  return result.status === 0 ? result.stdout.trim() : "";
}

function main(argv) {
  assertNodeVersion(process.version);
  const before = trackedChanges();
  for (const step of buildSteps({ install: !argv.includes("--skip-install") }))
    run(step, UI);
  if (!existsSync(resolve(UI, "dist", "index.html"))) {
    throw new Error("the build produced no dist/index.html");
  }
  const after = trackedChanges();
  if (after !== before)
    throw new Error(`the build changed tracked files:\n${after}`);
  if (argv.includes("--build-only")) return;
  const flag = argv.indexOf("--port");
  run(launchCommand(flag >= 0 ? Number(argv[flag + 1]) : 8000), ROOT);
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(`local run: ${error.message}`);
    process.exit(1);
  }
}
