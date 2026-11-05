#!/usr/bin/env node
/**
 * Build a local Tauri desktop installer with version, commit and checksum (#14451).
 *
 *   npm run tauri:artifact                 build the bundles, then package them
 *   npm run tauri:artifact -- --skip-build package bundles already built
 *
 * Each installer is copied into the git-ignored ui/artifacts/ as
 * <app>_<version>_<commit>_<platform part>, with a sha256sum-format
 * <name>.sha256 beside it. Nothing is uploaded or published as a release.
 */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { run } from "./localRun.mjs";

const UI = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const BUNDLE_DIR = resolve(UI, "src-tauri", "target", "release", "bundle");
export const ARTIFACT_DIR = resolve(UI, "artifacts");
const INSTALLER = /\.(deb|rpm|AppImage|msi|exe|dmg)$/;

const escape = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** The artifact name for a Tauri installer file: app, version, commit, platform part. */
export function artifactName(file, { productName, version, commit }) {
  if (!/^[0-9a-f]{7,40}$/.test(commit)) {
    throw new Error(`commit must be a hex object name, got "${commit}"`);
  }
  const prefix = new RegExp(
    `^${escape(productName)}[_-]${escape(version)}[_-]`,
  );
  const slug = productName.toLowerCase().replace(/\s+/g, "-");
  return `${slug}_${version}_${commit}_${file.replace(prefix, "")}`;
}

/** Throw unless the Tauri app version equals the package version. */
export function assertSameVersion(appVersion, packageVersion) {
  if (appVersion !== packageVersion) {
    throw new Error(
      `tauri.conf.json version ${appVersion} differs from package.json version ${packageVersion}`,
    );
  }
}

/** The command that builds the desktop bundles, run in ui/. */
export function tauriBuildCommand() {
  return ["npm", ["run", "tauri:build"]];
}

function installers(dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile() && INSTALLER.test(entry.name))
    .map((entry) => join(entry.parentPath, entry.name));
}

/**
 * Copy every installer under `bundleDir` into `outDir` under its artifact name,
 * each with a sha256 file. Returns the written `{ name, path }` list.
 */
export function packageArtifacts({ bundleDir, outDir, ...app }) {
  const found = installers(bundleDir);
  if (found.length === 0)
    throw new Error(`no installer found under ${bundleDir}`);
  mkdirSync(outDir, { recursive: true });
  return found.map((source) => {
    const name = artifactName(basename(source), app);
    const path = join(outDir, name);
    copyFileSync(source, path);
    const digest = createHash("sha256")
      .update(readFileSync(path))
      .digest("hex");
    writeFileSync(`${path}.sha256`, `${digest}  ${name}\n`);
    return { name, path };
  });
}

function headCommit() {
  const result = spawnSync("git", ["rev-parse", "--short=12", "HEAD"], {
    cwd: UI,
    encoding: "utf8",
  });
  if (result.status !== 0) throw new Error("cannot read the git commit");
  return result.stdout.trim();
}

function main(argv) {
  const readJson = (path) =>
    JSON.parse(readFileSync(resolve(UI, path), "utf8"));
  const { productName, version } = readJson("src-tauri/tauri.conf.json");
  assertSameVersion(version, readJson("package.json").version);
  if (!argv.includes("--skip-build")) run(tauriBuildCommand(), UI);
  const app = { productName, version, commit: headCommit() };
  const written = packageArtifacts({
    bundleDir: BUNDLE_DIR,
    outDir: ARTIFACT_DIR,
    ...app,
  });
  for (const { path } of written) console.log(`${path}\n${path}.sha256`);
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(`tauri artifact: ${error.message}`);
    process.exit(1);
  }
}
