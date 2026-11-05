/**
 * Local Tauri desktop artifact with version, commit and checksum (#14451).
 *
 * `npm run tauri:artifact` builds the desktop bundles, then copies each
 * installer into the git-ignored ui/artifacts/ under a name carrying the app
 * version and commit, with a sha256 file beside it. Nothing is published.
 */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  ARTIFACT_DIR,
  artifactName,
  assertSameVersion,
  packageArtifacts,
  tauriBuildCommand,
} from "../../scripts/tauriArtifact.mjs";

const UI = resolve(__dirname, "../..");
const readJson = (path: string) =>
  JSON.parse(readFileSync(resolve(UI, path), "utf8"));
const app = {
  productName: "Golf Modeling Suite",
  version: "2.1.3",
  commit: "0123456789ab",
};

function fakeBundle(files: Record<string, string>) {
  const root = mkdtempSync(join(tmpdir(), "tauri-artifact-"));
  for (const [path, body] of Object.entries(files)) {
    mkdirSync(join(root, "bundle", path, ".."), { recursive: true });
    writeFileSync(join(root, "bundle", path), body);
  }
  return { bundleDir: join(root, "bundle"), outDir: join(root, "out") };
}

describe("Tauri local artifact (#14451)", () => {
  it("names an installer by app, version and commit, keeping its platform part", () => {
    expect(artifactName("Golf Modeling Suite_2.1.3_amd64.deb", app)).toBe(
      "golf-modeling-suite_2.1.3_0123456789ab_amd64.deb",
    );
    expect(artifactName("Golf Modeling Suite-2.1.3-1.x86_64.rpm", app)).toBe(
      "golf-modeling-suite_2.1.3_0123456789ab_1.x86_64.rpm",
    );
  });

  it("rejects a commit that is not a hex object name", () => {
    expect(() => artifactName("a.deb", { ...app, commit: "main" })).toThrow(
      /commit/,
    );
  });

  it("requires the app and package versions to agree", () => {
    expect(() => assertSameVersion("2.1.3", "2.1.4")).toThrow(
      /2\.1\.3.*2\.1\.4/,
    );
    expect(() =>
      assertSameVersion(
        readJson("src-tauri/tauri.conf.json").version,
        readJson("package.json").version,
      ),
    ).not.toThrow();
  });

  it("builds with the project's Tauri script and is wired to npm", () => {
    expect(tauriBuildCommand()).toEqual(["npm", ["run", "tauri:build"]]);
    expect(readJson("package.json").scripts["tauri:artifact"]).toBe(
      "node scripts/tauriArtifact.mjs",
    );
  });

  it("copies each installer with a sha256 file beside it", () => {
    const dirs = fakeBundle({
      "deb/Golf Modeling Suite_2.1.3_amd64.deb": "deb bytes",
      "appimage/Golf Modeling Suite_2.1.3_amd64.AppImage": "appimage bytes",
      "deb/control.txt": "not an installer",
    });
    const written = packageArtifacts({ ...dirs, ...app });
    expect(written.map((w) => w.name).sort()).toEqual([
      "golf-modeling-suite_2.1.3_0123456789ab_amd64.AppImage",
      "golf-modeling-suite_2.1.3_0123456789ab_amd64.deb",
    ]);
    for (const { path, name } of written) {
      const digest = createHash("sha256")
        .update(readFileSync(path))
        .digest("hex");
      expect(readFileSync(`${path}.sha256`, "utf8")).toBe(
        `${digest}  ${name}\n`,
      );
    }
  });

  it("fails when the build produced no installer", () => {
    const dirs = fakeBundle({ "deb/control.txt": "not an installer" });
    expect(() => packageArtifacts({ ...dirs, ...app })).toThrow(/no installer/);
    const missing = { ...dirs, bundleDir: join(dirs.outDir, "none") };
    expect(() => packageArtifacts({ ...missing, ...app })).toThrow(
      /no installer/,
    );
  });

  it("writes into a git-ignored folder, so no binary is committed", () => {
    const ignored = spawnSync(
      "git",
      ["check-ignore", "-q", join(ARTIFACT_DIR, "app.deb")],
      { cwd: UI },
    );
    expect(ignored.status).toBe(0);
  });
});
