import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";

const ALLOWED = new Set([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]);
const SCALE = /\bgray-(\d+)(?:\/\d+)?\b/g;

function findInvalid(source: string): string[] {
  return [...source.matchAll(SCALE)].map((match) => match[1]).filter((step) => !ALLOWED.has(Number(step)));
}

function sourceFiles(directory: string, root = directory): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    if (entry.isDirectory() && ["test", "tests", "__tests__", "fixtures", "__fixtures__"].includes(entry.name.toLowerCase())) return [];
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path, root);
    return isProductionSource(relative(root, path)) ? [path] : [];
  });
}

function isProductionSource(path: string): boolean {
  const normalized = path.replace(/\\/g, "/");
  return (
    /\.(?:[cm]?[jt]sx?|css)$/.test(normalized) &&
    !/(^|\/)(?:test|tests|__tests__|fixtures|__fixtures__)\//.test(normalized) &&
    !/\.(?:test|spec)\.(?:[cm]?[jt]sx?|css)$/.test(normalized)
  );
}

describe("Tailwind gray scale contract", () => {
  it("rejects undefined scale steps", () => {
    expect(findInvalid("bg-gray-850 hover:bg-gray-750/50")).toEqual(["850", "750"]);
    expect(findInvalid("bg-gray-800/50 text-gray-400")).toEqual([]);
    expect(isProductionSource("pages/ModelExplorer.tsx")).toBe(true);
    expect(isProductionSource("utils/grayScaleGuard.test.ts")).toBe(false);
    expect(isProductionSource("test/fixtures/colors.tsx")).toBe(false);
    expect(isProductionSource("components/button.spec.css")).toBe(false);
  });

  it("excludes nested fixture directories during source traversal", () => {
    const root = mkdtempSync(join(tmpdir(), "gray-scale-guard-"));
    try {
      mkdirSync(join(root, "pages", "fixtures"), { recursive: true });
      writeFileSync(join(root, "pages", "real.tsx"), "export const color = 'bg-gray-800';");
      writeFileSync(join(root, "pages", "legacy.jsx"), "export const color = 'bg-gray-850';");
      writeFileSync(join(root, "pages", "runtime.mjs"), "export const color = 'text-gray-805';");
      writeFileSync(join(root, "pages", "fixtures", "sample.tsx"), "export const color = 'bg-gray-850';");
      const files = sourceFiles(root);
      expect(files.map((path) => relative(root, path).replace(/\\/g, "/")).sort()).toEqual([
        "pages/legacy.jsx",
        "pages/real.tsx",
        "pages/runtime.mjs",
      ]);
      expect(files.flatMap((path) => findInvalid(readFileSync(path, "utf8")))).toEqual(["850", "805"]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("uses only defined gray scale steps throughout UI source", () => {
    const root = resolve(__dirname, "..");
    const offenders = sourceFiles(root)
      .flatMap((path) => {
        const invalid = findInvalid(readFileSync(path, "utf8"));
        return invalid.length ? [`${relative(root, path)}: ${invalid.join(", ")}`] : [];
      });
    expect(offenders).toEqual([]);
  });
});
