import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

interface PackageLock {
  packages: Record<string, { version?: string }>;
}

const packageLock = JSON.parse(
  readFileSync(resolve(process.cwd(), "package-lock.json"), "utf8"),
) as PackageLock;

const isAtLeastVersion = (actual: string | undefined, minimum: string): boolean => {
  if (actual === undefined) {
    return false;
  }

  const actualParts = actual.split(".").map(Number);
  const minimumParts = minimum.split(".").map(Number);
  if (
    actualParts.length !== 3 ||
    minimumParts.length !== 3 ||
    actualParts[0] !== minimumParts[0] ||
    [...actualParts, ...minimumParts].some((part) => !Number.isInteger(part))
  ) {
    return false;
  }

  for (let index = 0; index < actualParts.length; index += 1) {
    if (actualParts[index] !== minimumParts[index]) {
      return actualParts[index] > minimumParts[index];
    }
  }

  return true;
};

describe("UI dependency security lockfile contract (#11184)", () => {
  it("locks brace-expansion to the patched compatible 5.x release", () => {
    expect(
      isAtLeastVersion(
        packageLock.packages["node_modules/brace-expansion"]?.version,
        "5.0.12",
      ),
    ).toBe(true);
  });

  it("locks undici to the patched compatible 8.x release", () => {
    expect(
      isAtLeastVersion(packageLock.packages["node_modules/undici"]?.version, "8.10.2"),
    ).toBe(true);
  });
});
