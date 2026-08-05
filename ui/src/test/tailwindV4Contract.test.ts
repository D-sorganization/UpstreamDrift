import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("Tailwind v4 stylesheet contract", () => {
  it("imports the framework and explicitly loads the legacy theme config", () => {
    const stylesheet = readFileSync(
      resolve(process.cwd(), "src/index.css"),
      "utf8",
    );

    expect(stylesheet).toMatch(/^@import "tailwindcss";/);
    expect(stylesheet).toContain('@config "../tailwind.config.js";');
    expect(stylesheet).not.toContain("@tailwind utilities;");
  });
});
