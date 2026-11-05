/**
 * The web build calls no external paid API (issue #14450).
 *
 * Every request goes to the local API through the dev proxy or the same
 * origin. This scans the shipped sources for paid-API hosts and key names.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const UI = resolve(__dirname, "../..");
const PAID =
  /api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|api\.mistral\.ai|api\.cohere\.ai|maps\.googleapis\.com|api\.mapbox\.com|\b(OPENAI|ANTHROPIC|GEMINI|GOOGLE|MISTRAL|COHERE|MAPBOX)_API_KEY\b/;

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory())
      return name === "test" ? [] : sources(path);
    return /\.(ts|tsx|js|mjs|html)$/.test(name) && !/\.test\.tsx?$/.test(name)
      ? [path]
      : [];
  });
}

describe("no external paid APIs (#14450)", () => {
  it("ships no paid-API host or key name", () => {
    const files = [
      ...sources(join(UI, "src")),
      join(UI, "index.html"),
      join(UI, "vite.config.ts"),
    ];
    const hits = files.filter((file) => PAID.test(readFileSync(file, "utf8")));
    expect(hits).toEqual([]);
  });
});
