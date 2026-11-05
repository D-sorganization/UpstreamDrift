import { describe, expect, it } from "vitest";

import golden from "./greenTypes.golden.json";
import { greenFromType, greenLabel } from "./greenTypes";

describe("green types pinned to the shared core (#14464)", () => {
  it.each(golden.types)("labels $key as the desktop does", ({ key, label }) => {
    expect(greenLabel(key)).toBe(label);
  });

  it("sends no green spec for the cross slope", () => {
    expect(greenFromType(null, golden.presets)).toBeNull();
  });

  it.each(Object.entries(golden.presets))(
    "sends the %s preset by name with its core shape",
    (key, spec) => {
      expect(greenFromType(key, golden.presets)).toEqual({
        ...spec,
        preset: key,
      });
    },
  );
});
