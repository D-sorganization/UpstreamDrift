import { beforeEach, describe, expect, it } from "vitest";

import golden from "./units.golden.json";
import {
  formatQuantity,
  fromDisplay,
  loadUnitSystem,
  saveUnitSystem,
  toDisplay,
  unitSuffix,
  type UnitSystem,
} from "./units";

// Golden values come from src/shared/python/ui/units.py (issue #14458), so the
// React display layer and the PyQt6 tool agree on every conversion.
describe("units golden parity with units.py", () => {
  it.each([
    ["to_display_distance_ft", toDisplay, "distance", "si", "display"],
    ["from_display_distance_ft", fromDisplay, "distance", "display", "si"],
    ["to_display_speed", toDisplay, "speed", "si", "display"],
    ["from_display_speed", fromDisplay, "speed", "display", "si"],
  ] as const)("%s", (key, convert, quantity, source, target) => {
    const cases = golden[key];
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const got = convert(c[source], quantity, c.system as UnitSystem);
      expect(got).toBeCloseTo(c[target], 12);
    }
  });

  it("matches the distance and speed suffixes", () => {
    for (const system of ["metric", "imperial"] as const) {
      expect(unitSuffix("distance", system)).toBe(
        golden.distance_suffix_ft[system],
      );
      expect(unitSuffix("speed", system)).toBe(golden.speed_suffix[system]);
    }
  });
});

describe("small lengths", () => {
  it("shows millimetres in metric and inches in imperial", () => {
    expect(formatQuantity(0.0254, "smallLength", "metric", 1)).toBe("25.4 mm");
    expect(formatQuantity(0.0254, "smallLength", "imperial", 2)).toBe(
      "1.00 in",
    );
  });

  it("round-trips through the display unit", () => {
    for (const system of ["metric", "imperial"] as const) {
      const shown = toDisplay(0.0375, "smallLength", system);
      expect(fromDisplay(shown, "smallLength", system)).toBeCloseTo(0.0375, 12);
    }
  });
});

describe("break lengths (#14466)", () => {
  it("shows centimetres in metric and inches in imperial, like format_read", () => {
    expect(formatQuantity(0.38, "breakLength", "metric", 1)).toBe("38.0 cm");
    expect(formatQuantity(0.0254, "breakLength", "imperial", 1)).toBe("1.0 in");
  });
});

describe("unit preference", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to metric like units.py", () => {
    expect(loadUnitSystem()).toBe("metric");
  });

  it("persists the selected system", () => {
    saveUnitSystem("imperial");
    expect(loadUnitSystem()).toBe("imperial");
  });

  it("ignores an unknown stored value", () => {
    localStorage.setItem("ud.unitSystem", "furlongs");
    expect(loadUnitSystem()).toBe("metric");
  });
});

describe("force and torque (#14470)", () => {
  it("shows lbf and in·lbf in imperial with units.py's factors", () => {
    const lbf = 0.45359237 * 9.80665;
    expect(formatQuantity(lbf, "force", "imperial", 3)).toBe("1.000 lbf");
    expect(formatQuantity(0.0254 * lbf, "torque", "imperial", 3)).toBe(
      "1.000 in·lbf",
    );
    expect(formatQuantity(1.2, "torque", "metric", 2)).toBe("1.20 N·m");
  });
});
