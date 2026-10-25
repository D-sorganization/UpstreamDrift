import { describe, expect, it } from "vitest";
import golden from "./fixtures/eventTableView.json";
import {
  EVENT_TABLE_COLUMNS,
  MISSING_TEXT,
  buildEventTableView,
  formatFixed,
  type AnatomicalEventTableDocument,
} from "./eventTableView";

const document = golden.document as AnatomicalEventTableDocument;

describe("eventTableView (parity with the PyQt event table, #13175)", () => {
  it("produces exactly the golden cells the desktop widget shows", () => {
    expect(buildEventTableView(document)).toEqual(golden.view);
  });

  it("uses the shared column titles", () => {
    expect([...EVENT_TABLE_COLUMNS]).toEqual(golden.view.columns);
  });

  it("formats numbers by the shared rule", () => {
    for (const [value, digits, text] of golden.format_cases) {
      expect(formatFixed(value as number, digits as number)).toBe(text);
    }
  });

  it("shows a null value as missing with its reason", () => {
    const view = buildEventTableView(document);
    const valueCol = view.columns.indexOf("Value (deg)");
    const noteCol = view.columns.indexOf("Note");
    const missing = view.rows.filter((row) => row[valueCol] === MISSING_TEXT);
    expect(missing.length).toBeGreaterThan(0);
    for (const row of missing) expect(row[noteCol]).not.toBe("");
  });

  it("rejects a document with another schema", () => {
    expect(() => buildEventTableView({ ...document, schema: "other" })).toThrow(
      /schema/,
    );
  });

  it("rejects a row whose term the document does not describe", () => {
    const rows = [{ ...document.rows[0], term_id: "not_a_term" }];
    expect(() => buildEventTableView({ ...document, rows })).toThrow(
      /not_a_term/,
    );
  });

  it("rejects a non-finite number", () => {
    expect(() => formatFixed(Number.POSITIVE_INFINITY, 2)).toThrow(/finite/);
  });
});
