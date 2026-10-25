/**
 * Display view of the anatomical event table (HRG-E4, issue #13175).
 *
 * Mirror of `src/shared/python/anatomical_angles/event_table_view.py`, the
 * view the PyQt `AnatomicalEventTableWidget` renders. Both read the strict-JSON
 * export document (`event_table.to_json_dict`, also the response of
 * `POST /biomechanics/anatomical-event-table`). The golden fixture
 * `fixtures/eventTableView.json` is checked by both test suites, so the two
 * surfaces show identical text for the same bundle. Change both together.
 */

export const EVENT_TABLE_SCHEMA_ID = "upstreamdrift.anatomical_event_table";
export const EVENT_TABLE_COLUMNS = [
  "Event",
  "Frame",
  "Time (s)",
  "Term",
  "Side",
  "Value (deg)",
  "Observability",
  "Note",
] as const;
export const MISSING_TEXT = "missing";
const VALUE_DIGITS = 2;
const TIME_DIGITS = 3;

export interface AnatomicalEventRow {
  event: string;
  frame: number;
  time_s: number;
  term_id: string;
  role: string;
  side: string;
  value_deg: number | null;
  invalid_reason: string | null;
  observability: string;
  receipt_path: string;
  calibration: string[];
}

export interface AnatomicalEventTableDocument {
  schema: string;
  schema_version: number;
  units: { angle: string; time: string };
  handedness: string;
  event_source: string;
  reference: string;
  events: Record<string, { frame: number; time_s: number }>;
  terms: Record<string, Record<string, string>>;
  rows: AnatomicalEventRow[];
}

export interface EventTableView {
  caption: string;
  columns: string[];
  rows: string[][];
}

/**
 * `value` with `digits` decimals: `toFixed` rounding (half away from zero on
 * the exact binary value), with no sign on a result that is all zeros.
 */
export function formatFixed(value: number, digits: number): string {
  if (!Number.isFinite(value))
    throw new Error(`value must be finite, got ${value}`);
  if (digits < 0) throw new Error(`digits must be >= 0, got ${digits}`);
  const text = value.toFixed(digits);
  return /^-[0.]+$/.test(text) ? text.slice(1) : text;
}

function caption(doc: AnatomicalEventTableDocument): string {
  return (
    `Events from ${doc.event_source}; ${doc.handedness}-handed; ` +
    `angles in ${doc.units.angle}, time in ${doc.units.time}; ` +
    `reference: ${doc.reference}`
  );
}

function viewRow(
  row: AnatomicalEventRow,
  terms: AnatomicalEventTableDocument["terms"],
): string[] {
  if (!(row.term_id in terms)) {
    throw new Error(`row term '${row.term_id}' is not in the document terms`);
  }
  return [
    row.event,
    String(row.frame),
    formatFixed(row.time_s, TIME_DIGITS),
    row.term_id,
    `${row.role} (${row.side})`,
    row.value_deg === null
      ? MISSING_TEXT
      : formatFixed(row.value_deg, VALUE_DIGITS),
    row.observability,
    row.invalid_reason ?? "",
  ];
}

/** Text rows of an exported event-table document, in export row order. */
export function buildEventTableView(
  doc: AnatomicalEventTableDocument,
): EventTableView {
  if (doc?.schema !== EVENT_TABLE_SCHEMA_ID) {
    throw new Error(`document schema must be '${EVENT_TABLE_SCHEMA_ID}'`);
  }
  return {
    caption: caption(doc),
    columns: [...EVENT_TABLE_COLUMNS],
    rows: doc.rows.map((row) => viewRow(row, doc.terms)),
  };
}
