/**
 * Anatomical event table (HRG-E4, issue #13175).
 *
 * Web parity with the canonical PyQt `AnatomicalEventTableWidget`: it renders
 * the event-table export document through `buildEventTableView`, the mirror of
 * the Python view, so both surfaces show the same text. A missing value shows
 * its reason as the cell title as well as in the Note column. The document is
 * fetched from `POST /api/biomechanics/anatomical-event-table` by
 * `AnatomicalEventTableFromApi`; this component stays presentational.
 */

import {
  EVENT_TABLE_COLUMNS,
  MISSING_TEXT,
  buildEventTableView,
  type AnatomicalEventTableDocument,
  type EventTableView,
} from "./eventTableView";

const VALUE_COLUMN = EVENT_TABLE_COLUMNS.indexOf("Value (deg)");
const NOTE_COLUMN = EVENT_TABLE_COLUMNS.indexOf("Note");

export interface AnatomicalEventTableProps {
  document: AnatomicalEventTableDocument;
}

function toView(
  document: AnatomicalEventTableDocument,
): EventTableView | Error {
  try {
    return buildEventTableView(document);
  } catch (error) {
    return error instanceof Error ? error : new Error(String(error));
  }
}

export function AnatomicalEventTable({ document }: AnatomicalEventTableProps) {
  const view = toView(document);
  if (view instanceof Error) {
    return (
      <p role="alert" className="text-sm text-red-400">
        {view.message}
      </p>
    );
  }
  return (
    <div className="space-y-2">
      <p
        data-testid="anatomical-event-table-caption"
        className="text-xs text-gray-400"
      >
        {view.caption}
      </p>
      <table data-testid="anatomical-event-table" className="w-full text-xs">
        <thead>
          <tr>
            {view.columns.map((column) => (
              <th key={column} scope="col" className="px-2 py-1 text-left">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {view.rows.map((row) => (
            <tr key={`${row[0]}-${row[3]}`}>
              {row.map((text, c) => (
                <td
                  key={view.columns[c]}
                  className="px-2 py-1"
                  title={
                    c === VALUE_COLUMN && text === MISSING_TEXT
                      ? row[NOTE_COLUMN]
                      : undefined
                  }
                >
                  {text}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
