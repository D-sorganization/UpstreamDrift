/**
 * Swing report client (issue #12330, web parity with the PyQt dialog #12329).
 *
 * POST /api/swing-report/view validates a `swing-report-v1` document on the
 * server and returns the PyQt `ReportViewModel` rows. Rows are text only, so
 * no numeric value of a withheld quantity ever reaches the browser.
 * POST /api/swing-report/export returns the S4 package as a zip.
 */

import { apiFetch, apiFetchBlob } from './fetch';
import type { SwingReportPlotRow, SwingReportViewResponse } from './generated/types';

export type { SwingReportPlotRow, SwingReportViewResponse };

export type SwingReportMode = 'full' | 'custom';

/** A `swing-report-v1` document; the server validates the full contract. */
export type SwingReportDocument = Record<string, unknown>;

/** Packages render PNG plots server-side, which can outlast the default timeout. */
const EXPORT_TIMEOUT_MS = 120_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Precondition for every request: a JSON object carrying a string
 * `schema_version` and `report_id`. The server checks the rest of the contract.
 */
export function assertSwingReportDocument(doc: unknown): asserts doc is SwingReportDocument {
  if (
    !isRecord(doc) ||
    typeof doc.schema_version !== 'string' ||
    typeof doc.report_id !== 'string'
  ) {
    throw new TypeError(
      'Not a swing report document: expected an object with schema_version and report_id',
    );
  }
}

function parseRow(raw: unknown, index: number): SwingReportPlotRow {
  if (!isRecord(raw)) throw new TypeError(`rows[${index}] is not an object`);
  for (const key of ['plot_id', 'title', 'kind', 'status_label', 'detail'] as const) {
    if (typeof raw[key] !== 'string') throw new TypeError(`rows[${index}].${key} must be a string`);
  }
  if (typeof raw.enabled !== 'boolean') {
    throw new TypeError(`rows[${index}].enabled must be a boolean`);
  }
  const reason = raw.unavailable_reason;
  if (reason !== null && reason !== undefined && typeof reason !== 'string') {
    throw new TypeError(`rows[${index}].unavailable_reason must be a string or null`);
  }
  return raw as unknown as SwingReportPlotRow;
}

/**
 * Validate the view response at the boundary.
 *
 * Postcondition: every row is well formed and every default Custom id is an
 * enabled row, so the dialog can never pre-select a withheld plot.
 */
export function parseSwingReportView(raw: unknown): SwingReportViewResponse {
  if (!isRecord(raw) || !Array.isArray(raw.rows) || !Array.isArray(raw.default_custom_ids)) {
    throw new TypeError('Malformed swing report view');
  }
  if (typeof raw.report_id !== 'string' || typeof raw.package_filename !== 'string') {
    throw new TypeError('Swing report view needs report_id and package_filename');
  }
  const rows = raw.rows.map(parseRow);
  const enabled = new Set(rows.filter((r) => r.enabled).map((r) => r.plot_id));
  const bad = raw.default_custom_ids.filter((id) => typeof id !== 'string' || !enabled.has(id));
  if (bad.length > 0) {
    throw new TypeError(`Default selection holds unselectable plots: ${bad.join(', ')}`);
  }
  return raw as unknown as SwingReportViewResponse;
}

export async function fetchSwingReportView(
  report: unknown,
  signal?: AbortSignal,
): Promise<SwingReportViewResponse> {
  assertSwingReportDocument(report);
  const raw = await apiFetch<unknown>('/api/swing-report/view', {
    method: 'POST',
    body: JSON.stringify({ report }),
    signal,
  });
  return parseSwingReportView(raw);
}

/** Export the Full report or a non-empty Custom selection as a zip package. */
export async function exportSwingReport(
  report: unknown,
  mode: SwingReportMode,
  plotIds: readonly string[],
): Promise<Blob> {
  assertSwingReportDocument(report);
  if (mode === 'custom' && plotIds.length === 0) {
    throw new RangeError('A custom export needs at least one plot');
  }
  return apiFetchBlob('/api/swing-report/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ report, mode, plot_ids: plotIds }),
    timeoutMs: EXPORT_TIMEOUT_MS,
  });
}
