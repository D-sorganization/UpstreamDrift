/**
 * Full/Custom selection over swing report rows (issue #12330).
 *
 * Mirrors the PyQt `ReportViewModel` (#12329): only enabled plots can be
 * selected, Full is every enabled plot, Custom keeps its own set, and the
 * result is always in report order.
 */

import type { SwingReportMode, SwingReportPlotRow } from '@/api/swingReport';

type Row = Pick<SwingReportPlotRow, 'plot_id' | 'enabled'>;

export function enabledPlotIds(rows: readonly Row[]): string[] {
  return rows.filter((r) => r.enabled).map((r) => r.plot_id);
}

export function selectedPlotIds(
  rows: readonly Row[],
  mode: SwingReportMode,
  custom: readonly string[],
): string[] {
  const enabled = enabledPlotIds(rows);
  if (mode === 'full') return enabled;
  const chosen = new Set(custom);
  return enabled.filter((id) => chosen.has(id));
}

/**
 * Add or remove `plotId` from the Custom selection.
 *
 * @throws RangeError when `plotId` is withheld or unknown (never selectable).
 */
export function toggleCustomPlot(
  rows: readonly Row[],
  custom: readonly string[],
  plotId: string,
  checked: boolean,
): string[] {
  if (!enabledPlotIds(rows).includes(plotId)) {
    throw new RangeError(`Plot ${plotId} cannot be selected`);
  }
  const next = new Set(custom);
  if (checked) next.add(plotId);
  else next.delete(plotId);
  return selectedPlotIds(rows, 'custom', [...next]);
}
