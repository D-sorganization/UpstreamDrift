/**
 * Swing report Full/Custom dialog (issue #12330).
 *
 * Web parity with the canonical PyQt `SwingReportDialog` (#12329). It shows
 * the server-validated `ReportViewModel` rows (text only): withheld plots are
 * disabled with a "Not Validated" label and their reason, advertised plots
 * show their uncertainty, and Export downloads the S4 package for the Full
 * report or the Custom selection. It imports no engine code.
 */

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui';
import {
  exportSwingReport,
  fetchSwingReportView,
  type SwingReportMode,
  type SwingReportPlotRow,
  type SwingReportViewResponse,
} from '@/api/swingReport';
import { selectedPlotIds, toggleCustomPlot } from './swingReportSelection';

export type BlobDownload = (blob: Blob, filename: string) => void;

function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export interface SwingReportDialogProps {
  /** A `swing-report-v1` document; the server validates it before display. */
  report: unknown;
  /** Saves the exported package; injectable like the PyQt directory chooser. */
  download?: BlobDownload;
}

interface PlotRowProps {
  row: SwingReportPlotRow;
  checked: boolean;
  locked: boolean;
  onToggle: (plotId: string, checked: boolean) => void;
}

function PlotRowItem({ row, checked, locked, onToggle }: PlotRowProps) {
  const id = `swing-report-plot-${row.plot_id}`;
  return (
    <li className="py-1">
      <label htmlFor={id} className="flex items-center gap-2 text-sm text-gray-100" title={row.detail}>
        <input
          id={id}
          data-testid={id}
          type="checkbox"
          checked={checked}
          disabled={!row.enabled || locked}
          onChange={(e) => onToggle(row.plot_id, e.target.checked)}
        />
        {`${row.title} [${row.status_label}]`}
      </label>
      {row.detail && (
        <p data-testid={`swing-report-detail-${row.plot_id}`} className="ml-6 text-xs text-gray-400">
          {row.detail}
        </p>
      )}
    </li>
  );
}

export function SwingReportDialog({ report, download = saveBlob }: SwingReportDialogProps) {
  const [view, setView] = useState<SwingReportViewResponse | null>(null);
  const [mode, setMode] = useState<SwingReportMode>('full');
  const [custom, setCustom] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setView(null);
    setError(null);
    fetchSwingReportView(report, controller.signal)
      .then((v) => {
        setView(v);
        setCustom(v.default_custom_ids);
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
      });
    return () => controller.abort();
  }, [report]);

  if (!view) {
    return error ? <p role="alert" className="text-sm text-red-400">{error}</p> : <p>Loading report...</p>;
  }
  const selected = selectedPlotIds(view.rows, mode, custom);
  const onToggle = (plotId: string, checked: boolean) =>
    setCustom((prev) => toggleCustomPlot(view.rows, prev, plotId, checked));

  const onExport = async () => {
    setBusy(true);
    setError(null);
    try {
      download(await exportSwingReport(report, mode, selected), view.package_filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-label="Swing Report" className="flex flex-col gap-3">
      <h2 className="text-lg font-semibold text-gray-100">Swing Report</h2>
      <fieldset className="flex gap-4 text-sm text-gray-100">
        {(['full', 'custom'] as const).map((m) => (
          <label key={m} className="flex items-center gap-1">
            <input type="radio" name="swing-report-mode" checked={mode === m} onChange={() => setMode(m)} />
            {m === 'full' ? 'Full Report' : 'Custom Selection'}
          </label>
        ))}
      </fieldset>
      <ul>
        {view.rows.map((row) => (
          <PlotRowItem
            key={row.plot_id}
            row={row}
            checked={selected.includes(row.plot_id)}
            locked={mode === 'full'}
            onToggle={onToggle}
          />
        ))}
      </ul>
      {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
      <Button
        onClick={onExport}
        disabled={busy || selected.length === 0}
        disabledReason={selected.length === 0 ? 'Select at least one plot' : 'Export in progress'}
      >
        Export
      </Button>
    </section>
  );
}
