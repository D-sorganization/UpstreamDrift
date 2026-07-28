import { apiFetchBlob } from './fetch';
import { triggerBlobDownload } from './download';

/** Only the formats the backend actually implements (no xlsx/pdf). */
export type ExportFormat = 'csv' | 'json';

export const EXPORT_FORMATS: readonly ExportFormat[] = ['csv', 'json'];

export interface ExportResult {
  format: ExportFormat;
  filename: string;
  size_bytes: number;
}

/**
 * Export analysis data through the shared timeout/error-handled fetch path.
 */
export async function downloadAnalysisExport(format: ExportFormat): Promise<ExportResult> {
  const blob = await apiFetchBlob('/api/analysis/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      format,
      include_metrics: true,
      include_time_series: true,
    }),
  });
  const filename = `analysis_export.${format}`;
  triggerBlobDownload(blob, filename);
  return { format, filename, size_bytes: blob.size };
}
