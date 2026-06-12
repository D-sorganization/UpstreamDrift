/**
 * Recordings API client (issue #7451).
 *
 * Export/recording parity with the desktop app: lists persisted session
 * recordings, persists the active session recorder, deletes recordings,
 * and exposes per-format download URLs backed by the desktop serializers.
 */

import { getApiBase } from './backend';
import { apiFetch } from './fetch';

export interface RecordingMeta {
  id: string;
  engine: string | null;
  model: string | null;
  duration: number | null;
  frames: number;
  created: string;
}

export interface ExportFormatInfo {
  name: string;
  extension: string;
  available: boolean;
  description: string;
}

/** Map of format key (json/csv/mat/hdf5/c3d) to availability info. */
export type ExportFormats = Record<string, ExportFormatInfo>;

export async function fetchExportFormats(): Promise<ExportFormats> {
  const data = await apiFetch<{ formats: ExportFormats }>('/api/export/formats');
  return data.formats;
}

export async function listRecordings(): Promise<RecordingMeta[]> {
  const data = await apiFetch<{ recordings: RecordingMeta[] }>('/api/recordings');
  return data.recordings;
}

/** Persist the active session recorder to disk (409 if no session). */
export async function saveRecording(): Promise<RecordingMeta> {
  return apiFetch<RecordingMeta>('/api/recordings', { method: 'POST' });
}

export async function deleteRecording(id: string): Promise<void> {
  await apiFetch<{ deleted: string }>(`/api/recordings/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
}

/** Build the streaming download URL for a recording export. */
export function recordingExportUrl(id: string, format: string): string {
  return `${getApiBase()}/api/recordings/${encodeURIComponent(id)}/export?format=${encodeURIComponent(format)}`;
}
