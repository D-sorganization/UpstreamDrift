/**
 * Simulation controls API — camera presets, trajectory recording, and
 * client-side trajectory export (issue #7452).
 *
 * Server-side recording export (full recordings API) is tracked separately
 * in #7451; until it lands, `downloadTrajectory` exports the in-browser
 * frame buffer that `useSimulation` accumulates.
 */

import { apiFetch } from './fetch';
import type { SimulationFrame } from './client';

/** A camera preset with its server-side view vectors. */
export interface CameraPresetInfo {
  preset: string;
  position: number[];
  target: number[];
  up: number[];
}

/** Recording state returned by POST /api/simulation/recording. */
export interface RecordingControlResult {
  recording: boolean;
  frame_count: number;
  status: string;
  export_path?: string | null;
}

/** Canonical fallback used when the enumeration endpoint is unreachable. */
export const FALLBACK_CAMERA_PRESETS: string[] = [
  'side',
  'front',
  'top',
  'follow_ball',
  'follow_club',
];

/**
 * Fetch the canonical camera preset list from the backend.
 *
 * @returns Preset descriptors in server order.
 * @throws Error on HTTP/network failure or malformed payload.
 */
export async function fetchCameraPresets(): Promise<CameraPresetInfo[]> {
  const data = await apiFetch<{ presets?: unknown }>(
    '/api/simulation/camera/presets',
  );
  if (!Array.isArray(data.presets)) {
    throw new Error('Unexpected camera presets response shape');
  }
  return data.presets as CameraPresetInfo[];
}

/**
 * Apply a camera preset on the server (mirrors the PyQt6 viewer state).
 *
 * @param preset - Preset identifier, e.g. 'side' or 'follow_ball'.
 * @returns The applied preset with its position/target/up vectors.
 */
export async function applyCameraPreset(
  preset: string,
): Promise<CameraPresetInfo> {
  if (!preset) {
    throw new Error('preset must be a non-empty string');
  }
  return apiFetch<CameraPresetInfo>('/api/simulation/camera', {
    method: 'POST',
    body: JSON.stringify({ preset }),
  });
}

/**
 * Start or stop server-side trajectory recording.
 *
 * @param action - 'start' or 'stop'.
 * @returns Recording state including frame_count (populated on stop).
 */
export async function controlRecording(
  action: 'start' | 'stop',
): Promise<RecordingControlResult> {
  return apiFetch<RecordingControlResult>('/api/simulation/recording', {
    method: 'POST',
    body: JSON.stringify({ action }),
  });
}

// ── Client-side trajectory export ───────────────────────────────────────────

/** Collect stable, ordered state columns across all frames. */
function collectStateColumns(frames: SimulationFrame[]): string[] {
  const columns = new Map<string, number>();
  for (const frame of frames) {
    for (const [key, values] of Object.entries(frame.state ?? {})) {
      const width = Array.isArray(values) ? values.length : 0;
      columns.set(key, Math.max(columns.get(key) ?? 0, width));
    }
  }
  const result: string[] = [];
  for (const [key, width] of columns) {
    for (let i = 0; i < width; i++) {
      result.push(`${key}_${i}`);
    }
  }
  return result;
}

/**
 * Serialize accumulated simulation frames to CSV.
 *
 * Columns: frame, time, then each state vector flattened as `<key>_<i>`.
 *
 * @param frames - Frame buffer accumulated by useSimulation.
 * @returns CSV text including a header row.
 * @throws Error when frames is empty.
 */
export function buildTrajectoryCsv(frames: SimulationFrame[]): string {
  if (frames.length === 0) {
    throw new Error('Cannot export an empty trajectory');
  }
  const stateColumns = collectStateColumns(frames);
  const header = ['frame', 'time', ...stateColumns];
  const rows = frames.map((frame) => {
    const cells: (number | string)[] = [frame.frame, frame.time];
    for (const column of stateColumns) {
      const splitAt = column.lastIndexOf('_');
      const key = column.slice(0, splitAt);
      const index = Number(column.slice(splitAt + 1));
      const value = frame.state?.[key]?.[index];
      cells.push(value ?? '');
    }
    return cells.join(',');
  });
  return [header.join(','), ...rows].join('\n');
}

/**
 * Serialize accumulated simulation frames to pretty-printed JSON.
 *
 * @param frames - Frame buffer accumulated by useSimulation.
 * @returns JSON text with frame_count and frames keys.
 * @throws Error when frames is empty.
 */
export function buildTrajectoryJson(frames: SimulationFrame[]): string {
  if (frames.length === 0) {
    throw new Error('Cannot export an empty trajectory');
  }
  return JSON.stringify(
    {
      frame_count: frames.length,
      exported_at: new Date().toISOString(),
      frames,
    },
    null,
    2,
  );
}

/** Trigger a browser download of `content` as `filename`. */
function triggerDownload(content: string, filename: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/**
 * Download the accumulated frame buffer as a file.
 *
 * @param frames - Frame buffer accumulated by useSimulation.
 * @param format - 'csv' or 'json'.
 * @throws Error when frames is empty.
 */
export function downloadTrajectory(
  frames: SimulationFrame[],
  format: 'csv' | 'json',
): void {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  if (format === 'csv') {
    triggerDownload(
      buildTrajectoryCsv(frames),
      `trajectory-${stamp}.csv`,
      'text/csv',
    );
  } else {
    triggerDownload(
      buildTrajectoryJson(frames),
      `trajectory-${stamp}.json`,
      'application/json',
    );
  }
}
