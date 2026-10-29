/**
 * Anatomical event-table client (HRG-E4, #13175).
 *
 * POST /api/biomechanics/anatomical-event-table computes the wrist and
 * forearm angles at the swing events from the posted calibrated frames and
 * returns the strict-JSON export document (`event_table.to_json_dict`), the
 * same document the desktop widget renders. This client always asks for the
 * `json` format; the CSV body is a download concern, not a table input.
 */

import { apiFetch } from './fetch';
import type { AnatomicalEventTableDocument } from '@/components/analysis/anatomicalEventTable/eventTableView';

export type { AnatomicalEventTableDocument };

export const ANATOMICAL_EVENT_TABLE_PATH = '/api/biomechanics/anatomical-event-table';

/** World pose series of one calibrated segment (mirrors the API `SegmentFrames`). */
export interface SegmentFrames {
  /** Rotations, shape (N, 3, 3). */
  rotation_world: unknown[];
  /** Origins, shape (N, 3). */
  origin_world: unknown[];
  calibration_id: string;
}

/** Segment frames of one side (mirrors the API `ArmFrames`). */
export interface ArmFrames {
  humerus: SegmentFrames;
  forearm: SegmentFrames;
  hand: SegmentFrames;
  /** Defaults to `forearm` on the server when omitted. */
  radius?: SegmentFrames | null;
}

/** Request body (mirrors the API `AnatomicalEventTableRequest`, minus `format`). */
export interface AnatomicalEventTableRequest {
  time_s: number[];
  /** Event name to frame index. */
  events: Record<string, number>;
  handedness: string;
  event_source: string;
  receipt_path: string;
  /** Arm frames keyed by side. */
  arms: Record<string, ArmFrames>;
}

export function fetchAnatomicalEventTable(
  request: AnatomicalEventTableRequest,
  signal?: AbortSignal,
): Promise<AnatomicalEventTableDocument> {
  return apiFetch<AnatomicalEventTableDocument>(ANATOMICAL_EVENT_TABLE_PATH, {
    method: 'POST',
    body: JSON.stringify({ ...request, format: 'json' }),
    signal,
  });
}
