/**
 * Matched-swing results API client (MS-85, #10358).
 */

import { getApiBase } from './backend';

export interface RunCapabilities {
  has_candidate_npz: boolean;
  has_animation_gif: boolean;
  has_parity_report: boolean;
  candidate_profile: string | null;
  horizon_s: number | null;
}

export interface PhysicalGate {
  name: string;
  status: string;
  measured: number | null;
  threshold: number | null;
  unit: string;
}

export interface MatchedSwingRun {
  id: string;
  engine: string;
  lane: string;
  capture: string | null;
  candidate_sha256: string | null;
  receipt_sha256: string;
  horizon_s: number | null;
  verdict: string;
  metrics: Record<string, number | null>;
  capabilities: RunCapabilities;
  reason?: string | null;
  qualification_note?: string | null;
  gates?: PhysicalGate[];
}

export interface MatchedSwingLedgerResponse {
  schema_version: string;
  total: number;
  runs: MatchedSwingRun[];
}

export interface MatchedSwingLedgerParams {
  ranked?: boolean;
  driveMode?: string;
  profile?: string;
  capture?: string;
}

export interface CandidatePreviewFrame {
  id: string;
  frame_index: number;
  frame_count: number;
  joints: Array<{
    name: string;
    position: number[];
    confidence: number;
    parent: string | null;
  }>;
}

function apiUrl(path: string): string {
  return `${getApiBase()}${path}`;
}

export async function fetchMatchedSwingLedger(
  params: MatchedSwingLedgerParams = {},
): Promise<MatchedSwingLedgerResponse> {
  // URLSearchParams, not `new URL(...)`: the browser build's API base is ''
  // (relative), which `new URL` rejects.
  const query = new URLSearchParams();
  if (params.ranked) query.set('ranked', 'true');
  if (params.driveMode) query.set('drive_mode', params.driveMode);
  if (params.profile) query.set('profile', params.profile);
  if (params.capture) query.set('capture', params.capture);
  const suffix = query.toString() ? `?${query.toString()}` : '';
  const resp = await fetch(apiUrl(`/api/v1/matched-swings${suffix}`));
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load matched-swing ledger: ${resp.status} ${text}`);
  }
  return resp.json();
}

export async function fetchMatchedSwingReceipt(runId: string): Promise<Record<string, unknown>> {
  const resp = await fetch(apiUrl(`/api/v1/matched-swings/${encodeURIComponent(runId)}`));
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load receipt: ${resp.status} ${text}`);
  }
  return resp.json();
}

export function matchedSwingAnimationUrl(runId: string): string {
  return apiUrl(`/api/v1/matched-swings/${encodeURIComponent(runId)}/animation.gif`);
}

export interface MatchedSwingAnimationInfo {
  schema_version: string;
  frame_count: number;
  durations_ms: number[];
  width: number;
  height: number;
}

export async function fetchMatchedSwingAnimationInfo(
  runId: string,
): Promise<MatchedSwingAnimationInfo> {
  const resp = await fetch(
    apiUrl(`/api/v1/matched-swings/${encodeURIComponent(runId)}/animation/frames`),
  );
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load animation info: ${resp.status} ${text}`);
  }
  return resp.json();
}

export function matchedSwingAnimationFrameUrl(runId: string, index: number): string {
  return apiUrl(
    `/api/v1/matched-swings/${encodeURIComponent(runId)}/animation/frames/${index}`,
  );
}

export function matchedSwingReportUrl(runId: string): string {
  return apiUrl(`/api/v1/matched-swings/${encodeURIComponent(runId)}/report`);
}

export async function fetchCandidatePreviewFrame(
  runId: string,
  frameIndex: number,
): Promise<CandidatePreviewFrame> {
  const url = new URL(apiUrl(`/api/v1/matched-swings/${encodeURIComponent(runId)}/candidate`));
  url.searchParams.set('preview_frame', String(frameIndex));
  const resp = await fetch(url.toString());
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load candidate preview: ${resp.status} ${text}`);
  }
  return resp.json();
}

export async function fetchParityReport(runId: string): Promise<Record<string, unknown>> {
  const resp = await fetch(apiUrl(`/api/v1/matched-swings/${encodeURIComponent(runId)}/parity`));
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to load parity report: ${resp.status} ${text}`);
  }
  return resp.json();
}

export function verdictBadgeClass(verdict: string): string {
  const normalized = verdict.toUpperCase();
  if (normalized === 'PASSED' || normalized === 'ACCEPTED') {
    return 'bg-emerald-700 text-emerald-50';
  }
  if (normalized === 'REJECTED' || normalized === 'FAILED') {
    return 'bg-red-800 text-red-50';
  }
  if (normalized === 'UNVERIFIED' || normalized === 'UNCLASSIFIED') {
    return 'bg-amber-700 text-amber-50';
  }
  return 'bg-gray-700 text-gray-100';
}

export function formatMetric(value: number | null | undefined, unit: 'mm' | 'deg' = 'mm'): string {
  if (value == null || Number.isNaN(value)) {
    return '—';
  }
  if (unit === 'mm') {
    return `${(value * 1000).toFixed(2)} mm`;
  }
  return `${((value * 180) / Math.PI).toFixed(2)}°`;
}
