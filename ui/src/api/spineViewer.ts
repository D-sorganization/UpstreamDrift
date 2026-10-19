/**
 * Spine Viewer API client and pure helpers (SPN-15c, #12815).
 *
 * The server verifies the study's artifact hash and builds the view model with
 * the same Python core as the PyQt6 viewer; nothing is recomputed here.
 */

import { apiFetch } from "./fetch";

export const SPINE_QUANTITIES = [
  "compression",
  "tension",
  "ap_shear",
  "ml_shear",
] as const;

export type SpineQuantity = (typeof SPINE_QUANTITIES)[number];

export interface SpineHeatmapPayload {
  quantity: string;
  units: string;
  sign_convention: string;
  observable: string;
  distribution_source: string;
  scientific_status: string;
  status_label: string;
  time_s: number[];
  /** Rows are time samples, columns are levels (caudal to cranial). */
  values: number[][];
}

export interface SpineStudyPayload {
  study_id: string;
  receipt_status: string;
  status_reasons: string[];
  scientific_status: string;
  distribution_source: string;
  banner: string;
  levels: string[];
  quantities: string[];
  heatmaps: Record<string, SpineHeatmapPayload>;
}

export type DisplayStatus = "qualified" | "unqualified";

/** Fail-closed: only the exact string "qualified" reads as qualified. */
export function displayStatus(
  status: string | null | undefined,
): DisplayStatus {
  return status === "qualified" ? "qualified" : "unqualified";
}

/**
 * Reduce rows to at most `limit` bins, keeping the signed value of largest
 * magnitude per column in each bin (peaks are never averaged away).
 */
export function binRows(rows: number[][], limit: number): number[][] {
  if (!Number.isInteger(limit) || limit < 1) {
    throw new RangeError("limit must be a positive integer");
  }
  if (rows.length <= limit) return rows;
  const size = Math.ceil(rows.length / limit);
  const out: number[][] = [];
  for (let start = 0; start < rows.length; start += size) {
    const chunk = rows.slice(start, start + size);
    out.push(
      chunk[0].map((_, col) =>
        chunk.reduce(
          (best, row) =>
            Math.abs(row[col]) > Math.abs(best) ? row[col] : best,
          0,
        ),
      ),
    );
  }
  return out;
}

export interface SpineAssetItem {
  asset_id: string;
  status: string;
  is_fixture: boolean;
  simtk_url?: string | null;
  env_var?: string | null;
  note?: string | null;
  citation?: { text?: string; doi?: string } | null;
}

export interface SpineStudyJobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  receipt_path: string | null;
  error: string | null;
}

export interface RunSpineStudyPayload {
  study_id: string;
  asset_id?: string;
  model_id?: string;
  coupling_law_id?: string | null;
  motion_path: string;
  golf_model_path?: string;
  model_path?: string;
  pelvis?: string;
  [key: string]: unknown;
}

export interface PollOptions {
  intervalMs?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
  onStatus?: (status: SpineStudyJobStatus) => void;
}

const BASE = "/api/tools/spine-viewer";

export async function fetchSpineReceipts(): Promise<string[]> {
  const body = await apiFetch<{ receipts: string[] }>(`${BASE}/receipts`);
  return body.receipts;
}

export function fetchSpineStudy(receipt: string): Promise<SpineStudyPayload> {
  return apiFetch<SpineStudyPayload>(
    `${BASE}/study?receipt=${encodeURIComponent(receipt)}`,
  );
}

export async function fetchSpineAssets(): Promise<SpineAssetItem[]> {
  const body = await apiFetch<{ assets?: SpineAssetItem[] }>(`${BASE}/assets`);
  return Array.isArray(body?.assets) ? body.assets : [];
}

export async function submitSpineStudy(
  payload: RunSpineStudyPayload,
): Promise<SpineStudyJobStatus> {
  return apiFetch<SpineStudyJobStatus>(`${BASE}/studies`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchSpineStudyJob(
  jobId: string,
): Promise<SpineStudyJobStatus> {
  return apiFetch<SpineStudyJobStatus>(
    `${BASE}/studies/${encodeURIComponent(jobId)}`,
  );
}

export async function pollSpineStudyJob(
  jobId: string,
  options?: PollOptions,
): Promise<SpineStudyJobStatus> {
  // Spine studies run for minutes (full-body chains longer), so the default
  // poll is slow and the timeout generous; callers may tighten both.
  const intervalMs = options?.intervalMs ?? 2_000;
  const timeoutMs = options?.timeoutMs ?? 2 * 60 * 60 * 1_000;
  const start = Date.now();
  while (true) {
    if (options?.signal?.aborted) {
      throw new DOMException("The poll was aborted", "AbortError");
    }
    if (Date.now() - start > timeoutMs) {
      throw new Error(`Spine study job timed out after ${timeoutMs}ms`);
    }
    const status = await fetchSpineStudyJob(jobId);
    options?.onStatus?.(status);
    if (
      status.status === "completed" ||
      status.status === "failed" ||
      status.status === "cancelled"
    ) {
      return status;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
