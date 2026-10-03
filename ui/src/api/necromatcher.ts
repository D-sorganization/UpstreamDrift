import { apiFetch, apiFetchForm } from './fetch';
import { getApiBase } from './backend';
import type { AssetRequest, IdentityRequest, ModelRequest, SwingRequest } from './generated/types';

const root = '/api/v1/necromatcher';
export interface SourceFitScope {
  schema: 'necromatcher/source-fit-scope/1';
  capture_id: string; capture_hash: string; source_clock_sha256: string;
  first_frame: number; end_exclusive_frame: number; purpose: 'both_hands_on_club';
  review: {
    artifact: {artifact_id: string; path: string; hash: string; schema: string; kind: string};
    receipt_bytes: number; first_identity: Record<string, unknown>;
    excluded_identity: Record<string, unknown> | null;
    reason: string; uncertainty_policy: string; contact_calibrated: false;
  };
}
export interface SourceFitScopeBinding {
  frame_indices: number[]; first_pts: [number, number]; last_pts: [number, number];
  source_clock_sha256: string;
}
export interface FitConstraintRecipe {
  ground: {normal: number[]; height_m: number};
  position_weight: number; rotation_weight: number; ground_weight: number;
  position_scale_m: number; rotation_scale_rad: number; ground_scale_m: number;
  pinned_spheres?: string[];
}
export interface ContactPinPhaseRecipe {
  start_pts: [number, number]; end_pts: [number, number];
  pinned_spheres: string[]; review_frame_sha256: string[];
}
export interface ContactPinScheduleRecipe {
  capture_id: string; capture_sha256: string;
  status: 'authored_contact_hypothesis'; phases: ContactPinPhaseRecipe[];
}
export interface ScheduledFitConstraintRecipe {
  base: FitConstraintRecipe; schedule: ContactPinScheduleRecipe;
}
export interface ImageFitRecipe {
  max_iterations: number; prior_weight: number; smoothness_weight: number; closure_weight: number;
  constraint_options?: FitConstraintRecipe | ScheduledFitConstraintRecipe | null;
  interior_fractions?: number[];
  coordinate_bounds?: [string, number, number][];
  initialization_policy?: 'strict' | 'authored_range_project_zero_slopes';
}
export interface RefitOptions {
  frame_indices: number[]; knot_count: number; coordinate_scales: number[];
  unknown_visibility_weight: number; budget_wall_s: number; config: ImageFitRecipe;
  operation?: 'fit' | 'author_initialization';
  initialization_source?: 'sampled_parent' | 'preserved_spline';
  shaft_evidence?: Record<string, unknown>;
  source_scope?: SourceFitScope;
}
export interface RefitPlan {
  source_fit_id: string; frame_indices: number[]; coordinate_order: string[]; coordinate_units: string[];
  recorded_options: RefitOptions | null;
  baseline_config?: ImageFitRecipe;
  source_scope?: SourceFitScope | null;
  source_scope_binding?: SourceFitScopeBinding | null;
  preserved_spline?: {available: boolean; knot_count: number | null; source_interval: [number, number] | null; reason: string};
}
export interface ResearchRun {
  run_id: string; source_fit_id: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  acceptance: 'partial' | 'interrupted' | 'accepted' | 'rejected';
  blockers: string[]; message: string; fraction: number | null;
  control_available?: boolean;
  source_fit_scope?: SourceFitScope | null;
  source_fit_scope_binding?: SourceFitScopeBinding | null;
}
export interface RefitRun extends ResearchRun {new_fit_id: string}
export interface VideoExportRun extends ResearchRun {
  qualification: 'monocular_research_hypothesis'; download_available: boolean;
  execution_started: boolean; execution_verified: boolean;
  artifact_state?: 'verified_stat_baseline' | 'changed_or_unverified';
  producer_source_commit?: string | null;
  shape_overlay?: {opacity: number};
}
export type VideoOverlayOptions = {shaft_evidence?: Record<string, unknown>; shape_overlay?: {opacity: number}};
export const submitVideoExport = (fit: string, options?: VideoOverlayOptions) => apiFetch<VideoExportRun>(`${root}/fits/${encodeURIComponent(fit)}/video-exports`, {method: 'POST', ...(options ? {body: JSON.stringify(options)} : {})});
export const fetchVideoExport = (run: string) => apiFetch<VideoExportRun>(`${root}/video-exports/${encodeURIComponent(run)}`);
export const cancelVideoExport = (run: string) => apiFetch<VideoExportRun>(`${root}/video-exports/${encodeURIComponent(run)}/cancel`, {method: 'POST'});
export const videoExportDownloadUrl = (run: string) => `${getApiBase()}${root}/video-exports/${encodeURIComponent(run)}/download`;
export const fetchRefitPlan = (fit: string) => apiFetch<RefitPlan>(`${root}/fits/${encodeURIComponent(fit)}/refit-plan`);
export function registerReviewedWindow(fit: string, file: File) {
  const body = new FormData(); body.append('file', file);
  return apiFetchForm<SourceFitScope>(`${root}/fits/${encodeURIComponent(fit)}/source-scope-reviews`, body, {timeoutMs: 300_000});
}
export const submitRefit = (fit: string, payload: RefitOptions & {new_fit_id: string}) => apiFetch<RefitRun>(`${root}/fits/${encodeURIComponent(fit)}/refits`, {method: 'POST', body: JSON.stringify(payload)});
export const fetchRefit = (run: string) => apiFetch<RefitRun>(`${root}/refits/${encodeURIComponent(run)}`);
export const cancelRefit = (run: string) => apiFetch<RefitRun>(`${root}/refits/${encodeURIComponent(run)}/cancel`, {method: 'POST'});
export interface HistoricalPlayer { subject_id: string; display_name: string; metadata: Record<string, unknown> }
export interface HistoricalSwing { session_id: string; subject_id: string; name: string; metadata: Record<string, unknown> }
export interface HistoricalAsset {
  dataset_id: string; session_id: string; kind: 'image_capture' | 'native_model' | 'torque_profile' | 'kinematic_fit';
  metadata: { qualification: string; frame_count?: number; engine?: string; dofs?: string[]; model_id?: string; capture_id?: string; hash?: string };
}
export interface CaptureFrame {
  capture_id: string; frame_index: number; frame_count: number; image_width: number; image_height: number;
  frame: { pts_ticks: number; timebase_numerator: number; timebase_denominator: number; physical_time_s: number | null };
  observation: { status: string; landmarks: Record<string, { x: number; y: number; visibility: number | null }> };
}
export const fetchPlayers = () => apiFetch<{ players: HistoricalPlayer[] }>(`${root}/players`);
export const fetchSwings = (player: string) => apiFetch<{ swings: HistoricalSwing[] }>(`${root}/swings?player_id=${encodeURIComponent(player)}`);
export const fetchAssets = (swing: string) => apiFetch<{ assets: HistoricalAsset[] }>(`${root}/swings/${encodeURIComponent(swing)}/assets`);
export const fetchCaptureFrame = (capture: string, frame: number) => apiFetch<CaptureFrame>(`${root}/captures/${encodeURIComponent(capture)}/frames/${frame}`);
export interface FitProjection {
  fit_id: string; capture_id: string; frame_index: number; frame: CaptureFrame['frame'];
  points: CaptureFrame['observation']['landmarks']; coordinates: 'image_pixels';
  qualification: 'monocular_research_hypothesis'; camera_qualified: false; physical_time_qualified: false;
}
export const fetchFitProjection = (fit: string, frame: number) => apiFetch<FitProjection>(`${root}/fits/${encodeURIComponent(fit)}/frames/${frame}/projection`);
export const captureFrameImageUrl = (capture: string, frame: number) => `${getApiBase()}${root}/captures/${encodeURIComponent(capture)}/frames/${frame}/image`;
export const swingExportUrl = (swing: string) => `${getApiBase()}${root}/swings/${encodeURIComponent(swing)}/export`;
export const createPlayer = (id: string, name: string) => apiFetch<HistoricalPlayer>(`${root}/players`, { method: 'POST', body: JSON.stringify({ id, name } satisfies IdentityRequest) });
export const createSwing = (id: string, player_id: string, name: string) => apiFetch<HistoricalSwing>(`${root}/swings`, { method: 'POST', body: JSON.stringify({ id, player_id, name } satisfies SwingRequest) });
export function importAsset(swing: string, kind: 'captures' | 'models' | 'profiles' | 'fits', payload: AssetRequest & Partial<Pick<ModelRequest, 'engine' | 'dofs'>>) {
  return apiFetch<HistoricalAsset>(`${root}/swings/${encodeURIComponent(swing)}/${kind}`, { method: 'POST', body: JSON.stringify(payload), timeoutMs: 300_000 });
}
