import { apiFetch } from './fetch';
import { getApiBase } from './backend';
import type { AssetRequest, IdentityRequest, ModelRequest, SwingRequest } from './generated/types';

const root = '/api/v1/necromatcher';
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
