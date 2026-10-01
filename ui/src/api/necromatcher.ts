import { apiFetch } from './fetch';
import { getApiBase } from './backend';

const root = '/api/v1/necromatcher';
export interface HistoricalPlayer { subject_id: string; display_name: string; metadata: Record<string, unknown> }
export interface HistoricalSwing { session_id: string; subject_id: string; name: string; metadata: Record<string, unknown> }
export interface HistoricalAsset {
  dataset_id: string; session_id: string; kind: 'image_capture' | 'native_model' | 'torque_profile';
  metadata: { qualification: string; frame_count?: number; engine?: string; dofs?: string[]; model_id?: string; hash?: string };
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
export const captureFrameImageUrl = (capture: string, frame: number) => `${getApiBase()}${root}/captures/${encodeURIComponent(capture)}/frames/${frame}/image`;
export const swingExportUrl = (swing: string) => `${getApiBase()}${root}/swings/${encodeURIComponent(swing)}/export`;
export const createPlayer = (id: string, name: string) => apiFetch<HistoricalPlayer>(`${root}/players`, { method: 'POST', body: JSON.stringify({ id, name }) });
export const createSwing = (id: string, player_id: string, name: string) => apiFetch<HistoricalSwing>(`${root}/swings`, { method: 'POST', body: JSON.stringify({ id, player_id, name }) });
export function importAsset(swing: string, kind: 'captures' | 'models' | 'profiles', payload: { id: string; source_path: string; engine?: string; dofs?: string[] }) {
  return apiFetch<HistoricalAsset>(`${root}/swings/${encodeURIComponent(swing)}/${kind}`, { method: 'POST', body: JSON.stringify(payload), timeoutMs: 300_000 });
}
