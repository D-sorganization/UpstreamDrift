import { beforeEach, expect, it, vi } from 'vitest';
import { submitVideoExport, fetchVideoExport, cancelVideoExport, videoExportDownloadUrl } from './necromatcher';
const request = vi.hoisted(() => vi.fn());
vi.mock('./fetch', () => ({apiFetch: request}));
vi.mock('./backend', () => ({getApiBase: () => 'http://backend.test'}));
beforeEach(() => {request.mockReset(); request.mockResolvedValue({});});
it('submits a bodyless export for an encoded fit identity', async () => {
  await submitVideoExport('fit/one');
  expect(request).toHaveBeenCalledExactlyOnceWith('/api/v1/necromatcher/fits/fit%2Fone/video-exports', {method:'POST'});
});
it('polls and cancels only an encoded run identity', async () => {
  await fetchVideoExport('run/one'); await cancelVideoExport('run/one');
  expect(request).toHaveBeenNthCalledWith(1, '/api/v1/necromatcher/video-exports/run%2Fone');
  expect(request).toHaveBeenNthCalledWith(2, '/api/v1/necromatcher/video-exports/run%2Fone/cancel', {method:'POST'});
});
it('uses the canonical backend base for export downloads in Tauri', () => {
  expect(videoExportDownloadUrl('run/one')).toBe('http://backend.test/api/v1/necromatcher/video-exports/run%2Fone/download');
});
