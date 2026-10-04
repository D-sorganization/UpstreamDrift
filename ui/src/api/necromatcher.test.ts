import { beforeEach, expect, it, vi } from 'vitest';
import { submitVideoExport, fetchVideoExport, cancelVideoExport, videoExportDownloadUrl, submitRefit, registerReviewedWindow, type RefitOptions } from './necromatcher';
const request = vi.hoisted(() => vi.fn());
const formRequest = vi.hoisted(() => vi.fn());
vi.mock('./fetch', () => ({apiFetch: request, apiFetchForm: formRequest}));
vi.mock('./backend', () => ({getApiBase: () => 'http://backend.test'}));
beforeEach(() => {request.mockReset(); request.mockResolvedValue({});});
it('transports the typed lossless restricted seed pair without inventing receipt or optimization', async () => {
  const payload: RefitOptions & {new_fit_id: string} = {
    new_fit_id: 'seed', frame_indices: [0, 1, 3], knot_count: 3, coordinate_scales: [1],
    unknown_visibility_weight: 0.5, budget_wall_s: 300,
    operation: 'restrict_initialization', initialization_source: 'restricted_spline',
    config: {max_iterations: 30, prior_weight: 0.1, smoothness_weight: 0.01, closure_weight: 100, initialization_policy: 'strict'},
  };
  await submitRefit('parent', payload);
  expect(request).toHaveBeenCalledExactlyOnceWith('/api/v1/necromatcher/fits/parent/refits', {method: 'POST', body: JSON.stringify(payload)});
  expect(payload).not.toHaveProperty('spline_interval_restriction');
});
it('uploads exact review file bytes through the shared multipart transport', async () => {
  const file = new File(['{ "schema": "review" }\n'], 'review.json', {type: 'application/json'});
  await registerReviewedWindow('fit/one', file);
  expect(formRequest).toHaveBeenCalledWith('/api/v1/necromatcher/fits/fit%2Fone/source-scope-reviews', expect.any(FormData), {timeoutMs: 300_000});
  expect(formRequest.mock.calls[0][1].get('file')).toEqual(file);
  expect(request).not.toHaveBeenCalled();
});
it('submits a bodyless export for an encoded fit identity', async () => {
  await submitVideoExport('fit/one');
  expect(request).toHaveBeenCalledExactlyOnceWith('/api/v1/necromatcher/fits/fit%2Fone/video-exports', {method:'POST'});
});
it('transports optional shape and shaft display records together', async () => {
  const options = {shape_overlay: {opacity: 0.6}, shaft_evidence: {schema: 'reviewed'}};
  await submitVideoExport('fit/one', options);
  expect(request).toHaveBeenCalledExactlyOnceWith('/api/v1/necromatcher/fits/fit%2Fone/video-exports', {method:'POST', body: JSON.stringify(options)});
});
it('polls and cancels only an encoded run identity', async () => {
  await fetchVideoExport('run/one'); await cancelVideoExport('run/one');
  expect(request).toHaveBeenNthCalledWith(1, '/api/v1/necromatcher/video-exports/run%2Fone');
  expect(request).toHaveBeenNthCalledWith(2, '/api/v1/necromatcher/video-exports/run%2Fone/cancel', {method:'POST'});
});
it('uses the canonical backend base for export downloads in Tauri', () => {
  expect(videoExportDownloadUrl('run/one')).toBe('http://backend.test/api/v1/necromatcher/video-exports/run%2Fone/download');
});


it('sends nested fit configuration and explicit spline initialization without flattening', async () => {
  const payload: RefitOptions & {new_fit_id: string} = {
    new_fit_id: 'new', frame_indices: [0, 3], knot_count: 2, coordinate_scales: [1],
    unknown_visibility_weight: 0.5, budget_wall_s: 300, operation: 'fit', initialization_source: 'preserved_spline',
    config: {max_iterations: 30, prior_weight: 0.1, smoothness_weight: 0.01, closure_weight: 100,
      coordinate_bounds: [['hip', -1, 1]], interior_fractions: [0.5], initialization_policy: 'strict',
      constraint_options: {ground: {normal: [0, 0, 1], height_m: 0}, position_weight: 100,
        rotation_weight: 50, ground_weight: 100, position_scale_m: 0.01,
        rotation_scale_rad: 0.1, ground_scale_m: 0.01, pinned_spheres: ['heel_r', 'heel_l']},
    },
  };
  await submitRefit('fit/one', payload);
  expect(request).toHaveBeenCalledExactlyOnceWith('/api/v1/necromatcher/fits/fit%2Fone/refits', {method: 'POST', body: JSON.stringify(payload)});
  const transported = JSON.parse(request.mock.calls[0][1].body);
  expect(transported.config).toEqual(payload.config);
  expect(transported).not.toHaveProperty('coordinate_bounds');
});
