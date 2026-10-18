import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  asPlotModel,
  cancelLoopNullspaceJob,
  fetchLoopNullspacePresets,
  fetchLoopNullspaceSensitivityOptions,
  getLoopNullspaceJob,
  isTerminalLoopNullspaceJob,
  nextLoopNullspaceDelay,
  plotModelOf,
  submitLoopNullspaceJob,
  type LoopNullspaceJobStatusV1,
} from './loopNullSpace';
import { apiFetch } from './fetch';

vi.mock('./fetch', () => ({ apiFetch: vi.fn() }));

const GOLDEN = resolve(
  __dirname,
  '../../../tests/fixtures/loop_nullspace/goldens/plot_model_v1.json',
);
const golden = JSON.parse(readFileSync(GOLDEN, 'utf8')) as {
  plot_model: unknown;
};
const JOB = '123e4567-e89b-42d3-a456-426614174000';

const job = (over: Partial<LoopNullspaceJobStatusV1>): LoopNullspaceJobStatusV1 => ({
  job_id: JOB,
  status: 'running',
  progress_percent: 10,
  cache_key: 'k',
  cache_hit: false,
  worker_active: true,
  ...over,
});

afterEach(() => vi.clearAllMocks());

describe('loop-nullspace job client', () => {
  it('submits with POST and the request body', async () => {
    vi.mocked(apiFetch).mockResolvedValue(job({}));
    const request = {
      source_id: 's',
      engine: 'planar-numpy',
      preset: 'envelope_only',
      frame_indices: [0, 3],
    };
    await submitLoopNullspaceJob(request);
    const [path, init] = vi.mocked(apiFetch).mock.calls[0];
    expect(path).toBe('/api/analysis/loop-nullspace/jobs');
    expect(init?.method).toBe('POST');
    expect(JSON.parse(init?.body as string)).toEqual(request);
  });

  it('polls by id and cancels with DELETE', async () => {
    vi.mocked(apiFetch).mockResolvedValue(job({}));
    await getLoopNullspaceJob(JOB);
    await cancelLoopNullspaceJob(JOB);
    expect(vi.mocked(apiFetch).mock.calls[0][0]).toBe(`/api/analysis/loop-nullspace/jobs/${JOB}`);
    expect(vi.mocked(apiFetch).mock.calls[1][1]?.method).toBe('DELETE');
  });

  it('refuses a malformed job id without sending', () => {
    expect(() => getLoopNullspaceJob('../x')).toThrow(/not a loop-nullspace job id/);
    expect(apiFetch).not.toHaveBeenCalled();
  });

  it('backs polling off to the cap and knows terminal states', () => {
    expect(nextLoopNullspaceDelay(250)).toBe(500);
    expect(nextLoopNullspaceDelay(2000)).toBe(2000);
    expect(isTerminalLoopNullspaceJob(job({ status: 'cancelled' }))).toBe(true);
    expect(isTerminalLoopNullspaceJob(job({ status: 'running' }))).toBe(false);
  });
});

describe('plot model extraction', () => {
  it('accepts the golden and takes it from a result or a partial', () => {
    expect(asPlotModel(golden.plot_model)?.preset).toBe('envelope_only');
    expect(
      plotModelOf(job({ status: 'completed', result: { plot_model: golden.plot_model } })),
    ).not.toBeNull();
    expect(plotModelOf(job({ partial: { plot_model: golden.plot_model } }))).not.toBeNull();
  });

  it('fails closed on a missing, foreign-version or incomplete model', () => {
    expect(plotModelOf(job({ status: 'completed', result: { invariants: {} } }))).toBeNull();
    expect(plotModelOf(job({ partial: { invariants: {} } }))).toBeNull();
    expect(
      asPlotModel({
        ...(golden.plot_model as object),
        schema_version: '2.0.0',
      }),
    ).toBeNull();
    expect(asPlotModel({ schema_version: '1.0.0' })).toBeNull();
    expect(asPlotModel(null)).toBeNull();
  });
});

describe('plot model 2.0.0 (NSX-100)', () => {
  const v2 = (
    JSON.parse(
      readFileSync(
        resolve(__dirname, '../../../tests/fixtures/loop_nullspace/goldens/report_v2_variability.json'),
        'utf8',
      ),
    ) as { plot_model: unknown }
  ).plot_model;

  it('accepts the v2 golden and still accepts 1.0.0', () => {
    expect(asPlotModel(v2)?.sensitivity?.primary_outcome).toBe('face_angle');
    expect(asPlotModel(golden.plot_model)?.sensitivity).toBeUndefined();
  });

  it('refuses a 2.0.0 model without sensitivity and any other version', () => {
    const { sensitivity: _drop, ...rest } = v2 as Record<string, unknown>;
    expect(asPlotModel(rest)).toBeNull();
    expect(asPlotModel({ ...(v2 as object), schema_version: '3.0.0' })).toBeNull();
  });
});

describe('sensitivity options', () => {
  const cat = {
    stages: [],
    spectra: [],
    mechanisms: [],
    strategies: [],
    outcomes: [{ id: 'face_angle', label: 'face angle' }],
    default_outcome_priority: ['face_angle'],
    engines: [],
  };
  it('returns the served catalogue and rejects a malformed one', async () => {
    vi.mocked(apiFetch).mockResolvedValue(cat);
    await expect(fetchLoopNullspaceSensitivityOptions()).resolves.toEqual(cat);
    expect(vi.mocked(apiFetch).mock.calls.at(-1)![0]).toBe(
      '/api/analysis/loop-nullspace/sensitivity-options',
    );
    vi.mocked(apiFetch).mockResolvedValue({ stages: 3 });
    await expect(fetchLoopNullspaceSensitivityOptions()).rejects.toThrow(/malformed/);
  });
});

describe('preset list', () => {
  it('returns the served presets', async () => {
    vi.mocked(apiFetch).mockResolvedValue([{ id: 'envelope_only', label: 'Envelope only' }]);
    await expect(fetchLoopNullspacePresets()).resolves.toEqual([
      { id: 'envelope_only', label: 'Envelope only' },
    ]);
  });

  it.each([{}, null, [{ id: 'x' }], [{ id: 1, label: 'y' }]])(
    'rejects a malformed preset payload %j instead of returning it',
    async (payload) => {
      vi.mocked(apiFetch).mockResolvedValue(payload);
      await expect(fetchLoopNullspacePresets()).rejects.toThrow(/malformed preset list/);
    },
  );
});
