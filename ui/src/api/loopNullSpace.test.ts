import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  asPlotModel,
  cancelLoopNullspaceJob,
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
