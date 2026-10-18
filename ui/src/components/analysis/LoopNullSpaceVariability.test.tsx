import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { LoopNullSpacePanel } from './LoopNullSpacePanel';
import * as api from '@/api/loopNullSpace';
import type {
  LoopNullspaceJobStatusV1,
  NSXPlotModel,
  SensitivityCatalogue,
} from '@/api/loopNullSpace';

vi.mock('@/api/fetch', () => ({ apiFetch: vi.fn() }));
vi.setConfig({ testTimeout: 30_000 });

const GOLDENS = resolve(__dirname, '../../../../tests/fixtures/loop_nullspace/goldens');
const v2 = (
  JSON.parse(readFileSync(resolve(GOLDENS, 'report_v2_variability.json'), 'utf8')) as {
    plot_model: NSXPlotModel;
  }
).plot_model;
const JOB = '123e4567-e89b-42d3-a456-426614174000';
const CAT: SensitivityCatalogue = {
  stages: [{ id: 'torque', label: 'TORQUE' }],
  spectra: [{ id: 'white', label: 'WHITE' }],
  mechanisms: [],
  strategies: [
    {
      id: 'S1',
      name: 'free',
      mechanism: 'm',
      availability: 'implemented',
      reason: '',
    },
  ],
  outcomes: [{ id: 'face_angle', label: 'face angle' }],
  default_outcome_priority: ['face_angle'],
  engines: [
    { id: 'pinocchio', method: 'analytic', available: true, reason: '' },
    {
      id: 'simscape',
      method: 'none',
      available: false,
      reason: 'snapshot grid only',
    },
  ],
};
const job = (over: Partial<LoopNullspaceJobStatusV1>): LoopNullspaceJobStatusV1 => ({
  job_id: JOB,
  status: 'completed',
  stage: null,
  progress_percent: 100,
  cache_key: 'k',
  cache_hit: false,
  worker_active: false,
  ...over,
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

beforeEach(() => {
  vi.spyOn(api, 'fetchLoopNullspacePresets').mockResolvedValue([
    { id: 'envelope_only', label: 'E' },
  ]);
  vi.spyOn(api, 'fetchLoopNullspaceSensitivityOptions').mockResolvedValue(CAT);
});

describe('Variability and Consistency in the panel', () => {
  it('sends the chosen sensitivity options and engine, then renders the v2 model', async () => {
    const submit = vi
      .spyOn(api, 'submitLoopNullspaceJob')
      .mockResolvedValue(job({ result: { plot_model: v2 } }));
    render(<LoopNullSpacePanel />);
    await act(async () => {});
    fireEvent.change(screen.getByLabelText('Source'), {
      target: { value: 'src1' },
    });
    fireEvent.change(screen.getByLabelText('Engine'), {
      target: { value: 'simscape' },
    });
    fireEvent.click(screen.getByLabelText('Run variability analysis'));
    fireEvent.change(screen.getByLabelText('Channels and amplitudes (variance)'), {
      target: { value: 'shoulder=0.004, elbow=0.009' },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    });
    const request = submit.mock.calls[0][0];
    expect(request.engine).toBe('simscape');
    expect(request.options?.sensitivity?.outcome_priority).toEqual(['face_angle']);
    expect(request.options?.sensitivity?.disturbance_spec[0].covariance).toEqual([
      [0.004, 0],
      [0, 0.009],
    ]);
    expect(screen.getByTestId('nsx-variability')).toHaveTextContent('Variability & Consistency');
    expect(screen.getByTestId('nsx-var-kernel_heatmap')).toHaveTextContent('UNVALIDATED');
  });

  it('sends no sensitivity block while the analysis is off', async () => {
    const submit = vi.spyOn(api, 'submitLoopNullspaceJob').mockResolvedValue(job({ result: {} }));
    render(<LoopNullSpacePanel />);
    await act(async () => {});
    fireEvent.change(screen.getByLabelText('Source'), {
      target: { value: 'src1' },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    });
    expect(submit.mock.calls[0][0].options?.sensitivity).toBeUndefined();
  });

  it('refuses to submit while the channel text does not parse', async () => {
    const submit = vi.spyOn(api, 'submitLoopNullspaceJob');
    render(<LoopNullSpacePanel />);
    await act(async () => {});
    fireEvent.change(screen.getByLabelText('Source'), {
      target: { value: 'src1' },
    });
    fireEvent.click(screen.getByLabelText('Run variability analysis'));
    fireEvent.change(screen.getByLabelText('Channels and amplitudes (variance)'), {
      target: { value: 'oops' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    expect(screen.getByTestId('nsx-form-error')).toBeInTheDocument();
    expect(submit).not.toHaveBeenCalled();
  });

  it('says so when the catalogue cannot be loaded and keeps the plain engine field', async () => {
    vi.spyOn(api, 'fetchLoopNullspaceSensitivityOptions').mockRejectedValue(new Error('down'));
    render(<LoopNullSpacePanel />);
    await act(async () => {});
    expect(screen.getByTestId('nsx-catalogue-error')).toHaveTextContent('down');
    expect(screen.getByLabelText('Engine').tagName).toBe('INPUT');
  });
});
