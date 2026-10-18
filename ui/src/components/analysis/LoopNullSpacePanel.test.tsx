import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { LoopNullSpacePanel } from './LoopNullSpacePanel';
import { PlotModelView } from './loopNullSpace/PlotModelView';
import { DISPLAY_CLASS_TEXT, fmt, fmtList, parseFrameIndices } from './loopNullSpace/format';
import * as api from '@/api/loopNullSpace';
import type { LoopNullspaceJobStatusV1, NSXPlotModel } from '@/api/loopNullSpace';

vi.mock('@/api/fetch', () => ({ apiFetch: vi.fn() }));

// Rendering the full golden model is heavy on a loaded machine.
vi.setConfig({ testTimeout: 20_000 });

// The one cross-UI fixture, read in place (never copied): NSX-42/43/44.
const GOLDEN = resolve(
  __dirname,
  '../../../../tests/fixtures/loop_nullspace/goldens/plot_model_v1.json',
);
const model = (JSON.parse(readFileSync(GOLDEN, 'utf8')) as { plot_model: NSXPlotModel }).plot_model;
const JOB = '123e4567-e89b-42d3-a456-426614174000';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

const allBands = [...model.envelopes.bands, ...model.hand_split.bands];
const overlayFor = (id: string) => model.allocation.overlays.find((o) => o.band_id === id);

describe('PlotModelView on the shared golden', () => {
  it('draws exactly the bands the golden shows, with decision text', () => {
    render(<PlotModelView model={model} />);
    for (const band of allBands) {
      const fig = screen.getByTestId(`nsx-band-${band.quantity_id}`);
      const decision = within(screen.getByTestId(`nsx-band-${band.quantity_id}-decision`));
      expect(decision.getByText(`Label: ${band.decision.label}`)).toBeInTheDocument();
      for (const badge of band.decision.badges) {
        expect(decision.getByText(`Badge: ${badge}`)).toBeInTheDocument();
      }
      for (const chip of band.decision.hypothesis_chips) {
        expect(decision.getByText(`Hypothesis: ${chip}`)).toBeInTheDocument();
      }
      expect(fig).toHaveTextContent(`${band.quantity_id} (${band.units})`);
    }
    expect(screen.getAllByTestId(/^nsx-band-[^-]+.*-area$/).length).toBeGreaterThan(0);
  });

  it('labels every segment by class, sign and binding rows as text', () => {
    render(<PlotModelView model={model} />);
    for (const band of allBands) {
      band.segments.forEach((seg, i) => {
        const row = screen.getByTestId(`nsx-band-${band.quantity_id}-segment-${i}`);
        expect(row).toHaveTextContent(`${seg.first_frame_index}-${seg.last_frame_index}`);
        expect(row).toHaveTextContent(DISPLAY_CLASS_TEXT[seg.display_class]);
        expect(row).toHaveTextContent(seg.sign_class ?? 'unavailable');
        seg.binding_hypotheses.forEach((h) => expect(row).toHaveTextContent(h));
        seg.active_constraints.forEach((c) => expect(row).toHaveTextContent(c));
      });
    }
    const classes = new Set(allBands.flatMap((b) => b.segments.map((s) => s.display_class)));
    expect(classes).toEqual(new Set(['DETERMINED_SIGN', 'INDETERMINATE', 'UNBOUNDED']));
  });

  it('scales each chart to the golden extents', () => {
    render(<PlotModelView model={model} />);
    const tau1 = model.envelopes.bands[0];
    expect(screen.getByTestId('nsx-band-tau_1-ymin')).toHaveTextContent(
      fmt(Math.min(...(tau1.lower as number[]), ...(overlayFor('tau_1')!.values as number[]))),
    );
    expect(screen.getByTestId('nsx-band-tau_1-ymax')).toHaveTextContent(
      fmt(Math.max(...(tau1.upper as number[]))),
    );
  });

  it('draws a representative only on its band and withholds the rest with the reason', () => {
    render(<PlotModelView model={model} />);
    for (const o of model.allocation.overlays) {
      expect(screen.getByTestId(`nsx-band-${o.band_id}-representative`)).toBeInTheDocument();
      expect(screen.getByTestId(`nsx-band-${o.band_id}-representative-note`)).toHaveTextContent(
        o.decision.label,
      );
    }
    for (const w of [...model.envelopes.withheld, ...model.allocation.withheld]) {
      expect(screen.queryByTestId(`nsx-band-${w.quantity_id}`)).toBeNull();
    }
    const withheld = model.envelopes.withheld[0];
    expect(screen.getByTestId(`nsx-withheld-envelopes-${withheld.quantity_id}`)).toHaveTextContent(
      `${withheld.action} - ${withheld.reason}`,
    );
    const alloc = model.allocation.withheld[0];
    expect(screen.getByTestId(`nsx-withheld-allocation-${alloc.quantity_id}`)).toHaveTextContent(
      alloc.reason,
    );
  });

  it('shows invariant numbers only where shown, otherwise the reason and no number', () => {
    render(<PlotModelView model={model} />);
    for (const series of model.invariants.series) {
      for (const f of series.frames) {
        const row = screen.getByTestId(`nsx-invariant-${series.name}-${f.frame_index}`);
        if (f.shown) {
          expect(row).toHaveTextContent('Shown');
          expect(row).toHaveTextContent(fmtList(f.values));
        } else {
          expect(row).toHaveTextContent('Withheld');
          expect(row).toHaveTextContent(f.reason);
          expect(row.textContent).not.toMatch(/\d\.\d/);
        }
      }
    }
  });

  it('renders diagnostics, frame status, non-claims and the grip frame', () => {
    render(<PlotModelView model={model} />);
    const adm = model.diagnostics.admission!;
    expect(screen.getByTestId('nsx-admission')).toHaveTextContent(
      `Admission verdict: ${adm.verdict}`,
    );
    for (const r of model.diagnostics.residuals) {
      const li = screen.getByTestId(`nsx-residual-${r.name}`);
      if (!r.shown) expect(li).toHaveTextContent(`withheld - ${r.reason}`);
    }
    for (const f of model.diagnostics.frames) {
      expect(screen.getByTestId(`nsx-rank-${f.index}`)).toHaveTextContent(String(f.rank_A));
    }
    for (const f of model.envelopes.frames) {
      expect(screen.getByTestId(`nsx-frame-${f.index}`)).toHaveTextContent(
        f.feasible ? 'Feasible' : 'Infeasible',
      );
    }
    for (const c of model.non_claims) expect(screen.getByText(c)).toBeInTheDocument();
    expect(screen.getByTestId('nsx-hand-split-frame')).toHaveTextContent(model.hand_split.frame);
    expect(screen.getByTestId('nsx-allocation-non-claim')).toHaveTextContent(
      model.allocation.non_claim!,
    );
  });

  it('obeys a changed decision: a band whose decision admits none is withheld by reason', () => {
    const changed = structuredClone(model);
    changed.envelopes.bands[0].decision = {
      ...changed.envelopes.bands[0].decision,
      action: 'WITHHOLD',
      reason: 'policy says no',
    };
    render(<PlotModelView model={changed} />);
    expect(screen.queryByTestId('nsx-band-tau_1')).toBeNull();
    expect(screen.getByTestId('nsx-band-tau_1-withheld')).toHaveTextContent('policy says no');
  });

  it('never draws a value whose decision is not SHOW_VALUE, even if marked shown', () => {
    const changed = structuredClone(model);
    const frame = changed.invariants.series.find((s) => s.name === 'loop_power')!.frames[0];
    frame.decision = { ...frame.decision!, action: 'SHOW_REASON' };
    render(<PlotModelView model={changed} />);
    expect(screen.getByTestId('nsx-invariant-loop_power-0')).toHaveTextContent('Withheld');
    expect(screen.getByTestId('nsx-invariant-loop_power-0')).not.toHaveTextContent(
      fmtList(frame.values),
    );
  });

  it('does not bridge a bound the model does not mark BOUNDED', () => {
    const changed = structuredClone(model);
    changed.envelopes.bands[0].status_upper = changed.envelopes.bands[0].status_upper.map(
      () => 'UNBOUNDED',
    );
    render(<PlotModelView model={changed} />);
    expect(screen.queryByTestId('nsx-band-tau_1-area')).toBeNull();
  });
});

describe('parseFrameIndices', () => {
  it('accepts whole numbers and refuses anything else', () => {
    expect(parseFrameIndices('0, 3,6')).toEqual([0, 3, 6]);
    expect(parseFrameIndices('0,,3')).toBeNull();
    expect(parseFrameIndices('1.5')).toBeNull();
    expect(parseFrameIndices('-1')).toBeNull();
  });
});

const job = (over: Partial<LoopNullspaceJobStatusV1>): LoopNullspaceJobStatusV1 => ({
  job_id: JOB,
  status: 'running',
  stage: 'invariants',
  progress_percent: 45,
  cache_key: 'k',
  cache_hit: false,
  worker_active: true,
  ...over,
});

function fill() {
  fireEvent.change(screen.getByLabelText('Source'), {
    target: { value: 'src1' },
  });
  fireEvent.change(screen.getByLabelText('Engine'), {
    target: { value: 'planar-numpy' },
  });
}

describe('LoopNullSpacePanel job flow', () => {
  it('offers the registry presets and refuses an incomplete form', () => {
    render(<LoopNullSpacePanel />);
    expect(within(screen.getByLabelText('Preset')).getAllByRole('option')).toHaveLength(
      api.LOOP_NULLSPACE_PRESETS.length,
    );
    const submit = vi.spyOn(api, 'submitLoopNullspaceJob');
    fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    expect(screen.getByTestId('nsx-form-error')).toBeInTheDocument();
    expect(submit).not.toHaveBeenCalled();
  });

  it('submits the chosen preset, shows a partial without a model as held back, then the final model', async () => {
    vi.useFakeTimers();
    const submit = vi
      .spyOn(api, 'submitLoopNullspaceJob')
      .mockResolvedValue(job({ partial: { invariants: { loop_power: [1, 2] } } }));
    const get = vi.spyOn(api, 'getLoopNullspaceJob').mockResolvedValue(
      job({
        status: 'completed',
        progress_percent: 100,
        result: { plot_model: model },
      }),
    );
    render(<LoopNullSpacePanel />);
    fill();
    fireEvent.change(screen.getByLabelText('Preset'), {
      target: { value: 'smooth_capacity' },
    });
    fireEvent.change(screen.getByLabelText('Frame indices'), {
      target: { value: '0,3' },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    });
    expect(submit.mock.calls[0][0]).toEqual({
      source_id: 'src1',
      engine: 'planar-numpy',
      preset: 'smooth_capacity',
      frame_indices: [0, 3],
    });
    expect(screen.getByTestId('nsx-job-status')).toHaveTextContent('stage: invariants');
    expect(screen.getByTestId('nsx-no-model')).toHaveTextContent('held back');
    expect(screen.queryByTestId('nsx-plot-model')).toBeNull();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(get).toHaveBeenCalledWith(JOB);
    expect(screen.getByTestId('nsx-plot-model')).toBeInTheDocument();
    expect(screen.queryByTestId('nsx-partial-banner')).toBeNull();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
  });

  it('renders a partial plot model with a visible partial banner', async () => {
    vi.spyOn(api, 'submitLoopNullspaceJob').mockResolvedValue(
      job({ partial: { plot_model: model } }),
    );
    render(<LoopNullSpacePanel />);
    fill();
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    });
    expect(screen.getByTestId('nsx-partial-banner')).toBeInTheDocument();
    expect(screen.getByTestId('nsx-plot-model')).toBeInTheDocument();
  });

  it('cancels the running job', async () => {
    vi.spyOn(api, 'submitLoopNullspaceJob').mockResolvedValue(job({}));
    const cancel = vi
      .spyOn(api, 'cancelLoopNullspaceJob')
      .mockResolvedValue(job({ status: 'cancelled', worker_active: false }));
    render(<LoopNullSpacePanel />);
    fill();
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    });
    expect(cancel).toHaveBeenCalledWith(JOB);
    expect(screen.getByTestId('nsx-job-status')).toHaveTextContent('Job status: cancelled');
    expect(screen.getByTestId('nsx-no-model')).toHaveTextContent('No result to show');
  });

  it('shows a refused submission and a completed job without a model', async () => {
    vi.spyOn(api, 'submitLoopNullspaceJob').mockRejectedValueOnce(new Error('unknown preset'));
    render(<LoopNullSpacePanel />);
    fill();
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    });
    expect(screen.getByTestId('nsx-error')).toHaveTextContent('unknown preset');
    vi.spyOn(api, 'submitLoopNullspaceJob').mockResolvedValue(
      job({ status: 'completed', cache_hit: true, result: { invariants: {} } }),
    );
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    });
    expect(screen.getByTestId('nsx-job-status')).toHaveTextContent('served from cache');
    expect(screen.getByTestId('nsx-no-model')).toHaveTextContent('no plot model');
  });
});
