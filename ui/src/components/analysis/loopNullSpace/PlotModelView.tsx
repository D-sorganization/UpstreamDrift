/**
 * Renders an `NSXPlotModel` (NSX-44, #12576). Everything shown, withheld or
 * explained here comes from the model's own decisions: a band is drawn only
 * when its decision admits a band, a value only when it is `shown` under a
 * `SHOW_VALUE` decision, a representative only on its band. Anything else is a
 * visible reason. No quantity is computed in this file.
 */

import type { Band, DecisionView, NSXPlotModel, Overlay, WithheldItem } from '@/api/loopNullSpace';
import { BandChart } from './BandChart';
import { fmt, fmtList } from './format';

const BAND_ACTIONS = ['SHOW_BAND', 'SHOW_REPRESENTATIVE_ON_BAND'];

const admitsBand = (d: DecisionView) => BAND_ACTIONS.includes(d.action);

function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section data-testid={`nsx-panel-${id}`} className="mb-4">
      <h4 className="text-sm font-semibold text-gray-200 mb-2">{title}</h4>
      {children}
    </section>
  );
}

function Unavailable({ reason, testId }: { reason: string | null; testId: string }) {
  return reason ? (
    <p data-testid={testId} className="text-xs text-amber-300">
      Unavailable: {reason}
    </p>
  ) : null;
}

function Withheld({ items, panel }: { items: WithheldItem[]; panel: string }) {
  if (items.length === 0) return null;
  return (
    <ul data-testid={`nsx-${panel}-withheld`} className="text-xs text-amber-300 mb-2">
      {items.map((w) => (
        <li key={w.quantity_id} data-testid={`nsx-withheld-${panel}-${w.quantity_id}`}>
          {w.quantity_id}: {w.action} - {w.reason}
        </li>
      ))}
    </ul>
  );
}

/** A band whose decision admits none is a reason line, not a chart. */
function BandOrReason({ band, overlay }: { band: Band; overlay?: Overlay }) {
  if (!admitsBand(band.decision)) {
    return (
      <p data-testid={`nsx-band-${band.quantity_id}-withheld`} className="text-xs text-amber-300">
        {band.quantity_id}: {band.decision.action} - {band.decision.reason}
      </p>
    );
  }
  const shownOverlay =
    overlay && overlay.decision.action === 'SHOW_REPRESENTATIVE_ON_BAND' ? overlay : undefined;
  return <BandChart band={band} overlay={shownOverlay} />;
}

function InvariantsSection({ model }: { model: NSXPlotModel }) {
  const { series, unavailable_reason } = model.invariants;
  return (
    <Section id="invariants" title="Invariants">
      <Unavailable reason={unavailable_reason} testId="nsx-invariants-unavailable" />
      {series.map((s) => (
        <table
          key={s.name}
          data-testid={`nsx-invariant-${s.name}`}
          className="text-xs text-gray-300 w-full mb-2"
        >
          <caption className="text-left text-gray-200">
            {s.name} ({s.units})
          </caption>
          <thead>
            <tr className="text-left text-gray-400">
              <th scope="col">Frame</th>
              <th scope="col">Status</th>
              <th scope="col">Values or reason</th>
            </tr>
          </thead>
          <tbody>
            {s.frames.map((f) => {
              const show = f.shown && f.decision?.action === 'SHOW_VALUE';
              return (
                <tr key={f.frame_index} data-testid={`nsx-invariant-${s.name}-${f.frame_index}`}>
                  <td>{f.frame_index}</td>
                  <td>{show ? 'Shown' : 'Withheld'}</td>
                  <td>
                    {show ? fmtList(f.values) : f.reason || f.decision?.reason || 'unavailable'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ))}
    </Section>
  );
}

function EnvelopesSection({ model }: { model: NSXPlotModel }) {
  const { bands, withheld, frames, unavailable_reason } = model.envelopes;
  const overlays = new Map(model.allocation.overlays.map((o) => [o.band_id, o]));
  return (
    <Section id="envelopes" title="Envelopes">
      <Unavailable reason={unavailable_reason} testId="nsx-envelopes-unavailable" />
      <Withheld items={withheld} panel="envelopes" />
      {bands.map((b) => (
        <BandOrReason key={b.quantity_id} band={b} overlay={overlays.get(b.quantity_id)} />
      ))}
      <table data-testid="nsx-frame-status" className="text-[10px] text-gray-300 w-full">
        <caption className="text-left text-gray-200">Frame status</caption>
        <thead>
          <tr className="text-left text-gray-400">
            <th scope="col">Frame</th>
            <th scope="col">Feasibility</th>
            <th scope="col">Chebyshev</th>
            <th scope="col">Margin</th>
            <th scope="col">Conflicting hypotheses</th>
            <th scope="col">Evidence</th>
          </tr>
        </thead>
        <tbody>
          {frames.map((f) => (
            <tr key={f.index} data-testid={`nsx-frame-${f.index}`}>
              <td>{f.index}</td>
              <td>{f.feasible ? 'Feasible' : 'Infeasible'}</td>
              <td>{f.chebyshev_status}</td>
              <td>{fmt(f.chebyshev_margin)}</td>
              <td>{f.conflict_hypotheses.join(', ') || 'none'}</td>
              <td>{f.rests_on_unverified ? 'Rests on unverified' : 'Verified'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

function HandSplitSection({ model }: { model: NSXPlotModel }) {
  const { frame, bands, withheld, unavailable_reason } = model.hand_split;
  const overlays = new Map(model.allocation.overlays.map((o) => [o.band_id, o]));
  return (
    <Section id="hand_split" title="Hand Split">
      <p data-testid="nsx-hand-split-frame" className="text-xs text-gray-400">
        Coordinate frame: {frame}
      </p>
      <Unavailable reason={unavailable_reason} testId="nsx-hand-split-unavailable" />
      <Withheld items={withheld} panel="hand_split" />
      {bands.map((b) => (
        <BandOrReason key={b.quantity_id} band={b} overlay={overlays.get(b.quantity_id)} />
      ))}
    </Section>
  );
}

function AllocationSection({ model }: { model: NSXPlotModel }) {
  const a = model.allocation;
  return (
    <Section id="allocation" title="Allocation">
      <Unavailable reason={a.unavailable_reason} testId="nsx-allocation-unavailable" />
      <p data-testid="nsx-allocation-status" className="text-xs text-gray-300">
        Status: {a.status ?? 'unavailable'}
        {a.reason ? ` - ${a.reason}` : ''}
      </p>
      <p data-testid="nsx-allocation-objective" className="text-xs text-gray-300">
        Objective total: {fmt(a.objective_total)}
      </p>
      {a.excluded_frame_indices.length > 0 && (
        <p className="text-xs text-amber-300">
          Excluded frames: {a.excluded_frame_indices.join(', ')}
        </p>
      )}
      <Withheld items={a.withheld} panel="allocation" />
      <p className="text-xs text-gray-400">
        Representatives are drawn on their bands in the Envelopes and Hand Split panels.
      </p>
      {a.non_claim && (
        <p data-testid="nsx-allocation-non-claim" className="text-xs text-gray-300">
          Non-claim: {a.non_claim}
        </p>
      )}
    </Section>
  );
}

function DiagnosticsSection({ model }: { model: NSXPlotModel }) {
  const d = model.diagnostics;
  return (
    <Section id="diagnostics" title="Diagnostics">
      {d.admission && (
        <p data-testid="nsx-admission" className="text-xs text-gray-300">
          Admission verdict: {d.admission.verdict}; derivative method:{' '}
          {d.admission.derivative_method}; noise floor: {fmt(d.admission.noise_floor)}; position
          residual {d.admission.position_evaluated ? 'evaluated' : 'not evaluated'}
          {d.admission.reasons.length > 0 && `; reasons: ${d.admission.reasons.join('; ')}`}
          {d.admission.failing_frames.length > 0 &&
            `; failing frames: ${d.admission.failing_frames.join(', ')}`}
        </p>
      )}
      <ul className="text-xs text-gray-300 mb-2">
        {d.residuals.map((r) => (
          <li key={r.name} data-testid={`nsx-residual-${r.name}`}>
            Scaled residual {r.name} ({r.units}):{' '}
            {r.shown && r.decision?.action === 'SHOW_VALUE'
              ? fmtList(r.values)
              : `withheld - ${r.reason || r.decision?.reason || 'unavailable'}`}
          </li>
        ))}
      </ul>
      {d.kkt && (
        <p data-testid="nsx-kkt" className="text-xs text-gray-300">
          Allocation KKT residuals: primal {fmt(d.kkt.primal)}, dual {fmt(d.kkt.dual)},
          complementarity {fmt(d.kkt.complementarity)}
        </p>
      )}
      <table data-testid="nsx-rank-table" className="text-[10px] text-gray-300 w-full">
        <caption className="text-left text-gray-200">Rank and condition</caption>
        <thead>
          <tr className="text-left text-gray-400">
            <th scope="col">Frame</th>
            <th scope="col">Rank A</th>
            <th scope="col">Nullity</th>
            <th scope="col">Rank Jc</th>
            <th scope="col">Condition</th>
          </tr>
        </thead>
        <tbody>
          {d.frames.map((f) => (
            <tr key={f.index} data-testid={`nsx-rank-${f.index}`}>
              <td>{f.index}</td>
              <td>{f.rank_A}</td>
              <td>{f.nullity}</td>
              <td>{f.rank_J_c}</td>
              <td>{fmt(f.condition_number)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

export function PlotModelView({ model }: { model: NSXPlotModel }) {
  return (
    <div data-testid="nsx-plot-model">
      <p data-testid="nsx-model-status" className="text-xs text-gray-300 mb-2">
        Status: {model.status}
        {model.reason ? ` - ${model.reason}` : ''}; preset: {model.preset}; engine:{' '}
        {model.engine_name}; registry {model.registry_version}; frames:{' '}
        {model.frame_indices.join(', ')}
      </p>
      <ul data-testid="nsx-non-claims" className="text-[11px] text-gray-400 list-disc pl-4 mb-3">
        {model.non_claims.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
      <InvariantsSection model={model} />
      <EnvelopesSection model={model} />
      <HandSplitSection model={model} />
      <AllocationSection model={model} />
      <DiagnosticsSection model={model} />
    </div>
  );
}
