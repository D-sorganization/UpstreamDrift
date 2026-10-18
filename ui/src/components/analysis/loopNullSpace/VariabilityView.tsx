/**
 * Variability and Consistency panels of a plot model 2.0.0 (NSX-100, #12729).
 * Everything here is the payload's: decision labels, UNVALIDATED badges and
 * hypothesis chips are shown verbatim, a withheld item keeps its reason and has
 * no number, and an unavailable panel reads "UNAVAILABLE:" plus its reason. No
 * quantity is computed or gated in this file.
 */

import type {
  DecisionView,
  DispersionEntry,
  ParetoPoint,
  SensitivityPanels,
} from '@/api/loopNullSpace';
import { DecisionBadges } from './DecisionBadges';
import { fmt } from './format';

const UNAVAILABLE = 'UNAVAILABLE: ';

function Panel({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section data-testid={`nsx-var-${id}`} className="mb-4">
      <h5 className="text-sm font-semibold text-gray-200 mb-1">{title}</h5>
      {children}
    </section>
  );
}

const Why = ({ text }: { text: string }) => (
  <p className="text-xs text-amber-300">Withheld: {text}</p>
);

const Badges = ({ decision, id }: { decision: DecisionView | null; id: string }) =>
  decision ? <DecisionBadges decision={decision} testId={`nsx-var-${id}-decision`} /> : null;

function Unavailable({ reason, id }: { reason: string; id: string }) {
  return (
    <p data-testid={`nsx-var-${id}-unavailable`} className="text-xs text-amber-300">
      {UNAVAILABLE}
      {reason}
    </p>
  );
}

function Kernel({ s }: { s: SensitivityPanels }) {
  const k = s.kernel_heatmap;
  if (s.status !== 'complete') return <Unavailable reason={s.reason ?? ''} id="kernel_heatmap" />;
  return (
    <>
      <Badges decision={k.decision} id="kernel_heatmap" />
      {!k.shown ? (
        <Why text={k.reason} />
      ) : (
        <table className="text-[10px] text-gray-300 w-full">
          <caption className="text-left text-gray-200">
            {k.outcome} ({k.units}) per input;{' '}
            {k.is_band ? 'family band (lower..upper)' : 'allocation-invariant kernel'}
          </caption>
          <thead>
            <tr className="text-left text-gray-400">
              <th scope="col">Time (s)</th>
              {k.input_labels.map((l) => (
                <th scope="col" key={l}>
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {k.step_times_s.map((t, i) => (
              <tr key={i}>
                <td>{fmt(t)}</td>
                {k.lower[i].map((lo, j) => (
                  <td key={j}>{k.is_band ? `${fmt(lo)}..${fmt(k.upper[i][j])}` : fmt(lo)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

function Budget({ s }: { s: SensitivityPanels }) {
  const b = s.variance_budget;
  if (s.status !== 'complete') return <Unavailable reason={s.reason ?? ''} id="variance_budget" />;
  return (
    <>
      <Badges decision={b.decision} id="variance_budget" />
      {!b.shown ? (
        <Why text={b.reason} />
      ) : (
        <table className="text-[10px] text-gray-300 w-full">
          <caption className="text-left text-gray-200">
            {b.outcome} variance ({b.units}) by input class
          </caption>
          <thead>
            <tr className="text-left text-gray-400">
              <th scope="col">Input Class</th>
              <th scope="col">Variance</th>
              <th scope="col">Share</th>
            </tr>
          </thead>
          <tbody>
            {b.parts.map((p) => (
              <tr key={p.input_class}>
                <td>{p.input_class}</td>
                <td>{fmt(p.variance)}</td>
                <td>{fmt(p.share)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

function PointRow({ p }: { p: ParetoPoint }) {
  return (
    <>
      <tr data-testid={`nsx-var-point-${p.label}`}>
        <td>{p.label}</td>
        <td>{p.shown ? 'Shown' : 'Withheld'}</td>
        <td>{p.rank ?? ''}</td>
        <td>{p.shown ? fmt(p.variance_ratio) : 'unavailable'}</td>
        <td>{p.shown ? fmt(p.chs_mps) : 'unavailable'}</td>
        <td>{p.shown ? `${fmt(p.effort)} (${p.effort_basis})` : 'unavailable'}</td>
        <td>{p.reason}</td>
      </tr>
      {p.decision && (
        <tr>
          <td colSpan={7}>
            <Badges decision={p.decision} id={`point-${p.label}`} />
          </td>
        </tr>
      )}
    </>
  );
}

function Pareto({ s }: { s: SensitivityPanels }) {
  const p = s.strategy_pareto;
  if (p.unavailable_reason)
    return <Unavailable reason={p.unavailable_reason} id="strategy_pareto" />;
  return (
    <>
      <p className="text-xs text-gray-300">
        Mode: {p.mode}; ranking basis: {p.ranking_basis}
        {p.winner ? `; winner: ${p.winner}` : ''}
      </p>
      <table className="text-[10px] text-gray-300 w-full">
        <thead>
          <tr className="text-left text-gray-400">
            <th scope="col">Arm</th>
            <th scope="col">State</th>
            <th scope="col">Rank</th>
            <th scope="col">Variance Ratio</th>
            <th scope="col">Clubhead Speed (m/s)</th>
            <th scope="col">Effort</th>
            <th scope="col">Reason</th>
          </tr>
        </thead>
        <tbody>
          {p.points.map((pt) => (
            <PointRow key={pt.label} p={pt} />
          ))}
          {p.skipped.map((sk) => (
            <tr key={sk.label} data-testid={`nsx-var-skipped-${sk.label}`}>
              <td>{sk.label}</td>
              <td>{sk.status}</td>
              <td />
              <td>unavailable</td>
              <td>unavailable</td>
              <td>unavailable</td>
              <td>{sk.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <ul className="text-[11px] text-gray-400 list-disc pl-4">
        {p.fronts.map((f) => (
          <li key={f.name}>
            Front {f.name}: {f.labels.join(', ') || 'none'} (withheld: {f.withheld_count})
          </li>
        ))}
        {p.assumptions.map((a) => (
          <li key={a}>{a}</li>
        ))}
        {p.non_claim && <li>{p.non_claim}</li>}
      </ul>
    </>
  );
}

function EntryRow({ e }: { e: DispersionEntry }) {
  return (
    <>
      <tr data-testid={`nsx-var-sd-${e.label}`}>
        <td>{e.label}</td>
        <td>{e.shown ? 'Shown' : 'Withheld'}</td>
        <td>{e.shown ? fmt(e.lower) : 'unavailable'}</td>
        <td>{e.shown ? fmt(e.upper) : 'unavailable'}</td>
        <td>{e.units}</td>
        <td>{e.reason}</td>
      </tr>
      {e.decision && (
        <tr>
          <td colSpan={6}>
            <Badges decision={e.decision} id={`sd-${e.label}`} />
          </td>
        </tr>
      )}
    </>
  );
}

function Dispersion({ s }: { s: SensitivityPanels }) {
  const d = s.face_dispersion;
  if (d.unavailable_reason)
    return <Unavailable reason={d.unavailable_reason} id="face_dispersion" />;
  const entries = [d.linear, ...d.arms].filter((e): e is DispersionEntry => e !== null);
  return (
    <>
      <table className="text-[10px] text-gray-300 w-full">
        <caption className="text-left text-gray-200">
          {d.outcome} standard deviation ({d.units})
        </caption>
        <thead>
          <tr className="text-left text-gray-400">
            <th scope="col">Source</th>
            <th scope="col">State</th>
            <th scope="col">Lower</th>
            <th scope="col">Upper</th>
            <th scope="col">Units</th>
            <th scope="col">Reason</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => (
            <EntryRow key={e.label} e={e} />
          ))}
        </tbody>
      </table>
      {d.mc.map((m) => (
        <div
          key={m.outcome}
          data-testid={`nsx-var-mc-${m.outcome}`}
          className="text-xs text-gray-300"
        >
          Monte Carlo {m.outcome} ({m.status}):{' '}
          {m.shown
            ? `predicted ${fmt(m.predicted_sd)}, Monte Carlo ${fmt(m.mc_sd)}, gap ${fmt(m.gap)}, failure rate ${fmt(m.failure_rate)}`
            : `withheld - ${m.reason}`}
          <Badges decision={m.decision} id={`mc-${m.outcome}`} />
        </div>
      ))}
    </>
  );
}

export function VariabilityView({ sensitivity: s }: { sensitivity: SensitivityPanels }) {
  const complete = s.status === 'complete';
  return (
    <div data-testid="nsx-variability-view">
      {complete ? (
        <ul data-testid="nsx-var-header" className="text-xs text-gray-300 mb-2">
          <li>Primary outcome: {s.primary_outcome}</li>
          <li>Outcomes: {s.outcomes.join(', ')}</li>
          <li>Declared mechanisms: {s.declared_mechanisms.join(', ') || 'none'}</li>
          <li>Noise hypotheses: {s.noise_hypotheses.join(', ') || 'none'}</li>
          <li>Linear validity: {s.linear_validity}</li>
          <li>Cache identity: {s.cache_identity}</li>
          {s.skipped_priority.length > 0 && (
            <li>Outcomes skipped from the priority: {s.skipped_priority.join(', ')}</li>
          )}
        </ul>
      ) : (
        <p data-testid="nsx-var-status" className="text-xs text-amber-300 mb-2">
          {UNAVAILABLE}
          {s.reason} (stage status: {s.status})
        </p>
      )}
      <ul
        data-testid="nsx-var-non-claims"
        className="text-[11px] text-gray-400 list-disc pl-4 mb-3"
      >
        {s.non_claims.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
      <Panel id="kernel_heatmap" title="Kernel Heatmap">
        <Kernel s={s} />
      </Panel>
      <Panel id="variance_budget" title="Variance Budget">
        <Budget s={s} />
      </Panel>
      <Panel id="strategy_pareto" title="Strategy Pareto">
        <Pareto s={s} />
      </Panel>
      <Panel id="face_dispersion" title="Face Dispersion">
        <Dispersion s={s} />
      </Panel>
    </div>
  );
}
