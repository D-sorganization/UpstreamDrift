/**
 * BunkerShot3D designer workbench, web page (#9545, slices 3 and 5).
 *
 * Web counterpart of the PyQt workbench (`src/tools/bunker_shot_gui/gui.py`)
 * over the job routes `src/api/routes/bunker_workbench_jobs.py`: one bounded
 * nominal solve of design A, and the A/B comparison. Each runs as a job with
 * a labelled progress bar and a Cancel button (`useBunkerJob`); a failed or
 * cancelled job, or a result that does not answer the current submission, is
 * shown as such and never as a result. Every result shows its
 * fidelity tier, verdict, in-frame stamp and sand-property sources, and the
 * comparison shows the carry-objective standing. A lead on an unqualified
 * objective is reported as exploratory, never as a ranking (#9239).
 */

import { useState, type ReactNode } from 'react';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';
import {
  exploratoryLeader,
  rankedWinner,
  type BunkerDesignV1,
  type BunkerJobStatusV1,
  type BunkerWorkbenchComparisonV1,
  type BunkerWorkbenchEvaluationV1,
} from '@/api/bunkerWorkbench';
import { useBunkerJob, type BunkerJobView } from './useBunkerJob';

interface DesignForm {
  name: string;
  loft: string;
  bounce: string;
  width: string;
}

interface ObjectiveForm {
  targetCarry: string;
  tolerance: string;
}

type Result =
  | { kind: 'evaluation'; record: BunkerWorkbenchEvaluationV1 }
  | { kind: 'comparison'; record: BunkerWorkbenchComparisonV1 };

const DESIGN_FIELDS = [
  ['loft', 'loft (deg)', 'loft_deg'],
  ['bounce', 'marketed bounce (deg)', 'marketed_bounce_deg'],
  ['width', 'sole width (mm)', 'sole_width_mm'],
] as const;

const INPUT_CLASS = 'rounded bg-gray-900 border border-gray-700 px-2 py-1 text-sm';

/** A finite number, or an explicit error naming the field (never a silent NaN). */
function requireNumber(text: string, label: string): number {
  const value = Number(text);
  if (text.trim() === '' || !Number.isFinite(value)) {
    throw new Error(`${label} must be a number, got "${text}"`);
  }
  return value;
}

/** A blank field keeps the grind preset's value (`null` on the wire). */
function optionalNumber(text: string, label: string): number | null {
  return text.trim() === '' ? null : requireNumber(text, label);
}

function toDesign(title: string, form: DesignForm): BunkerDesignV1 {
  const design: BunkerDesignV1 = { name: form.name.trim() };
  for (const [key, label, wire] of DESIGN_FIELDS) {
    design[wire] = optionalNumber(form[key], `${title} ${label}`);
  }
  return design;
}

function toObjective(form: ObjectiveForm) {
  return {
    use: 'exploratory' as const,
    target_carry_m: requireNumber(form.targetCarry, 'Target carry (m)'),
    tolerance_fraction: requireNumber(form.tolerance, 'Carry tolerance (fraction)'),
  };
}

function TextField(props: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-gray-400">
      {props.label}
      <input
        aria-label={props.label}
        className={INPUT_CLASS}
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
      />
    </label>
  );
}

function DesignFields(props: { title: string; form: DesignForm; onChange: (f: DesignForm) => void }) {
  const { title, form, onChange } = props;
  return (
    <fieldset className="flex flex-col gap-2 border border-gray-700 rounded p-2">
      <legend className="px-1 text-sm text-white">{title}</legend>
      <TextField label={`${title} name`} value={form.name} onChange={(name) => onChange({ ...form, name })} />
      {DESIGN_FIELDS.map(([key, label]) => (
        <TextField
          key={key}
          label={`${title} ${label}`}
          value={form[key]}
          onChange={(value) => onChange({ ...form, [key]: value })}
        />
      ))}
    </fieldset>
  );
}

function Row(props: { term: string; children: ReactNode }) {
  return (
    <div className="flex gap-2">
      <dt className="text-gray-400 min-w-32">{props.term}</dt>
      <dd>{props.children}</dd>
    </div>
  );
}

function EvaluationCard({ record, title }: { record: BunkerWorkbenchEvaluationV1; title: string }) {
  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 text-sm">
      <h3 className="font-medium text-white">{title}</h3>
      <p className="my-2 inline-block rounded bg-amber-900 px-2 py-1 font-mono text-amber-50">{record.stamp}</p>
      <dl className="flex flex-col gap-1">
        <Row term="Tier">{`${record.tier} (${record.tier_model})`}</Row>
        <Row term="Verdict">{record.verdict.status}</Row>
        <Row term="Headline">{record.verdict.headline}</Row>
        <Row term="Carry">{record.carry ? `${record.carry.carry_m.toFixed(2)} m` : 'unavailable'}</Row>
        <Row term="Sand sources">
          <ul>
            {Object.entries(record.sources.sand).map(([prop, stamp]) => (
              <li key={prop}>{`${prop}: ${stamp.basis} (${stamp.source})`}</li>
            ))}
          </ul>
        </Row>
        <Row term="Digest">
          <span className="font-mono">{record.digest.slice(0, 12)}</span>
        </Row>
      </dl>
    </section>
  );
}

function ComparisonSummary({ record }: { record: BunkerWorkbenchComparisonV1 }) {
  const winner = rankedWinner(record);
  const leader = exploratoryLeader(record);
  const standing = record.objective_standing;
  return (
    <section className="rounded border border-gray-700 bg-gray-800 p-3 text-sm">
      <h3 className="font-medium text-white">A/B comparison</h3>
      <dl className="mt-2 flex flex-col gap-1">
        <Row term="Objective standing">{standing.disposition}</Row>
        <Row term="Reasons">
          <ul>{standing.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
        </Row>
        <Row term="Winner">
          <span data-testid="bunker-winner">{winner ?? `Not ranked (${standing.disposition})`}</span>
        </Row>
        {leader !== null && (
          <Row term="Exploratory leader">
            <span data-testid="bunker-exploratory-leader">{`${leader} (not a ranking)`}</span>
          </Row>
        )}
        <Row term="Statement">{record.verdict_statement}</Row>
      </dl>
    </section>
  );
}

function ResultView({ result }: { result: Result }) {
  if (result.kind === 'evaluation') {
    return <EvaluationCard record={result.record} title="Design A, nominal solve" />;
  }
  const { record } = result;
  return (
    <>
      <ComparisonSummary record={record} />
      <EvaluationCard record={record.left} title="Design A" />
      <EvaluationCard record={record.right} title="Design B" />
    </>
  );
}

/** The typed result of a completed job; the record's kind is the submission's. */
function resultOf(view: Extract<BunkerJobView, { phase: 'completed' }>): Result {
  return view.kind === 'evaluate'
    ? { kind: 'evaluation', record: view.record as unknown as BunkerWorkbenchEvaluationV1 }
    : { kind: 'comparison', record: view.record as unknown as BunkerWorkbenchComparisonV1 };
}

function JobProgress(props: { status: BunkerJobStatusV1; onCancel: () => void }) {
  const { status, onCancel } = props;
  const percent = Math.round(Math.min(100, Math.max(0, status.progress_percent)));
  const text = `${percent}% of ${status.shots_planned} shots`;
  return (
    <section className="flex flex-col gap-2 rounded border border-gray-700 bg-gray-800 p-3 text-sm">
      <p>{status.status === 'pending' ? 'Queued…' : 'Solving…'}</p>
      <progress aria-label="Bunker job progress" aria-valuetext={text} className="w-full" max={100} value={percent} />
      <p className="text-xs text-gray-400">{text}</p>
      <button type="button" className="self-start rounded bg-gray-700 px-3 py-1 text-white" onClick={onCancel}>
        Cancel job
      </button>
    </section>
  );
}

function JobOutcome(props: { view: BunkerJobView; onCancel: () => void }) {
  const { view, onCancel } = props;
  switch (view.phase) {
    case 'submitting':
      return <p role="status">Submitting…</p>;
    case 'running':
      return <JobProgress status={view.status} onCancel={onCancel} />;
    case 'completed':
      return <ResultView result={resultOf(view)} />;
    case 'cancelled':
      return <p role="status">Job cancelled; it produced no result.</p>;
    case 'failed':
      return <p role="alert" className="text-red-300">{view.message}</p>;
    case 'stale':
      return (
        <p role="alert" className="text-red-300">
          Discarded a stale result: it does not answer the current submission. Run it again.
        </p>
      );
    default:
      return (
        <p className="text-gray-400">Solve design A or compare A/B. Every result carries its tier, verdict and sources.</p>
      );
  }
}

export function BunkerWorkbenchPage() {
  const [designA, setDesignA] = useState<DesignForm>({ name: 'Design A', loft: '56', bounce: '12', width: '15' });
  const [designB, setDesignB] = useState<DesignForm>({ name: 'Design B', loft: '56', bounce: '8', width: '11' });
  const [objective, setObjective] = useState<ObjectiveForm>({ targetCarry: '12', tolerance: '0.10' });
  const { view, notice, busy, submit, cancel } = useBunkerJob();

  const base = () => ({ handedness: 'right' as const, objective: toObjective(objective) });
  const solve = () => submit('evaluate', () => ({ ...base(), design: toDesign('Design A', designA) }));
  const compare = () =>
    submit('compare', () => ({
      ...base(),
      left: toDesign('Design A', designA),
      right: toDesign('Design B', designB),
    }));

  const leftPanel = (
    <div className="flex flex-col gap-3 p-4 text-gray-200">
      <h2 className="text-base font-semibold text-white">BunkerShot3D Workbench</h2>
      <p className="text-xs text-gray-400">Right-handed delivery; blank fields keep the grind preset.</p>
      <DesignFields title="Design A" form={designA} onChange={setDesignA} />
      <DesignFields title="Design B" form={designB} onChange={setDesignB} />
      <TextField label="Target carry (m)" value={objective.targetCarry} onChange={(targetCarry) => setObjective({ ...objective, targetCarry })} />
      <TextField label="Carry tolerance (fraction)" value={objective.tolerance} onChange={(tolerance) => setObjective({ ...objective, tolerance })} />
      <button type="button" className="rounded bg-blue-700 px-3 py-1 text-white disabled:opacity-50" disabled={busy} onClick={solve}>
        Solve design A
      </button>
      <button type="button" className="rounded bg-blue-700 px-3 py-1 text-white disabled:opacity-50" disabled={busy} onClick={compare}>
        Compare A/B
      </button>
    </div>
  );

  return (
    <WorkspaceShell leftPanel={leftPanel}>
      <div className="flex flex-col h-full gap-4 p-4 overflow-y-auto text-gray-200">
        <JobOutcome view={view} onCancel={cancel} />
        {notice && <p role="status" className="text-amber-300">{notice}</p>}
      </div>
    </WorkspaceShell>
  );
}

export default BunkerWorkbenchPage;
