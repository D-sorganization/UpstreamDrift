/**
 * Putting Analytics client (#14235).
 *
 * The web twin of the desktop tool's pure-data layer
 * (`src/tools/putting_analytics_gui/core.py`): it turns the form's text fields
 * into the arguments the shared putting service takes, posts them to the
 * `/tools/putting-analytics` routes (the same service functions the desktop
 * tool and the MCP tools call), and turns each answer into the same
 * Quantity/Value rows the desktop table shows. It adds no physics.
 *
 * The miss-pricing curve (`baseline`) always comes from the user: like the
 * service, the page ships no reference make-rate data (see #14176).
 */

import { apiFetch } from './fetch';

export const STUDY_MODES = ['optimize', 'misread_cost', 'library_study'] as const;
export type StudyMode = (typeof STUDY_MODES)[number];

export const MODE_LABELS: Record<StudyMode, string> = {
  optimize: 'Best aim and pace',
  misread_cost: 'Cost of misreads',
  library_study: 'Putt-class library',
};

const STUDY_PATHS: Record<StudyMode, string> = {
  optimize: 'optimize',
  misread_cost: 'misread-cost',
  library_study: 'library-study',
};

export const PUTTING_ANALYTICS_PATH = '/api/tools/putting-analytics';

/** A study runs for seconds at large rollout counts; allow well past that. */
const STUDY_TIMEOUT_MS = 120_000;

export type Row = [string, string];
type Json = Record<string, unknown>;

/** The form as entered: numbers already parsed, lists still as text. */
export interface StudyForm {
  distanceFt: number;
  gradePct: number;
  aspectDeg: number;
  stimp: number;
  lineSdDeg: number;
  speedSd: number;
  objective: string;
  n: number;
  seed: number;
  baselineDistances: string;
  baselinePutts: string;
  aims: string;
  paces: string;
  reads: string;
  classes: string;
}

export const DEFAULT_FORM: StudyForm = {
  distanceFt: 10,
  gradePct: 0,
  aspectDeg: 0,
  stimp: 10,
  lineSdDeg: 1,
  speedSd: 0.08,
  objective: 'sg',
  n: 200,
  seed: 0,
  baselineDistances: '',
  baselinePutts: '',
  aims: '-4 -2 0 2 4',
  paces: '0.5 1 2 3',
  reads: '# slope_scale stimp_ratio (1 1 is a perfect read)\n1 1\n0.7 1\n1.3 1',
  classes:
    '# distance_ft slope_pct orientation stimp count\n4 1 uphill 10 4\n6 2 left_to_right 10 3',
};

/** Comma- or space-separated numbers; throws an error naming the field. */
export function parseFloats(text: string, name: string): number[] {
  const tokens = text.replace(/,/g, ' ').split(/\s+/).filter(Boolean);
  if (tokens.length === 0) throw new Error(`${name} is empty`);
  return tokens.map((token) => {
    const value = Number(token);
    if (!Number.isFinite(value)) throw new Error(`${name}: "${token}" is not a number`);
    return value;
  });
}

function lines(text: string): string[][] {
  return text
    .split('\n')
    .map((line) => line.replace(/,/g, ' ').split(/\s+/).filter(Boolean))
    .filter((row) => row.length > 0 && !row[0].startsWith('#'));
}

/** One read per line: `slope_scale stimp_ratio` (1 1 is a perfect read). */
export function parseReads(text: string): Json[] {
  const reads = lines(text).map((row) => {
    if (row.length !== 2) throw new Error("each read is 'slope_scale stimp_ratio'");
    const [scale, ratio] = parseFloats(row.join(' '), 'read');
    return { slope_scale: scale, stimp_ratio: ratio };
  });
  if (reads.length === 0) throw new Error('enter at least one read');
  return reads;
}

/** One class per line: `distance_ft slope_pct orientation stimp count`. */
export function parseClasses(text: string): Json[] {
  const classes = lines(text).map((row) => {
    if (row.length !== 5) {
      throw new Error("each class is 'distance_ft slope_pct orientation stimp count'");
    }
    const [distance, slope] = parseFloats(row.slice(0, 2).join(' '), 'class');
    const [stimp, count] = parseFloats(row.slice(3).join(' '), 'class');
    return { distance_ft: distance, slope_pct: slope, orientation: row[2], stimp, count };
  });
  if (classes.length === 0) throw new Error('enter at least one putt class');
  return classes;
}

/** Service arguments for `mode`; throws on bad input. */
export function studyArgs(form: StudyForm, mode: StudyMode): Json {
  const args: Json = {
    distance_ft: form.distanceFt,
    grade_pct: form.gradePct,
    aspect_deg: form.aspectDeg,
    stimp: form.stimp,
    line_sd_deg: form.lineSdDeg,
    speed_sd: form.speedSd,
    objective: form.objective,
    n: form.n,
    seed: form.seed,
    baseline: {
      distances_ft: parseFloats(form.baselineDistances, 'baseline distances'),
      expected_putts: parseFloats(form.baselinePutts, 'baseline putts'),
    },
    aims_deg: parseFloats(form.aims, 'aims'),
    paces_ft: parseFloats(form.paces, 'paces'),
  };
  if (mode === 'misread_cost') args.reads = parseReads(form.reads);
  if (mode === 'library_study') args.classes = parseClasses(form.classes);
  return args;
}

/** Python's `f"{value:.4g}"` for non-integers, `str(value)` otherwise. */
function fmt(value: unknown): string {
  if (typeof value !== 'number' || Number.isInteger(value)) return String(value);
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  if (exponent >= -4 && exponent < 4) return String(Number(value.toPrecision(4)));
  const [mantissa, power] = value.toExponential(3).split('e');
  const digits = power.replace(/^[+-]/, '').padStart(2, '0');
  return `${Number(mantissa)}e${power[0]}${digits}`;
}

function describeAction(entry: Json): string {
  const action = entry.action as { aim_offset_deg: number; pace_ft_past: number };
  const aim = action.aim_offset_deg;
  return `aim ${aim >= 0 ? '+' : ''}${aim.toFixed(1)} deg, pace ${action.pace_ft_past.toFixed(1)} ft`;
}

function optimizeRows(result: Json): Row[] {
  const best = result.best as Json;
  const rows: Row[] = [['Best', describeAction(best)]];
  for (const [key, value] of Object.entries(best)) {
    if (key !== 'action') rows.push([key, fmt(value)]);
  }
  (result.top as Json[]).forEach((entry, index) => {
    const strokes = (entry.expected_strokes as number).toFixed(3);
    rows.push([`#${index + 1}`, `${describeAction(entry)}: ${strokes} strokes`]);
  });
  return rows;
}

const isObject = (value: unknown): value is Json =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Every scalar in a nested answer, keyed by its dotted (and indexed) path. */
function flatRows(result: Json, prefix = ''): Row[] {
  return Object.entries(result).flatMap(([key, value]): Row[] => {
    const name = `${prefix}${key}`;
    if (isObject(value)) return flatRows(value, `${name}.`);
    if (Array.isArray(value) && value.every(isObject)) {
      return value.flatMap((item, index) => flatRows(item, `${name}[${index}].`));
    }
    if (Array.isArray(value)) return [[name, value.map(fmt).join(', ')]];
    return [[name, fmt(value)]];
  });
}

/** Table rows for a study's answer, as the desktop tool shows them. */
export function resultRows(mode: StudyMode, result: Json): Row[] {
  return mode === 'optimize' ? optimizeRows(result) : flatRows(result);
}

/** Run one study through the API; the answer comes back unchanged. */
export function runStudy(mode: StudyMode, args: Json): Promise<Json> {
  return apiFetch<Json>(`${PUTTING_ANALYTICS_PATH}/${STUDY_PATHS[mode]}`, {
    method: 'POST',
    body: JSON.stringify(args),
    timeoutMs: STUDY_TIMEOUT_MS,
  });
}
