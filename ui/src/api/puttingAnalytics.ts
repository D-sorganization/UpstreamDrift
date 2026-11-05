/**
 * Putting Analytics client (#14235).
 * Web twin of desktop pure-data layer (`src/tools/putting_analytics_gui/core.py`).
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
const STUDY_TIMEOUT_MS = 120_000;

export type Row = [string, string];
type Json = Record<string, unknown>;

export interface StudyForm {
  distanceFt: number; gradePct: number; aspectDeg: number; stimp: number;
  lineSdDeg: number; speedSd: number; objective: string; n: number; seed: number;
  baselineDistances: string; baselinePutts: string; aims: string; paces: string;
  reads: string; classes: string;
}

export const DEFAULT_FORM: StudyForm = {
  distanceFt: 10, gradePct: 0, aspectDeg: 0, stimp: 10, lineSdDeg: 1, speedSd: 0.08,
  objective: 'sg', n: 200, seed: 0, baselineDistances: '', baselinePutts: '',
  aims: '-4 -2 0 2 4', paces: '0.5 1 2 3',
  reads: '# slope_scale stimp_ratio (1 1 is a perfect read)\n1 1\n0.7 1\n1.3 1',
  classes: '# distance_ft slope_pct orientation stimp count\n4 1 uphill 10 4\n6 2 left_to_right 10 3',
};

export function parseFloats(text: string, name: string): number[] {
  const tokens = text.replace(/,/g, ' ').split(/\s+/).filter(Boolean);
  if (!tokens.length) throw new Error(`${name} is empty`);
  return tokens.map((token) => {
    const val = Number(token);
    if (!Number.isFinite(val)) throw new Error(`${name}: "${token}" is not a number`);
    return val;
  });
}

const lines = (text: string) =>
  text.split('\n').map((l) => l.replace(/,/g, ' ').split(/\s+/).filter(Boolean)).filter((r) => r.length && !r[0].startsWith('#'));

export function parseReads(text: string): Json[] {
  const reads = lines(text).map((row) => {
    if (row.length !== 2) throw new Error("each read is 'slope_scale stimp_ratio'");
    const [scale, ratio] = parseFloats(row.join(' '), 'read');
    return { slope_scale: scale, stimp_ratio: ratio };
  });
  if (!reads.length) throw new Error('enter at least one read');
  return reads;
}

export function parseClasses(text: string): Json[] {
  const classes = lines(text).map((row) => {
    if (row.length !== 5) throw new Error("each class is 'distance_ft slope_pct orientation stimp count'");
    const [dist, slope] = parseFloats(row.slice(0, 2).join(' '), 'class');
    const [stimp, count] = parseFloats(row.slice(3).join(' '), 'class');
    return { distance_ft: dist, slope_pct: slope, orientation: row[2], stimp, count };
  });
  if (!classes.length) throw new Error('enter at least one putt class');
  return classes;
}

export function studyArgs(form: StudyForm, mode: StudyMode): Json {
  const args: Json = {
    distance_ft: form.distanceFt, grade_pct: form.gradePct, aspect_deg: form.aspectDeg,
    stimp: form.stimp, line_sd_deg: form.lineSdDeg, speed_sd: form.speedSd,
    objective: form.objective, n: form.n, seed: form.seed,
    baseline: {
      distances_ft: parseFloats(form.baselineDistances, 'baseline distances'),
      expected_putts: parseFloats(form.baselinePutts, 'baseline putts'),
    },
    aims_deg: parseFloats(form.aims, 'aims'), paces_ft: parseFloats(form.paces, 'paces'),
  };
  if (mode === 'misread_cost') args.reads = parseReads(form.reads);
  if (mode === 'library_study') args.classes = parseClasses(form.classes);
  return args;
}

function fmt(value: unknown): string {
  if (typeof value !== 'number' || Number.isInteger(value)) return String(value);
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  if (exponent >= -4 && exponent < 4) return String(Number(value.toPrecision(4)));
  const [m, power] = value.toExponential(3).split('e');
  return `${Number(m)}e${power[0]}${power.replace(/^[+-]/, '').padStart(2, '0')}`;
}

const describeAction = (e: Json) => {
  const a = e.action as { aim_offset_deg: number; pace_ft_past: number };
  return `aim ${a.aim_offset_deg >= 0 ? '+' : ''}${a.aim_offset_deg.toFixed(1)} deg, pace ${a.pace_ft_past.toFixed(1)} ft`;
};

function optimizeRows(result: Json): Row[] {
  const best = result.best as Json;
  const rows: Row[] = [['Best', describeAction(best)], ...Object.entries(best).filter(([k]) => k !== 'action').map(([k, v]): Row => [k, fmt(v)])];
  (result.top as Json[]).forEach((entry, idx) => rows.push([`#${idx + 1}`, `${describeAction(entry)}: ${(entry.expected_strokes as number).toFixed(3)} strokes`]));
  return rows;
}

const isObject = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);

function flatRows(result: Json, prefix = ''): Row[] {
  return Object.entries(result).flatMap(([k, v]): Row[] => {
    const name = `${prefix}${k}`;
    if (isObject(v)) return flatRows(v, `${name}.`);
    if (Array.isArray(v) && v.every(isObject)) return v.flatMap((item, idx) => flatRows(item, `${name}[${idx}].`));
    return [[name, Array.isArray(v) ? v.map(fmt).join(', ') : fmt(v)]];
  });
}

export const resultRows = (mode: StudyMode, result: Json): Row[] => mode === 'optimize' ? optimizeRows(result) : flatRows(result);

export const runStudy = (mode: StudyMode, args: Json): Promise<Json> =>
  apiFetch<Json>(`${PUTTING_ANALYTICS_PATH}/${STUDY_PATHS[mode]}`, {
    method: 'POST', body: JSON.stringify(args), timeoutMs: STUDY_TIMEOUT_MS,
  });
