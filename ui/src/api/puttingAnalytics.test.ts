import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_FORM,
  parseClasses,
  parseFloats,
  parseReads,
  resultRows,
  runStudy,
  studyArgs,
} from './puttingAnalytics';

const FORM = { ...DEFAULT_FORM, baselineDistances: '2 5 10', baselinePutts: '1.01 1.3 1.6' };

describe('putting analytics client (#14235)', () => {
  it('parses comma- or space-separated numbers and names the field on error', () => {
    expect(parseFloats('1, 2 3.5', 'aims')).toEqual([1, 2, 3.5]);
    expect(() => parseFloats('', 'aims')).toThrow('aims is empty');
    expect(() => parseFloats('1 x', 'aims')).toThrow(/aims/);
  });

  it('parses reads and classes like the desktop tool, skipping # comments', () => {
    expect(parseReads('# hint\n1 1\n0.7, 1')).toEqual([
      { slope_scale: 1, stimp_ratio: 1 },
      { slope_scale: 0.7, stimp_ratio: 1 },
    ]);
    expect(() => parseReads('# only a comment')).toThrow('enter at least one read');
    expect(parseClasses('4 1 uphill 10 4')).toEqual([
      { distance_ft: 4, slope_pct: 1, orientation: 'uphill', stimp: 10, count: 4 },
    ]);
    expect(() => parseClasses('4 1 uphill')).toThrow(/each class is/);
  });

  it('builds the same service arguments as the desktop form', () => {
    const args = studyArgs(FORM, 'misread_cost');
    expect(args).toMatchObject({
      distance_ft: 10,
      objective: 'sg',
      n: 200,
      baseline: { distances_ft: [2, 5, 10], expected_putts: [1.01, 1.3, 1.6] },
      aims_deg: [-4, -2, 0, 2, 4],
      paces_ft: [0.5, 1, 2, 3],
      reads: [
        { slope_scale: 1, stimp_ratio: 1 },
        { slope_scale: 0.7, stimp_ratio: 1 },
        { slope_scale: 1.3, stimp_ratio: 1 },
      ],
    });
    expect(args).not.toHaveProperty('classes');
    expect(studyArgs(FORM, 'library_study')).toHaveProperty('classes');
    expect(() => studyArgs({ ...FORM, baselinePutts: '' }, 'optimize')).toThrow(
      'baseline putts is empty',
    );
  });

  it('formats rows like the desktop table', () => {
    const action = { aim_offset_deg: 2, pace_ft_past: 1 };
    const best = { action, expected_strokes: 1.43219, make_prob: 0.5 };
    expect(resultRows('optimize', { best, top: [best] })).toEqual([
      ['Best', 'aim +2.0 deg, pace 1.0 ft'],
      ['expected_strokes', '1.432'],
      ['make_prob', '0.5'],
      ['#1', 'aim +2.0 deg, pace 1.0 ft: 1.432 strokes'],
    ]);
    expect(
      resultRows('misread_cost', { reads: 2, rows: [{ cost: 0.123456 }], grid: [1, 2.5] }),
    ).toEqual([
      ['reads', '2'],
      ['rows[0].cost', '0.1235'],
      ['grid', '1, 2.5'],
    ]);
  });
});

describe('runStudy', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('posts the arguments to the matching route and returns its answer unchanged', async () => {
    const answer = { reads: 1 };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(answer), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const args = studyArgs(FORM, 'misread_cost');
    await expect(runStudy('misread_cost', args)).resolves.toEqual(answer);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/tools/putting-analytics/misread-cost');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual(args);
  });
});
