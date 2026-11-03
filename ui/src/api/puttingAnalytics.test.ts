import {
  DEFAULT_FORM, parseClasses, parseFloats, parseReads, resultRows, runStudy, studyArgs,
} from './puttingAnalytics';

const FORM = { ...DEFAULT_FORM, baselineDistances: '2 5 10', baselinePutts: '1.01 1.3 1.6' };

describe('putting analytics client (#14235)', () => {
  it('parses comma- or space-separated numbers and names the field on error', () => {
    expect(parseFloats('1, 2 3.5', 'aims')).toEqual([1, 2, 3.5]);
    expect(() => parseFloats('', 'aims')).toThrow('aims is empty');
    expect(() => parseFloats('1 nope', 'aims')).toThrow('aims: "nope" is not a number');
  });

  it('parses read lines skipping comments', () => {
    const reads = parseReads('# comment\n1 1\n0.8 1.2');
    expect(reads).toEqual([{ slope_scale: 1, stimp_ratio: 1 }, { slope_scale: 0.8, stimp_ratio: 1.2 }]);
    expect(() => parseReads('')).toThrow('enter at least one read');
    expect(() => parseReads('1 2 3')).toThrow("each read is 'slope_scale stimp_ratio'");
  });

  it('parses class lines skipping comments', () => {
    const classes = parseClasses('# header\n4 1 uphill 10 4\n6 2 downhill 10 2');
    expect(classes).toEqual([
      { distance_ft: 4, slope_pct: 1, orientation: 'uphill', stimp: 10, count: 4 },
      { distance_ft: 6, slope_pct: 2, orientation: 'downhill', stimp: 10, count: 2 },
    ]);
    expect(() => parseClasses('')).toThrow('enter at least one putt class');
    expect(() => parseClasses('4 1 uphill 10')).toThrow("each class is 'distance_ft slope_pct orientation stimp count'");
  });

  it('assembles service arguments for optimize, misread cost and library study', () => {
    const opt = studyArgs(FORM, 'optimize');
    expect(opt.distance_ft).toBe(10);
    expect(opt.baseline).toEqual({ distances_ft: [2, 5, 10], expected_putts: [1.01, 1.3, 1.6] });
    expect(opt.reads).toBeUndefined();

    const misread = studyArgs(FORM, 'misread_cost');
    expect(misread.reads).toHaveLength(3);

    const library = studyArgs(FORM, 'library_study');
    expect(library.classes).toHaveLength(2);
  });

  it('formats optimize results with best action and rank rows', () => {
    const rows = resultRows('optimize', {
      best: { action: { aim_offset_deg: -1.5, pace_ft_past: 1.5 }, expected_strokes: 1.42 },
      top: [
        { action: { aim_offset_deg: -1.5, pace_ft_past: 1.5 }, expected_strokes: 1.42 },
        { action: { aim_offset_deg: 0, pace_ft_past: 1 }, expected_strokes: 1.45 },
      ],
    });
    expect(rows).toEqual([
      ['Best', 'aim -1.5 deg, pace 1.5 ft'],
      ['expected_strokes', '1.42'],
      ['#1', 'aim -1.5 deg, pace 1.5 ft: 1.420 strokes'],
      ['#2', 'aim +0.0 deg, pace 1.0 ft: 1.450 strokes'],
    ]);
  });

  it('flattens nested results for misread and library studies', () => {
    const rows = resultRows('misread_cost', {
      cost: 0.12,
      by_read: [{ loss: 0.05, read: { scale: 0.8 } }],
    });
    expect(rows).toEqual([
      ['cost', '0.12'],
      ['by_read[0].loss', '0.05'],
      ['by_read[0].read.scale', '0.8'],
    ]);
  });

  it('posts the study to the matching route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'content-type': 'application/json' } }),
    );
    vi.stubGlobal('fetch', fetchMock);
    afterEach(() => vi.unstubAllGlobals());

    const result = await runStudy('optimize', { distance_ft: 10 });
    expect(result).toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/tools/putting-analytics/optimize',
      expect.objectContaining({ method: 'POST' }),
    );
  });
});
