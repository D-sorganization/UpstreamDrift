import { describe, expect, it } from 'vitest';
import {
  enabledPlotIds,
  selectedPlotIds,
  toggleCustomPlot,
} from './swingReportSelection';
import view from './fixtures/validFullView.json';

const rows = view.rows;
const WITHHELD = ['face_angle_ts', 'joint_load_ts'];

describe('swing report selection (mirrors the PyQt ReportViewModel)', () => {
  it('lists only enabled plots; the withheld set is unchanged', () => {
    expect(enabledPlotIds(rows)).toEqual(['clubhead_speed_ts', 'clubhead_path']);
    const disabled = rows.filter((r) => !r.enabled).map((r) => r.plot_id);
    expect(disabled).toEqual(WITHHELD);
  });

  it('Full selects every enabled plot regardless of the Custom set', () => {
    expect(selectedPlotIds(rows, 'full', [])).toEqual(['clubhead_speed_ts', 'clubhead_path']);
  });

  it('Custom keeps its own selection in report order', () => {
    expect(selectedPlotIds(rows, 'custom', ['clubhead_path', 'clubhead_speed_ts'])).toEqual([
      'clubhead_speed_ts',
      'clubhead_path',
    ]);
    expect(selectedPlotIds(rows, 'custom', ['clubhead_path'])).toEqual(['clubhead_path']);
  });

  it('round-trips a Custom toggle', () => {
    const added = toggleCustomPlot(rows, ['clubhead_speed_ts'], 'clubhead_path', true);
    expect(added).toEqual(['clubhead_speed_ts', 'clubhead_path']);
    expect(toggleCustomPlot(rows, added, 'clubhead_path', false)).toEqual(['clubhead_speed_ts']);
  });

  it.each([...WITHHELD, 'no_such_plot'])('refuses to select %s', (plotId) => {
    expect(() => toggleCustomPlot(rows, [], plotId, true)).toThrow(/cannot be selected/);
  });
});
