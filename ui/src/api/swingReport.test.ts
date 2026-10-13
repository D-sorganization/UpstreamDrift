import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fetchModule from './fetch';
import {
  exportSwingReport,
  fetchSwingReportView,
  parseSwingReportView,
} from './swingReport';
import view from '@/components/analysis/swingReport/fixtures/validFullView.json';

const doc = { schema_version: '1.0.0', report_id: 'r1' };

afterEach(() => {
  vi.restoreAllMocks();
});

describe('fetchSwingReportView', () => {
  it('posts the document and returns the validated view', async () => {
    const spy = vi.spyOn(fetchModule, 'apiFetch').mockResolvedValue(view);
    await expect(fetchSwingReportView(doc)).resolves.toEqual(view);
    expect(spy).toHaveBeenCalledWith(
      '/api/swing-report/view',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ report: doc }) }),
    );
  });

  it.each([null, [], 'report', { report_id: 'r1' }, { schema_version: '1.0.0' }])(
    'rejects a non-report document %j without calling the API',
    async (bad) => {
      const spy = vi.spyOn(fetchModule, 'apiFetch');
      await expect(fetchSwingReportView(bad)).rejects.toThrow(/swing report/i);
      expect(spy).not.toHaveBeenCalled();
    },
  );
});

describe('parseSwingReportView', () => {
  it('accepts the Python view fixture unchanged', () => {
    expect(parseSwingReportView(view)).toEqual(view);
  });

  it('rejects a row whose enabled flag is not boolean', () => {
    const bad = { ...view, rows: [{ ...view.rows[0], enabled: 'yes' }] };
    expect(() => parseSwingReportView(bad)).toThrow(/enabled/);
  });

  it('rejects a default Custom id that is not an enabled row', () => {
    const bad = { ...view, default_custom_ids: ['face_angle_ts'] };
    expect(() => parseSwingReportView(bad)).toThrow(/face_angle_ts/);
  });
});

describe('exportSwingReport', () => {
  it('posts a Full export and returns the zip blob', async () => {
    const blob = new Blob(['zip'], { type: 'application/zip' });
    const spy = vi.spyOn(fetchModule, 'apiFetchBlob').mockResolvedValue(blob);
    await expect(exportSwingReport(doc, 'full', [])).resolves.toBe(blob);
    const [path, init] = spy.mock.calls[0];
    expect(path).toBe('/api/swing-report/export');
    expect(JSON.parse(String(init?.body))).toEqual({ report: doc, mode: 'full', plot_ids: [] });
  });

  it('sends the Custom selection', async () => {
    const spy = vi.spyOn(fetchModule, 'apiFetchBlob').mockResolvedValue(new Blob());
    await exportSwingReport(doc, 'custom', ['clubhead_path']);
    expect(JSON.parse(String(spy.mock.calls[0][1]?.body)).plot_ids).toEqual(['clubhead_path']);
  });

  it('rejects an empty Custom selection without calling the API', async () => {
    const spy = vi.spyOn(fetchModule, 'apiFetchBlob');
    await expect(exportSwingReport(doc, 'custom', [])).rejects.toThrow(/at least one plot/);
    expect(spy).not.toHaveBeenCalled();
  });
});
