import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SwingReportDialog } from './SwingReportDialog';
import * as api from '@/api/swingReport';
import view from './fixtures/validFullView.json';

const doc = { schema_version: '1.0.0', report_id: view.report_id };

function renderDialog(download = vi.fn()) {
  vi.spyOn(api, 'fetchSwingReportView').mockResolvedValue(view);
  render(<SwingReportDialog report={doc} download={download} />);
  return download;
}

const box = (plotId: string) => screen.getByTestId(`swing-report-plot-${plotId}`) as HTMLInputElement;

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('SwingReportDialog (parity with the PyQt dialog, #12329)', () => {
  it('shows a withheld plot disabled with a visible reason and Not Validated label', async () => {
    renderDialog();
    await waitFor(() => expect(box('face_angle_ts')).toBeInTheDocument());
    fireEvent.click(screen.getByLabelText('Custom Selection'));
    expect(box('face_angle_ts')).toBeDisabled();
    expect(screen.getByText('Face Angle Ts [Not Validated]')).toBeInTheDocument();
    expect(screen.getByTestId('swing-report-detail-face_angle_ts')).toHaveTextContent(
      'Not validated (#10382).',
    );
  });

  it('shows the uncertainty for an advertised plot', async () => {
    renderDialog();
    await waitFor(() =>
      expect(screen.getByTestId('swing-report-detail-clubhead_speed_ts')).toHaveTextContent(
        'Uncertainty: engine_spread (provisional_cross_engine_spread), units m/s',
      ),
    );
  });

  it('Full selects every enabled plot and locks the checkboxes', async () => {
    renderDialog();
    await waitFor(() => expect(box('clubhead_path')).toBeChecked());
    expect(box('clubhead_speed_ts')).toBeChecked();
    expect(box('clubhead_path')).toBeDisabled();
    expect(box('face_angle_ts')).not.toBeChecked();
  });

  it('Custom starts from the default selection and round-trips toggles', async () => {
    renderDialog();
    await waitFor(() => expect(box('clubhead_path')).toBeInTheDocument());
    fireEvent.click(screen.getByLabelText('Custom Selection'));
    expect(box('clubhead_speed_ts')).toBeChecked();
    expect(box('clubhead_path')).not.toBeChecked();
    fireEvent.click(box('clubhead_path'));
    fireEvent.click(box('clubhead_speed_ts'));
    expect(box('clubhead_path')).toBeChecked();
    expect(box('clubhead_speed_ts')).not.toBeChecked();
    fireEvent.click(box('clubhead_path'));
    expect(screen.getByRole('button', { name: /export/i })).toBeDisabled();
  });

  it('exports the Custom selection through the API and downloads the package', async () => {
    const blob = new Blob(['zip']);
    const exportSpy = vi.spyOn(api, 'exportSwingReport').mockResolvedValue(blob);
    const download = renderDialog();
    await waitFor(() => expect(box('clubhead_path')).toBeInTheDocument());
    fireEvent.click(screen.getByLabelText('Custom Selection'));
    fireEvent.click(box('clubhead_path'));
    fireEvent.click(screen.getByRole('button', { name: /export/i }));
    await waitFor(() => expect(download).toHaveBeenCalledWith(blob, view.package_filename));
    expect(exportSpy).toHaveBeenCalledWith(doc, 'custom', ['clubhead_speed_ts', 'clubhead_path']);
  });

  it('exports the Full report', async () => {
    const exportSpy = vi.spyOn(api, 'exportSwingReport').mockResolvedValue(new Blob());
    renderDialog();
    await waitFor(() => expect(box('clubhead_path')).toBeChecked());
    fireEvent.click(screen.getByRole('button', { name: /export/i }));
    await waitFor(() =>
      expect(exportSpy).toHaveBeenCalledWith(doc, 'full', ['clubhead_speed_ts', 'clubhead_path']),
    );
  });

  it('reports an export failure instead of downloading', async () => {
    vi.spyOn(api, 'exportSwingReport').mockRejectedValue(new Error('disk full'));
    const download = renderDialog();
    await waitFor(() => expect(box('clubhead_path')).toBeChecked());
    fireEvent.click(screen.getByRole('button', { name: /export/i }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('disk full'));
    expect(download).not.toHaveBeenCalled();
  });

  it('shows why a report cannot be opened', async () => {
    vi.spyOn(api, 'fetchSwingReportView').mockRejectedValue(new Error('violates contract'));
    render(<SwingReportDialog report={doc} />);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('violates contract'));
  });
});
