/**
 * Tests for RecordingsPanel (issue #7451).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

import { RecordingsPanel } from './RecordingsPanel';
import * as recordingsApi from '@/api/recordings';

vi.mock('@/api/recordings', async () => {
  const actual =
    await vi.importActual<typeof import('@/api/recordings')>('@/api/recordings');
  return {
    ...actual,
    listRecordings: vi.fn(),
    fetchExportFormats: vi.fn(),
    saveRecording: vi.fn(),
    deleteRecording: vi.fn(),
  };
});

const mockListRecordings = vi.mocked(recordingsApi.listRecordings);
const mockFetchExportFormats = vi.mocked(recordingsApi.fetchExportFormats);
const mockSaveRecording = vi.mocked(recordingsApi.saveRecording);
const mockDeleteRecording = vi.mocked(recordingsApi.deleteRecording);

const REC: recordingsApi.RecordingMeta = {
  id: 'rec_20260612_abcd1234',
  engine: 'mujoco',
  model: 'pendulum.xml',
  duration: 0.24,
  frames: 25,
  created: '2026-06-12T00:00:00+00:00',
};

const FORMATS: recordingsApi.ExportFormats = {
  json: { name: 'JSON', extension: '.json', available: true, description: '' },
  csv: { name: 'CSV', extension: '.csv', available: true, description: '' },
  mat: { name: 'MATLAB', extension: '.mat', available: false, description: '' },
};

describe('RecordingsPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockListRecordings.mockResolvedValue([REC]);
    mockFetchExportFormats.mockResolvedValue(FORMATS);
    mockSaveRecording.mockResolvedValue(REC);
    mockDeleteRecording.mockResolvedValue(undefined);
  });

  it('does not fetch until expanded', () => {
    render(<RecordingsPanel />);
    expect(mockListRecordings).not.toHaveBeenCalled();
    expect(mockFetchExportFormats).not.toHaveBeenCalled();
  });

  it('lists recordings after expanding', async () => {
    render(<RecordingsPanel />);
    fireEvent.click(screen.getByRole('button', { name: /recordings/i }));

    await waitFor(() => {
      expect(screen.getByTestId(`recording-${REC.id}`)).toBeInTheDocument();
    });
    expect(mockListRecordings).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/mujoco · 25 frames/)).toBeInTheDocument();
  });

  it('renders download links only for available formats', async () => {
    render(<RecordingsPanel />);
    fireEvent.click(screen.getByRole('button', { name: /recordings/i }));

    await waitFor(() => {
      expect(
        screen.getByLabelText(`Download ${REC.id} as JSON`),
      ).toBeInTheDocument();
    });
    expect(screen.getByLabelText(`Download ${REC.id} as CSV`)).toHaveAttribute(
      'href',
      expect.stringContaining(`/api/recordings/${REC.id}/export?format=csv`),
    );
    // MATLAB is unavailable (scipy missing in the mocked formats) — hidden.
    expect(
      screen.queryByLabelText(`Download ${REC.id} as MATLAB`),
    ).not.toBeInTheDocument();
  });

  it('saves the current recording and refreshes the list', async () => {
    render(<RecordingsPanel />);
    fireEvent.click(screen.getByRole('button', { name: /recordings/i }));
    await waitFor(() => expect(mockListRecordings).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole('button', { name: /save current recording/i }));
    await waitFor(() => expect(mockSaveRecording).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(mockListRecordings).toHaveBeenCalledTimes(2));
  });

  it('deletes only after confirmation', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    render(<RecordingsPanel />);
    fireEvent.click(screen.getByRole('button', { name: /recordings/i }));
    await waitFor(() => {
      expect(screen.getByLabelText(`Delete recording ${REC.id}`)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByLabelText(`Delete recording ${REC.id}`));
    expect(mockDeleteRecording).not.toHaveBeenCalled();

    confirmSpy.mockReturnValue(true);
    fireEvent.click(screen.getByLabelText(`Delete recording ${REC.id}`));
    await waitFor(() => expect(mockDeleteRecording).toHaveBeenCalledWith(REC.id));
    confirmSpy.mockRestore();
  });

  it('shows an error message when loading fails', async () => {
    mockListRecordings.mockRejectedValue(new Error('backend down'));
    render(<RecordingsPanel />);
    fireEvent.click(screen.getByRole('button', { name: /recordings/i }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('backend down');
    });
  });

  it('disables save while a simulation is running', async () => {
    render(<RecordingsPanel isRunning />);
    fireEvent.click(screen.getByRole('button', { name: /recordings/i }));

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: /save current recording/i }),
      ).toBeDisabled();
    });
  });
});
