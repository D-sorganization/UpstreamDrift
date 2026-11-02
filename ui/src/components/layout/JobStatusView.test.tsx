import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { JobStatusView } from './JobStatusView';

describe('JobStatusView component (#14268)', () => {
  it('renders nothing when phase is idle', () => {
    const { container } = render(<JobStatusView status={{ phase: 'idle' }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders submitting state with accessible role="status"', () => {
    render(<JobStatusView status={{ phase: 'submitting' }} />);
    const status = screen.getByRole('status');
    expect(status).toHaveTextContent('Submitting…');
  });

  it('renders queued state with job ID and queued badge', () => {
    render(
      <JobStatusView
        status={{
          jobId: '11111111-2222-3333-4444-555555555555',
          phase: 'queued',
          progressPercent: 0,
        }}
      />
    );
    expect(screen.getByText('Queued…')).toBeInTheDocument();
    expect(screen.getByTestId('job-id-display')).toHaveTextContent(
      '11111111-2222-3333-4444-555555555555'
    );
  });

  it('renders running progress bar with shots planned and triggers cancel', () => {
    const onCancel = vi.fn();
    render(
      <JobStatusView
        status={{
          jobId: 'job-abc-123',
          phase: 'running',
          progressPercent: 45,
          shotsPlanned: 52,
          label: 'Bunker job progress',
        }}
        onCancel={onCancel}
      />
    );

    const progress = screen.getByRole('progressbar', { name: 'Bunker job progress' });
    expect(progress).toHaveAttribute('value', '45');
    expect(progress).toHaveAttribute('aria-valuetext', '45% of 52 shots');
    expect(screen.getByText('45% of 52 shots')).toBeInTheDocument();

    const cancelBtn = screen.getByRole('button', { name: 'Cancel job' });
    expect(cancelBtn).toHaveClass('touch-target');
    fireEvent.click(cancelBtn);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('renders stage description when shotsPlanned is not provided', () => {
    render(
      <JobStatusView
        status={{
          phase: 'running',
          progressPercent: 60,
          stage: 'optimizing trajectory',
          label: 'Simulation progress',
        }}
      />
    );

    const progress = screen.getByRole('progressbar', { name: 'Simulation progress' });
    expect(progress).toHaveAttribute('aria-valuetext', '60% · optimizing trajectory');
    expect(screen.getByText('60% · optimizing trajectory')).toBeInTheDocument();
  });

  it('renders failed state with alert role and triggers retry', () => {
    const onRetry = vi.fn();
    render(
      <JobStatusView
        status={{
          phase: 'failed',
          error: 'ValueError: sole unconstructible',
        }}
        onRetry={onRetry}
      />
    );

    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('ValueError: sole unconstructible');

    const retryBtn = screen.getByRole('button', { name: 'Retry job' });
    expect(retryBtn).toHaveClass('touch-target');
    fireEvent.click(retryBtn);
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('renders cancelled state with status role', () => {
    render(<JobStatusView status={{ phase: 'cancelled' }} />);
    expect(screen.getByRole('status')).toHaveTextContent('Job cancelled; it produced no result.');
  });

  it('renders stale result state with alert role and allows retry', () => {
    const onRetry = vi.fn();
    render(<JobStatusView status={{ phase: 'stale' }} onRetry={onRetry} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Discarded a stale result');
    expect(screen.getByRole('button', { name: 'Retry job' })).toBeInTheDocument();
  });

  it('formats job ID with break-all for narrow 390px mobile screens', () => {
    render(
      <JobStatusView
        status={{
          jobId: 'very-long-unique-job-uuid-12345678-abcdef-999999999',
          phase: 'running',
          progressPercent: 10,
        }}
      />
    );
    const display = screen.getByTestId('job-id-display');
    expect(display).toHaveClass('break-all');
  });
});
