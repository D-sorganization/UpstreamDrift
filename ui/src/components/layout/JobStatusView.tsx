import type { ReactNode } from 'react';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

export type JobPhase =
  | 'idle'
  | 'submitting'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'stale';

export interface JobStatusData {
  jobId?: string | null;
  phase: JobPhase;
  progressPercent?: number | null;
  stage?: string | null;
  error?: string | null;
  label?: string;
  shotsPlanned?: number | null;
}

export interface JobStatusViewProps {
  status: JobStatusData;
  onCancel?: () => void;
  onRetry?: () => void;
  children?: ReactNode;
  className?: string;
}

const TONE_MAP: Record<JobPhase, BadgeTone> = {
  idle: 'neutral',
  submitting: 'primary',
  queued: 'warning',
  running: 'primary',
  completed: 'success',
  failed: 'danger',
  cancelled: 'neutral',
  stale: 'warning',
};

const TITLE_MAP: Record<JobPhase, string> = {
  idle: 'Idle',
  submitting: 'Submitting…',
  queued: 'Queued…',
  running: 'Solving…',
  completed: 'Completed',
  failed: 'Job failed',
  cancelled: 'Cancelled',
  stale: 'Stale result',
};

/**
 * Mobile-friendly, responsive long-running job status card (#14268).
 *
 * Designed for mobile viewports (390px phone screens) with touch-friendly 44px
 * action buttons, accessible ARIA roles (status/alert/progressbar), monospaced
 * job ID with break-all, and resilient reconnection/retry controls.
 */
export function JobStatusView({
  status,
  onCancel,
  onRetry,
  children,
  className = '',
}: JobStatusViewProps) {
  const { jobId, phase, progressPercent, stage, error, label, shotsPlanned } = status;

  if (phase === 'idle') return null;

  if (phase === 'submitting') {
    return (
      <div
        role="status"
        className={`flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-800 p-3 text-sm text-gray-200 ${className}`.trim()}
      >
        <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-blue-400 border-t-transparent" />
        <span>Submitting…</span>
      </div>
    );
  }

  const percent = Math.round(Math.min(100, Math.max(0, progressPercent ?? 0)));
  const progressText = shotsPlanned
    ? `${percent}% of ${shotsPlanned} shots`
    : `${percent}% · ${stage ?? 'working'}`;
  const ariaLabel = label ?? 'Job progress';

  return (
    <section
      aria-label="Job status"
      className={`flex flex-col gap-3 rounded-lg border border-gray-700 bg-gray-800 p-3 text-sm text-gray-200 shadow-sm sm:max-w-xl ${className}`.trim()}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Badge tone={TONE_MAP[phase]}>{TITLE_MAP[phase]}</Badge>
        </div>
        {jobId && (
          <span
            data-testid="job-id-display"
            className="text-xs font-mono text-gray-400 break-all select-all"
            title={`Job ID: ${jobId}`}
          >
            {jobId}
          </span>
        )}
      </div>

      {(phase === 'running' || phase === 'queued') && (
        <div className="flex flex-col gap-1.5">
          <progress
            aria-label={ariaLabel}
            aria-valuetext={progressText}
            aria-valuenow={percent}
            aria-valuemin={0}
            aria-valuemax={100}
            className="h-2.5 w-full overflow-hidden rounded-full bg-gray-900 [&::-webkit-progress-bar]:bg-gray-900 [&::-webkit-progress-value]:bg-blue-600 [&::-moz-progress-bar]:bg-blue-600"
            max={100}
            value={percent}
          />
          <p className="text-xs text-gray-400">{progressText}</p>
        </div>
      )}

      {phase === 'failed' && (
        <p role="alert" className="text-sm text-red-300">
          {error ?? 'Job failed: no reason was given'}
        </p>
      )}

      {phase === 'cancelled' && (
        <p role="status" className="text-sm text-gray-300">
          Job cancelled; it produced no result.
        </p>
      )}

      {phase === 'stale' && (
        <p role="alert" className="text-sm text-red-300">
          {error ?? 'Discarded a stale result: it does not answer the current submission. Run it again.'}
        </p>
      )}

      {children}

      <div className="flex flex-wrap items-center gap-2 pt-1">
        {(phase === 'running' || phase === 'queued') && onCancel && (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={onCancel}
            className="w-full sm:w-auto"
          >
            Cancel job
          </Button>
        )}
        {(phase === 'failed' || phase === 'stale') && onRetry && (
          <Button
            type="button"
            variant="primary"
            size="sm"
            onClick={onRetry}
            className="w-full sm:w-auto"
          >
            Retry job
          </Button>
        )}
      </div>
    </section>
  );
}
