import { displayStatus, type SpineStudyPayload } from '@/api/spineViewer';

/**
 * Always-visible scientific status and distribution source (SPN-15c, #12815).
 * Fail-closed: anything but the exact status "qualified" reads UNQUALIFIED.
 */
export function SpineStatusBanner({ study }: { study: SpineStudyPayload }) {
  const status = displayStatus(study.scientific_status);
  const qualified = status === 'qualified';
  const source = study.distribution_source?.trim() || 'not recorded';
  return (
    <div
      role="status"
      data-testid="spine-status-banner"
      data-status={status}
      className={
        qualified
          ? 'border-b border-green-800 bg-green-900/30 px-6 py-3 text-sm text-green-200'
          : 'border-b border-amber-700 bg-amber-900/30 px-6 py-3 text-sm text-amber-200'
      }
    >
      <div className="font-semibold">
        {qualified ? 'QUALIFIED' : 'UNQUALIFIED'}
        <span className="ml-2 font-normal">
          (run verdict: {study.receipt_status})
        </span>
      </div>
      <div>
        Distribution source:{' '}
        <span data-testid="spine-distribution-source">{source}</span>
      </div>
      {study.status_reasons.length > 0 && (
        <ul className="mt-1 list-disc pl-5">
          {study.status_reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
