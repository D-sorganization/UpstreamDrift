/**
 * AboutModal — web parity for the desktop About dialog (issue #7459).
 *
 * Shows backend version info from GET /api/v1/about (app version resolved
 * via the same chain as the desktop dialog), the frontend version from
 * ui/package.json (imported at build time), and the same support links as
 * the desktop dialog (user guide, report a bug).
 */

import { useEffect, useState } from 'react';
import { X, ExternalLink, Loader2, AlertTriangle } from 'lucide-react';
import { fetchAboutInfo, type AboutInfo } from '@/api/about';
import packageJson from '../../../package.json';

export const FRONTEND_VERSION: string = packageJson.version;

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-1 text-sm">
      <span className="text-gray-400">{label}</span>
      <span className="text-gray-100 font-mono text-right break-all">{value}</span>
    </div>
  );
}

export function AboutModal({ isOpen, onClose }: Props) {
  const [info, setInfo] = useState<AboutInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    fetchAboutInfo()
      .then((data) => {
        if (!cancelled) {
          setInfo(data);
          setError(null);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="About UpstreamDrift"
        className="w-full max-w-md rounded-xl border border-gray-700 bg-gray-900 p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">UpstreamDrift</h2>
            <p className="text-xs text-gray-500">
              Biomechanical motion analysis platform
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close about dialog"
            className="rounded-md p-1 text-gray-400 hover:bg-white/10 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <div className="mb-4 rounded-lg border border-gray-700/60 bg-gray-800/50 p-4">
          <InfoRow label="Frontend" value={FRONTEND_VERSION} />
          {error && (
            <div className="mt-2 flex items-center gap-2 text-sm text-amber-300" role="alert">
              <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />
              <span>Backend info unavailable: {error}</span>
            </div>
          )}
          {!error && !info && (
            <div className="mt-2 flex items-center gap-2 text-sm text-gray-400" role="status">
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              Loading backend info...
            </div>
          )}
          {info && (
            <>
              <InfoRow label="Backend" value={info.app_version} />
              <InfoRow label="Python" value={info.python_version} />
              <InfoRow label="Platform" value={info.platform} />
              {info.git_commit && (
                <InfoRow label="Commit" value={info.git_commit.slice(0, 12)} />
              )}
              {Object.entries(info.dependencies)
                .filter(([, v]) => v !== 'not installed')
                .map(([name, version]) => (
                  <InfoRow key={name} label={name} value={version} />
                ))}
            </>
          )}
        </div>

        <div className="flex items-center gap-4 text-sm">
          <a
            href={info?.links.user_guide ?? 'https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/user_guide/getting_started.md'}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-blue-300 hover:text-blue-200 hover:underline"
          >
            <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
            User Guide
          </a>
          <a
            href={info?.links.report_bug ?? 'https://github.com/D-sorganization/UpstreamDrift/issues'}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-blue-300 hover:text-blue-200 hover:underline"
          >
            <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
            Report a Bug
          </a>
        </div>

        <p className="mt-4 text-xs text-gray-600">
          Copyright 2024-2026 UpstreamDrift Contributors.
        </p>
      </div>
    </div>
  );
}
