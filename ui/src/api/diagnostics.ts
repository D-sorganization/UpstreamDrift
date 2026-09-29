/**
 * Diagnostics parity API client (issue #7458).
 *
 * Fetches the desktop-grade diagnostics report and the integrations-health
 * probe results from the backend. Works in both Tauri and browser modes —
 * no Tauri APIs involved.
 *
 * The category set is served by the backend (derived from the single
 * `DIAGNOSTIC_CHECKS` enumeration in `src/launchers/launcher_diagnostics.py`);
 * the UI renders whatever categories arrive and never hard-codes its own list.
 */

import { apiFetch } from './fetch';

/** Status values for launcher diagnostic checks (desktop taxonomy). */
export type CheckStatus = 'pass' | 'fail' | 'warning' | 'info';

export interface DiagnosticCheck {
  name: string;
  status: CheckStatus | string;
  message: string;
  details: Record<string, unknown>;
  duration_ms: number;
}

export interface DiagnosticsSummary {
  total_checks: number;
  passed: number;
  failed: number;
  warnings: number;
  status: 'healthy' | 'degraded' | string;
  timestamp: string;
  expected_tiles: number;
}

export interface FullDiagnosticsReport {
  summary: DiagnosticsSummary;
  categories: string[];
  checks: DiagnosticCheck[];
  recommendations: string[];
}

/** Status taxonomy shared with the desktop Integrations Health panel. */
export type IntegrationStatus =
  | 'healthy'
  | 'configured'
  | 'warning'
  | 'error'
  | 'unconfigured'
  | 'unknown';

export interface IntegrationRecord {
  kind: string;
  name: string;
  status: IntegrationStatus | string;
  last_checked: string | null;
  last_error: string | null;
  detail: string | null;
}

export interface IntegrationsHealthReport {
  generated_at: string;
  records: IntegrationRecord[];
  /** Server-rendered markdown (secrets already redacted server-side). */
  markdown: string;
}

export function getFullDiagnostics(): Promise<FullDiagnosticsReport> {
  return apiFetch<FullDiagnosticsReport>('/api/v1/diagnostics/full');
}

export function getIntegrationsHealth(): Promise<IntegrationsHealthReport> {
  return apiFetch<IntegrationsHealthReport>('/api/v1/integrations/health');
}

/** Tailwind text-color class for a status value (both taxonomies). */
export function statusColorClass(status: string): string {
  switch (status) {
    case 'pass':
    case 'healthy':
      return 'text-green-400';
    case 'configured':
      return 'text-blue-400';
    case 'fail':
    case 'error':
      return 'text-red-400';
    case 'warning':
      return 'text-yellow-400';
    case 'unconfigured':
      return 'text-gray-400';
    case 'info':
      return 'text-blue-400';
    default:
      return 'text-gray-400';
  }
}

/**
 * Build a copy-as-markdown report combining both server reports.
 * The integrations section reuses the server-rendered (already redacted)
 * markdown verbatim.
 */
export function buildDiagnosticsMarkdown(
  report: FullDiagnosticsReport | null,
  integrations: IntegrationsHealthReport | null,
): string {
  const lines: string[] = ['# UpstreamDrift Diagnostics', ''];

  if (report) {
    const s = report.summary;
    lines.push(
      `**Status:** ${s.status} — ${s.passed} passed, ${s.failed} failed, ${s.warnings} warnings (${s.total_checks} checks, ${s.timestamp})`,
      '',
      '| Check | Status | Message |',
      '|-------|--------|---------|',
    );
    for (const check of report.checks) {
      lines.push(`| ${check.name} | ${check.status} | ${check.message} |`);
    }
    if (report.recommendations.length > 0) {
      lines.push('', '## Recommendations', '');
      for (const rec of report.recommendations) {
        lines.push(`- ${rec}`);
      }
    }
    lines.push('');
  }

  if (integrations) {
    lines.push(integrations.markdown);
  }

  return lines.join('\n');
}
