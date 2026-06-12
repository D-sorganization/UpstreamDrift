/**
 * Tests for the DiagnosticsPanel desktop-parity rendering (issue #7458).
 *
 * Verifies the panel works in BROWSER mode (no Tauri), renders the full
 * diagnostics report and integrations health with the shared status taxonomy
 * (healthy / configured / warning / error / unconfigured), and exposes
 * refresh + copy-as-markdown controls.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DiagnosticsPanel } from './DiagnosticsPanel';
import type {
  FullDiagnosticsReport,
  IntegrationsHealthReport,
} from '@/api/diagnostics';

vi.mock('@/api/backend', () => ({
  isTauri: () => false,
  startBackend: vi.fn(),
  stopBackend: vi.fn(),
  getDiagnostics: vi.fn().mockResolvedValue({
    backend: { running: true, pid: 123, port: 8001, error: null },
    python_found: true,
    python_version: '3.12.0',
    repo_root: '/repo',
    local_server_found: true,
  }),
  getApiBase: () => '',
}));

const fullReport: FullDiagnosticsReport = {
  summary: {
    total_checks: 3,
    passed: 1,
    failed: 1,
    warnings: 1,
    status: 'degraded',
    timestamp: '2026-06-12T00:00:00Z',
    expected_tiles: 17,
  },
  categories: ['engine_availability', 'asset_files', 'models_yaml'],
  checks: [
    {
      name: 'engine_availability',
      status: 'pass',
      message: '5/8 engines installed',
      details: {},
      duration_ms: 12,
    },
    {
      name: 'asset_files',
      status: 'warning',
      message: 'Missing 2 asset files',
      details: {},
      duration_ms: 3,
    },
    {
      name: 'models_yaml',
      status: 'fail',
      message: 'models.yaml is empty',
      details: {},
      duration_ms: 1,
    },
  ],
  recommendations: ['Restore missing asset files in src/launchers/assets/'],
};

const integrationsReport: IntegrationsHealthReport = {
  generated_at: '2026-06-12T00:00:00Z',
  records: [
    {
      kind: 'cli',
      name: 'claude',
      status: 'healthy',
      last_checked: '2026-06-12T00:00:00Z',
      last_error: null,
      detail: '/usr/bin/claude',
    },
    {
      kind: 'api',
      name: 'anthropic',
      status: 'configured',
      last_checked: '2026-06-12T00:00:00Z',
      last_error: null,
      detail: 'ANTHROPIC_API_KEY is set',
    },
    {
      kind: 'mcp',
      name: 'mcp_servers.json',
      status: 'warning',
      last_checked: '2026-06-12T00:00:00Z',
      last_error: 'No servers defined in config',
      detail: null,
    },
    {
      kind: 'mcp',
      name: 'broken_server',
      status: 'error',
      last_checked: '2026-06-12T00:00:00Z',
      last_error: 'parse failure',
      detail: null,
    },
    {
      kind: 'api',
      name: 'openai',
      status: 'unconfigured',
      last_checked: '2026-06-12T00:00:00Z',
      last_error: 'Environment variable OPENAI_API_KEY is not configured',
      detail: 'OPENAI_API_KEY is not set',
    },
  ],
  markdown: '# Integration Health\n| cli | claude | healthy |',
};

const apiFetchMock = vi.fn((path: string) => {
  if (path === '/api/v1/diagnostics/full') {
    return Promise.resolve(fullReport);
  }
  if (path === '/api/v1/integrations/health') {
    return Promise.resolve(integrationsReport);
  }
  // /api/engines backend-health probe
  return Promise.resolve([]);
});

vi.mock('@/api/fetch', () => ({
  apiFetch: (path: string) => apiFetchMock(path),
}));

async function openPanel() {
  render(<DiagnosticsPanel />);
  fireEvent.click(screen.getByLabelText('Toggle diagnostics panel'));
  await waitFor(() => {
    expect(screen.getByTestId('full-diagnostics')).toBeInTheDocument();
  });
}

beforeEach(() => {
  apiFetchMock.mockClear();
});

describe('DiagnosticsPanel (browser mode, issue #7458)', () => {
  it('fetches both reports from the API in browser mode', async () => {
    await openPanel();
    const paths = apiFetchMock.mock.calls.map(([path]) => path);
    expect(paths).toContain('/api/v1/diagnostics/full');
    expect(paths).toContain('/api/v1/integrations/health');
  });

  it('renders every check category served by the API', async () => {
    await openPanel();
    for (const check of fullReport.checks) {
      expect(screen.getByText(check.name)).toBeInTheDocument();
    }
  });

  it.each([
    ['pass', 'engine_availability'],
    ['warning', 'asset_files'],
    ['fail', 'models_yaml'],
  ])('renders check status %s', async (status) => {
    await openPanel();
    const matches = screen.getAllByText(status);
    expect(matches.length).toBeGreaterThan(0);
  });

  it.each([
    ['healthy', 'claude'],
    ['configured', 'anthropic'],
    ['warning', 'mcp_servers.json'],
    ['error', 'broken_server'],
    ['unconfigured', 'openai'],
  ])('renders integration status %s for %s', async (status, name) => {
    await openPanel();
    expect(screen.getByTestId('integrations-health')).toBeInTheDocument();
    expect(screen.getByText(`[${name === 'claude' ? 'cli' : name === 'anthropic' || name === 'openai' ? 'api' : 'mcp'}] ${name}`)).toBeInTheDocument();
    const matches = screen.getAllByText(status);
    expect(matches.length).toBeGreaterThan(0);
  });

  it('shows recommendations from the report', async () => {
    await openPanel();
    expect(
      screen.getByText(/Restore missing asset files/)
    ).toBeInTheDocument();
  });

  it('has a refresh button that refetches both reports', async () => {
    await openPanel();
    apiFetchMock.mockClear();
    fireEvent.click(screen.getByText('Refresh'));
    await waitFor(() => {
      const paths = apiFetchMock.mock.calls.map(([path]) => path);
      expect(paths).toContain('/api/v1/diagnostics/full');
      expect(paths).toContain('/api/v1/integrations/health');
    });
  });

  it('copies a markdown report to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    await openPanel();
    fireEvent.click(screen.getByText('Copy Markdown'));
    await waitFor(() => {
      expect(writeText).toHaveBeenCalledTimes(1);
    });
    const markdown = writeText.mock.calls[0][0] as string;
    expect(markdown).toContain('# UpstreamDrift Diagnostics');
    expect(markdown).toContain('engine_availability');
    expect(markdown).toContain('# Integration Health');
  });

  it('does not render Tauri-only lifecycle controls in browser mode', async () => {
    await openPanel();
    expect(screen.queryByText('Start')).not.toBeInTheDocument();
    expect(screen.queryByText('Stop')).not.toBeInTheDocument();
  });
});
