import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';

import { CharacterSpecPanel } from './CharacterSpecPanel';

const PRESET = {
  id: 'golfer_pro',
  name: 'Golfer Pro',
  description: 'Professional golfer body type',
  category: 'golf',
  parameters: {
    stature_m: 1.83,
    mass_kg: 82,
    trunk_scale: 1.05,
    arm_scale: 1.02,
    shoulder_scale: 1.08,
    grip_roll_deg: 0,
    club: 'driver',
  },
  provenance: 'Anthropometric tables for elite male golfers.',
  limitations: 'Only validated for adult male anthropometry.',
};

const PREVIEW_RESPONSE = {
  spec_sha256: 'abc123',
  stature_m: 1.75,
  bodies: [
    { name: 'torso', mass_kg: 30.5 },
    { name: 'pelvis', mass_kg: 10.2 },
  ],
  joints: [{ name: 'hip', parent: 'pelvis', child: 'torso' }],
};

const BUILD_SUMMARY_RESPONSE = {
  preset: null,
  spec_sha256: 'abc123',
  schema_version: '1.0',
  qualification: 'unqualified',
  bodies: 2,
  joints: 1,
  coordinates: 3,
  total_mass_kg: 40.7,
  club: 'driver',
};

function jsonResponse(body: unknown, init?: { ok?: boolean; status?: number; statusText?: string; headers?: Record<string, string> }) {
  return {
    ok: init?.ok ?? true,
    status: init?.status ?? 200,
    statusText: init?.statusText ?? 'OK',
    headers: new Headers(init?.headers ?? {}),
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
  };
}

describe('CharacterSpecPanel', () => {
  const mockFetch = vi.fn();
  let clickSpy: MockInstance;

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
    vi.stubGlobal('fetch', mockFetch);
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:mock-url'),
      revokeObjectURL: vi.fn(),
    });
    clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    // Default: presets endpoint returns one preset; everything else 404s
    // unless a test overrides mockFetch.mockImplementation below.
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/character-builder/presets')) {
        return Promise.resolve(jsonResponse({ presets: [PRESET] }));
      }
      return Promise.resolve(jsonResponse({ detail: 'not stubbed' }, { ok: false, status: 404 }));
    });
  });

  afterEach(() => {
    clickSpy.mockRestore();
  });

  it('loads presets on mount and populates the select', async () => {
    render(<CharacterSpecPanel />);

    expect(await screen.findByRole('option', { name: 'Golfer Pro' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Defaults (no preset)' })).toBeInTheDocument();
  });

  it('fills slider values from the selected preset', async () => {
    render(<CharacterSpecPanel />);
    await screen.findByRole('option', { name: 'Golfer Pro' });

    const select = screen.getByLabelText(/Preset/i) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'golfer_pro' } });

    expect((screen.getByLabelText(/Stature/i) as HTMLInputElement).value).toBe('1.83');
    expect((screen.getByLabelText(/Mass/i) as HTMLInputElement).value).toBe('82');
    expect((screen.getByLabelText(/Trunk Length Scale/i) as HTMLInputElement).value).toBe('1.05');
    expect((screen.getByLabelText(/Arm Length Scale/i) as HTMLInputElement).value).toBe('1.02');
    expect((screen.getByLabelText(/Shoulder Breadth Scale/i) as HTMLInputElement).value).toBe('1.08');
    expect(screen.getByText('Professional golfer body type')).toBeInTheDocument();
    expect(screen.getByText('Only validated for adult male anthropometry.')).toBeInTheDocument();
  });

  it('previews: posts the current parameters to /preview and /build and renders the result', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/character-builder/presets')) {
        return Promise.resolve(jsonResponse({ presets: [PRESET] }));
      }
      if (url.includes('/character-builder/preview')) {
        return Promise.resolve(jsonResponse(PREVIEW_RESPONSE));
      }
      if (url.includes('/character-builder/build')) {
        return Promise.resolve(jsonResponse(BUILD_SUMMARY_RESPONSE));
      }
      return Promise.resolve(jsonResponse({ detail: 'not stubbed' }, { ok: false, status: 404 }));
    });

    render(<CharacterSpecPanel />);
    await screen.findByRole('option', { name: 'Golfer Pro' });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Preview/i }));
    });

    const expectedBody = JSON.stringify({
      preset: null,
      stature_m: 1.75,
      mass_kg: 75,
      trunk_scale: 1.0,
      arm_scale: 1.0,
      shoulder_scale: 1.0,
      club: 'driver',
    });
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/character-builder/preview',
      expect.objectContaining({ method: 'POST', body: expectedBody }),
    );
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/character-builder/build',
      expect.objectContaining({ method: 'POST', body: expectedBody }),
    );

    expect(await screen.findByText('40.70 kg')).toBeInTheDocument();
    expect(screen.getByText('torso')).toBeInTheDocument();
    expect(screen.getByText('30.50 kg')).toBeInTheDocument();
    expect(screen.getByText('pelvis')).toBeInTheDocument();
  });

  it('shows the API detail message on a 422 preview error', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/character-builder/presets')) {
        return Promise.resolve(jsonResponse({ presets: [PRESET] }));
      }
      if (url.includes('/character-builder/preview')) {
        return Promise.resolve(
          jsonResponse(
            { detail: 'stature_m must be between 1.2 and 2.3' },
            { ok: false, status: 422, statusText: 'Unprocessable Entity' },
          ),
        );
      }
      return Promise.resolve(jsonResponse({ detail: 'not stubbed' }, { ok: false, status: 404 }));
    });

    render(<CharacterSpecPanel />);
    await screen.findByRole('option', { name: 'Golfer Pro' });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Preview/i }));
    });

    expect(
      await screen.findByText('stature_m must be between 1.2 and 2.3'),
    ).toBeInTheDocument();
  });

  it('exports URDF: posts to /export/urdf and triggers a download', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/character-builder/presets')) {
        return Promise.resolve(jsonResponse({ presets: [PRESET] }));
      }
      if (url.includes('/character-builder/export/urdf')) {
        return Promise.resolve(
          jsonResponse('<robot></robot>', {
            headers: {
              'Content-Disposition': 'attachment; filename="custom_abc123.urdf"',
              'Content-Type': 'text/xml',
            },
          }),
        );
      }
      return Promise.resolve(jsonResponse({ detail: 'not stubbed' }, { ok: false, status: 404 }));
    });

    render(<CharacterSpecPanel />);
    await screen.findByRole('option', { name: 'Golfer Pro' });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^URDF$/i }));
    });

    await waitFor(() =>
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/character-builder/export/urdf',
        expect.objectContaining({ method: 'POST' }),
      ),
    );
    expect(clickSpy).toHaveBeenCalled();
  });
});
