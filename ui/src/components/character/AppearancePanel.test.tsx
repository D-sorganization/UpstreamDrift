import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';

import { AppearancePanel } from './AppearancePanel';
import { rgbaToHex, type AppearanceLibrary } from './appearanceTypes';

const LIBRARY: AppearanceLibrary = {
  schema_version: 'appearance-v1',
  skin_tones: ['skin_light', 'skin_medium'],
  clothing: {
    golf_polo_shorts: {
      torso: 'polo_navy',
      upper_arm: 'polo_navy',
      thigh: 'shorts_khaki',
    },
    golf_polo_trousers: {
      torso: 'polo_white',
      upper_arm: 'polo_white',
      thigh: 'trousers_charcoal',
      shin: 'trousers_charcoal',
    },
  },
  club_finishes: ['satin_steel', 'chrome'],
  headwear: ['none', 'hair', 'cap'],
  headwear_default_material: { hair: 'hair_brown', cap: 'cap_navy' },
  ground_materials: ['turf', 'studio_floor'],
  materials: {
    skin_light: { base_color: [0.93, 0.76, 0.66, 1.0], roughness: 0.55, metallic: 0 },
    skin_medium: { base_color: [0.8, 0.6, 0.46, 1.0], roughness: 0.55, metallic: 0 },
    polo_navy: { base_color: [0.16, 0.27, 0.55, 1.0], roughness: 0.85, metallic: 0 },
    polo_white: { base_color: [0.93, 0.93, 0.92, 1.0], roughness: 0.85, metallic: 0 },
    shorts_khaki: { base_color: [0.78, 0.7, 0.52, 1.0], roughness: 0.9, metallic: 0 },
    trousers_charcoal: { base_color: [0.3, 0.32, 0.35, 1.0], roughness: 0.9, metallic: 0 },
    hair_brown: { base_color: [0.2, 0.13, 0.08, 1.0], roughness: 0.8, metallic: 0 },
    cap_navy: { base_color: [0.12, 0.2, 0.45, 1.0], roughness: 0.85, metallic: 0 },
  },
};

function jsonResponse(
  body: unknown,
  init?: { ok?: boolean; status?: number; statusText?: string; headers?: Record<string, string> },
) {
  return {
    ok: init?.ok ?? true,
    status: init?.status ?? 200,
    statusText: init?.statusText ?? 'OK',
    headers: new Headers(init?.headers ?? {}),
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
  };
}

describe('AppearancePanel', () => {
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

    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/character-builder/appearance/library')) {
        return Promise.resolve(jsonResponse(LIBRARY));
      }
      return Promise.resolve(jsonResponse({ detail: 'not stubbed' }, { ok: false, status: 404 }));
    });
  });

  afterEach(() => {
    clickSpy.mockRestore();
  });

  it('loads the library on mount and populates the selects', async () => {
    render(<AppearancePanel />);

    expect(await screen.findByRole('option', { name: 'skin_light' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'golf_polo_shorts' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'golf_polo_trousers' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'satin_steel' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'turf' })).toBeInTheDocument();

    // Defaults mirror AppearanceDocument (schema.py), not the first option.
    expect((screen.getByLabelText(/Skin Tone/i) as HTMLSelectElement).value).toBe('skin_medium');
    expect((screen.getByLabelText(/Clothing/i) as HTMLSelectElement).value).toBe('golf_polo_shorts');
    expect((screen.getByLabelText(/Headwear/i) as HTMLSelectElement).value).toBe('hair');
  });

  it('reports the initial palette once the library loads', async () => {
    const onPaletteChange = vi.fn();
    render(<AppearancePanel onPaletteChange={onPaletteChange} />);

    await screen.findByRole('option', { name: 'skin_light' });

    expect(onPaletteChange).toHaveBeenCalledWith(
      expect.objectContaining({ trunk: rgbaToHex(LIBRARY.materials.polo_navy.base_color) }),
    );
  });

  it('changing clothing calls onPaletteChange with the expected trunk colour', async () => {
    const onPaletteChange = vi.fn();
    render(<AppearancePanel onPaletteChange={onPaletteChange} />);

    await screen.findByRole('option', { name: 'skin_light' });
    onPaletteChange.mockClear();

    const clothingSelect = screen.getByLabelText(/Clothing/i) as HTMLSelectElement;
    fireEvent.change(clothingSelect, { target: { value: 'golf_polo_trousers' } });

    expect(onPaletteChange).toHaveBeenCalledWith(
      expect.objectContaining({
        trunk: rgbaToHex(LIBRARY.materials.polo_white.base_color),
        shank: rgbaToHex(LIBRARY.materials.trousers_charcoal.base_color),
      }),
    );
  });

  it('exports the current picks and triggers a download', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/character-builder/appearance/library')) {
        return Promise.resolve(jsonResponse(LIBRARY));
      }
      if (url.includes('/character-builder/appearance/export')) {
        return Promise.resolve(
          jsonResponse(
            { schema_version: 'appearance-v1' },
            {
              headers: {
                'Content-Disposition': 'attachment; filename="golfer.appearance.json"',
                'Content-Type': 'application/json',
              },
            },
          ),
        );
      }
      return Promise.resolve(jsonResponse({ detail: 'not stubbed' }, { ok: false, status: 404 }));
    });

    render(<AppearancePanel />);
    await screen.findByRole('option', { name: 'skin_light' });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Export Appearance/i }));
    });

    const expectedBody = JSON.stringify({
      skin_tone: 'skin_medium',
      clothing: 'golf_polo_shorts',
      club_finish: 'satin_steel',
      headwear: 'hair',
      headwear_material: 'hair_brown',
      ground_material: 'turf',
    });

    await waitFor(() =>
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/character-builder/appearance/export',
        expect.objectContaining({ method: 'POST', body: expectedBody }),
      ),
    );
    expect(clickSpy).toHaveBeenCalled();
  });

  it('shows the API detail message on a 422 export error', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/character-builder/appearance/library')) {
        return Promise.resolve(jsonResponse(LIBRARY));
      }
      if (url.includes('/character-builder/appearance/export')) {
        return Promise.resolve(
          jsonResponse(
            { detail: 'name must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$' },
            { ok: false, status: 422, statusText: 'Unprocessable Entity' },
          ),
        );
      }
      return Promise.resolve(jsonResponse({ detail: 'not stubbed' }, { ok: false, status: 404 }));
    });

    render(<AppearancePanel />);
    await screen.findByRole('option', { name: 'skin_light' });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Export Appearance/i }));
    });

    expect(
      await screen.findByText('name must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$'),
    ).toBeInTheDocument();
  });
});
