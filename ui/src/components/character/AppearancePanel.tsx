import { useCallback, useEffect, useMemo, useState } from 'react';
import { getApiBase } from '@/api/backend';
import { apiFetch } from '@/api/fetch';
import {
  errorMessage,
  type CharacterSpecRequestBody,
} from './characterSpecTypes';
import {
  DEFAULT_CLOTHING,
  DEFAULT_CLUB_FINISH,
  DEFAULT_HEADWEAR,
  DEFAULT_SKIN_TONE,
} from './appearanceDefaults.generated';
import { DEFAULT_GLOVE, GLOVE_CHOICES, hexToRgb, type ColourPicks, type GloveChoice } from './appearanceDocument';
import {
  appearancePalette,
  rgbaToHex,
  type AppearanceLibrary,
  type AppearancePalette,
  type AppearancePicks,
} from './appearanceTypes';

interface AppearancePanelProps {
  onPaletteChange?: (palette: AppearancePalette) => void;
  character?: CharacterSpecRequestBody | null;
}

/** A labelled `<select>`, with an optional decorative colour swatch. */
function LabeledSelect({
  id,
  label,
  value,
  onChange,
  options,
  swatchColor,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  options: string[];
  swatchColor?: string;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-semibold text-gray-300 mb-1">
        {label}
      </label>
      <div className="flex items-center gap-2">
        {swatchColor !== undefined && (
          <span
            aria-hidden="true"
            className="inline-block w-4 h-4 rounded-full border border-gray-600 flex-shrink-0"
            style={{ backgroundColor: swatchColor }}
          />
        )}
        <select
          id={id}
          value={value}
          onChange={onChange}
          className="w-full bg-gray-700 border-none text-gray-200 rounded px-2 py-1.5 text-sm focus:ring-1 focus:ring-blue-400"
        >
          {options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

const COLOUR_CONTROLS: ReadonlyArray<{
  key: keyof ColourPicks;
  id: string;
  label: string;
}> = [
  { key: 'skin_color', id: 'appearance-skin-color', label: 'Custom Skin Color' },
  { key: 'hair_color', id: 'appearance-hair-color', label: 'Hair Color' },
  { key: 'top_color', id: 'appearance-top-color', label: 'Shirt Color' },
  { key: 'bottom_color', id: 'appearance-bottom-color', label: 'Trousers/Shorts Color' },
];

/** A labelled `<input type="color">` with a Reset button that clears the custom colour. */
function ColourControl({
  id,
  label,
  value,
  isCustom,
  disabled,
  onChange,
  onReset,
}: {
  id: string;
  label: string;
  value: string;
  isCustom: boolean;
  disabled?: boolean;
  onChange: (hex: string) => void;
  onReset: () => void;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-semibold text-gray-300 mb-1">
        {label}
      </label>
      <div className="flex items-center gap-2">
        <input
          id={id}
          type="color"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          className="h-7 w-10 bg-gray-700 border-none rounded cursor-pointer disabled:opacity-40"
        />
        <button
          type="button"
          aria-label={`Reset ${label}`}
          disabled={!isCustom}
          onClick={onReset}
          className="text-xs text-gray-300 hover:text-white disabled:opacity-40"
        >
          Reset
        </button>
      </div>
    </div>
  );
}

/** `preferred` when the library offers it, else the library's first option. */
function pickDefault(options: string[], preferred: string): string {
  return options.includes(preferred) ? preferred : (options[0] ?? preferred);
}

/**
 * Initial picks: the `AppearanceDocument` defaults in
 * `src/shared/python/model_appearance/schema.py`, so an untouched export
 * matches the document the desktop and engines use by default.
 */
function defaultPicks(library: AppearanceLibrary): AppearancePicks {
  const skin_tone = pickDefault(library.skin_tones, DEFAULT_SKIN_TONE);
  const clothing = pickDefault(Object.keys(library.clothing), DEFAULT_CLOTHING);
  const club_finish = pickDefault(library.club_finishes, DEFAULT_CLUB_FINISH);
  const headwear = pickDefault(library.headwear, DEFAULT_HEADWEAR);
  const ground_material = pickDefault(library.ground_materials, 'turf');
  const headwear_material =
    headwear === 'none'
      ? undefined
      : library.headwear_default_material[headwear];
  return {
    skin_tone,
    clothing,
    club_finish,
    headwear,
    headwear_material,
    ground_material,
    glove: DEFAULT_GLOVE,
  };
}

/**
 * Appearance panel (CMB-7c, #11658): skin tone, clothing, club finish,
 * headwear and ground material, plus an Export Appearance download.
 *
 * Loads `/api/character-builder/appearance/library` on mount and reports the
 * derived preview palette to the parent via `onPaletteChange` so
 * `CharacterPreview` can recolour without this component reaching into it.
 *
 * When `character` (the spec panel's `CharacterSpecRequestBody`, CMB-7d,
 * #11658) is supplied, the export is bound to that spec character: the
 * server compiles it, stamps `spec_sha256` on the document and names the
 * file from the preset.
 */
export function AppearancePanel({
  onPaletteChange,
  character,
}: AppearancePanelProps) {
  const [library, setLibrary] = useState<AppearanceLibrary | null>(null);
  const [libraryError, setLibraryError] = useState<string | null>(null);
  const [picks, setPicks] = useState<AppearancePicks | null>(null);

  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch<AppearanceLibrary>('/api/character-builder/appearance/library')
      .then((data) => {
        if (cancelled) return;
        const initial = defaultPicks(data);
        setLibrary(data);
        setPicks(initial);
        onPaletteChange?.(appearancePalette(data, initial));
      })
      .catch((err: unknown) => {
        if (!cancelled) setLibraryError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [onPaletteChange]);

  const palette = useMemo(
    () => (library && picks ? appearancePalette(library, picks) : null),
    [library, picks],
  );

  const applyPicks = useCallback(
    (next: AppearancePicks) => {
      setPicks(next);
      if (library) onPaletteChange?.(appearancePalette(library, next));
    },
    [library, onPaletteChange],
  );

  const handleSkinToneChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (!picks) return;
    applyPicks({ ...picks, skin_tone: e.target.value });
  };

  const handleClothingChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (!picks) return;
    applyPicks({ ...picks, clothing: e.target.value });
  };

  const handleGloveChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (!picks) return;
    applyPicks({ ...picks, glove: e.target.value as GloveChoice });
  };

  const handleClubFinishChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (!picks) return;
    applyPicks({ ...picks, club_finish: e.target.value });
  };

  const handleHeadwearChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (!picks || !library) return;
    const headwear = e.target.value;
    const headwear_material =
      headwear === 'none'
        ? undefined
        : library.headwear_default_material[headwear];
    // A hair colour only applies to hair; drop it rather than send a request
    // the server rejects.
    const hair_color = headwear === 'hair' ? picks.hair_color : undefined;
    applyPicks({ ...picks, headwear, headwear_material, hair_color });
  };

  const handleColourChange = (key: keyof ColourPicks, hex: string) => {
    if (!picks) return;
    hexToRgb(hex); // <input type="color"> always yields #rrggbb
    applyPicks({ ...picks, [key]: hex });
  };

  const handleColourReset = (key: keyof ColourPicks) => {
    if (!picks) return;
    applyPicks({ ...picks, [key]: undefined });
  };

  /** Colour shown by a picker: the custom pick, else the current library colour. */
  const colourValue = (key: keyof ColourPicks): string => {
    const custom = picks?.[key];
    if (custom !== undefined) return custom;
    if (key === 'skin_color') return palette?.head ?? '#000000';
    if (key === 'top_color') return palette?.trunk ?? '#000000';
    if (key === 'bottom_color') return palette?.thigh ?? '#000000';
    const hairName = picks?.headwear_material;
    const hair = hairName ? library?.materials[hairName] : undefined;
    return hair ? rgbaToHex(hair.base_color) : '#000000';
  };

  const handleGroundChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (!picks) return;
    applyPicks({ ...picks, ground_material: e.target.value });
  };

  const handleExport = useCallback(async () => {
    if (!picks) return;
    setExporting(true);
    setExportError(null);
    try {
      const response = await fetch(
        `${getApiBase()}/api/character-builder/appearance/export`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(character ? { ...picks, character } : picks),
        },
      );
      if (!response.ok) {
        let detail: string | undefined;
        try {
          const body = (await response.json()) as { detail?: string };
          detail = body.detail;
        } catch {
          // Body was not JSON — fall through to the generic message below.
        }
        throw new Error(
          detail ?? `HTTP ${response.status} ${response.statusText}`,
        );
      }
      const text = await response.text();
      const disposition = response.headers.get('Content-Disposition') ?? '';
      const match = /filename="?([^";]+)"?/.exec(disposition);
      const filename = match ? match[1] : 'character.appearance.json';
      const mediaType =
        response.headers.get('Content-Type') ?? 'application/json';
      const blob = new Blob([text], { type: mediaType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(errorMessage(err));
    } finally {
      setExporting(false);
    }
  }, [picks, character]);

  if (libraryError) {
    return (
      <div className="space-y-4">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Appearance
        </h3>
        <div className="text-xs text-red-400 bg-red-950/30 p-2.5 rounded border border-red-900/50">
          Failed to load appearance library: {libraryError}
        </div>
      </div>
    );
  }

  if (!library || !picks) {
    return (
      <div className="space-y-4">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Appearance
        </h3>
        <div className="text-xs text-gray-400">
          Loading appearance options...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Appearance
      </h3>

      <LabeledSelect
        id="appearance-skin-tone-select"
        label="Skin Tone"
        value={picks.skin_tone}
        onChange={handleSkinToneChange}
        options={library.skin_tones}
        swatchColor={palette?.head}
      />

      <LabeledSelect
        id="appearance-clothing-select"
        label="Clothing"
        value={picks.clothing}
        onChange={handleClothingChange}
        options={Object.keys(library.clothing)}
        swatchColor={palette?.trunk}
      />

      <LabeledSelect
        id="appearance-club-finish-select"
        label="Club Finish"
        value={picks.club_finish}
        onChange={handleClubFinishChange}
        options={library.club_finishes}
      />

      <LabeledSelect
        id="appearance-glove-select"
        label="Glove"
        value={picks.glove ?? DEFAULT_GLOVE}
        onChange={handleGloveChange}
        options={library.glove ?? [...GLOVE_CHOICES]}
      />

      <LabeledSelect
        id="appearance-headwear-select"
        label="Headwear"
        value={picks.headwear}
        onChange={handleHeadwearChange}
        options={library.headwear}
      />

      {COLOUR_CONTROLS.map(({ key, id, label }) => (
        <ColourControl
          key={key}
          id={id}
          label={label}
          value={colourValue(key)}
          isCustom={picks[key] !== undefined}
          disabled={key === 'hair_color' && picks.headwear !== 'hair'}
          onChange={(hex) => handleColourChange(key, hex)}
          onReset={() => handleColourReset(key)}
        />
      ))}

      <LabeledSelect
        id="appearance-ground-select"
        label="Ground"
        value={picks.ground_material}
        onChange={handleGroundChange}
        options={library.ground_materials}
      />

      <button
        type="button"
        onClick={() => void handleExport()}
        disabled={exporting}
        className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 text-white rounded py-2 text-sm font-semibold transition-colors"
      >
        {exporting ? 'Exporting…' : 'Export Appearance'}
      </button>
      {character && (
        <p
          data-testid="appearance-spec-binding"
          className="text-xs text-gray-400"
        >
          Bound to the current spec character (preset:{' '}
          {character.preset ?? 'defaults'})
        </p>
      )}
      {exportError && (
        <div className="text-xs text-red-400 bg-red-950/30 p-2.5 rounded border border-red-900/50">
          {exportError}
        </div>
      )}
    </div>
  );
}
