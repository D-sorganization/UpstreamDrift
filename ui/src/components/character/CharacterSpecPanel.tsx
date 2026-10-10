import { useCallback, useEffect, useMemo, useState } from 'react';
import { getApiBase } from '@/api/backend';
import { apiFetch } from '@/api/fetch';
import { SpecSlider } from './SpecSlider';
import {
  DEFAULT_PARAMS,
  EXPORT_BUTTONS,
  errorMessage,
  type CharacterBuildSummary,
  type CharacterPresetSummary,
  type CharacterPreviewResponse,
  type ExportFormat,
  type SliderParams,
} from './characterSpecTypes';

/**
 * Spec-native Character Builder panel (CMB-7a, #11658).
 *
 * Drives the same `src/shared/python/humanoid_character_builder/spec_export.py`
 * endpoints the desktop tool uses (`src/tools/character_builder/gui.py`):
 * presets, preview/build and spec/URDF/MJCF/OpenSim export. The legacy
 * mesh-URDF `/character-builder/generate` flow in `CharacterBuilder.tsx`
 * is untouched.
 */
export function CharacterSpecPanel() {
  const [presets, setPresets] = useState<CharacterPresetSummary[]>([]);
  const [presetsError, setPresetsError] = useState<string | null>(null);
  const [selectedPresetId, setSelectedPresetId] = useState('');
  const [params, setParams] = useState<SliderParams>(DEFAULT_PARAMS);

  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [preview, setPreview] = useState<CharacterPreviewResponse | null>(null);
  const [summary, setSummary] = useState<CharacterBuildSummary | null>(null);

  const [exportingFmt, setExportingFmt] = useState<ExportFormat | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch<{ presets: CharacterPresetSummary[] }>(
      '/api/character-builder/presets',
    )
      .then((data) => {
        if (!cancelled) setPresets(data.presets);
      })
      .catch((err: unknown) => {
        if (!cancelled) setPresetsError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedPreset = useMemo(
    () => presets.find((p) => p.id === selectedPresetId) ?? null,
    [presets, selectedPresetId],
  );

  const handlePresetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const id = e.target.value;
      setSelectedPresetId(id);
      const preset = presets.find((p) => p.id === id);
      if (preset) {
        setParams({
          stature_m: preset.parameters.stature_m,
          mass_kg: preset.parameters.mass_kg,
          trunk_scale: preset.parameters.trunk_scale,
          arm_scale: preset.parameters.arm_scale,
          shoulder_scale: preset.parameters.shoulder_scale,
          club: preset.parameters.club,
        });
      }
      setPreview(null);
      setSummary(null);
      setPreviewError(null);
    },
    [presets],
  );

  // Overrides apply on top of a preset (CharacterSpecRequest semantics), so
  // sending every current slider value alongside the preset id is always
  // correct — selecting a preset fills these fields from its parameters,
  // and any further edit simply becomes the override for that field.
  const requestBody = useMemo(
    () => ({
      preset: selectedPresetId || null,
      stature_m: params.stature_m,
      mass_kg: params.mass_kg,
      trunk_scale: params.trunk_scale,
      arm_scale: params.arm_scale,
      shoulder_scale: params.shoulder_scale,
      club: params.club,
    }),
    [selectedPresetId, params],
  );

  const handlePreview = useCallback(async () => {
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const previewData = await apiFetch<CharacterPreviewResponse>(
        '/api/character-builder/preview',
        { method: 'POST', body: JSON.stringify(requestBody) },
      );
      const summaryData = await apiFetch<CharacterBuildSummary>(
        '/api/character-builder/build',
        { method: 'POST', body: JSON.stringify(requestBody) },
      );
      setPreview(previewData);
      setSummary(summaryData);
    } catch (err) {
      setPreviewError(errorMessage(err));
      setPreview(null);
      setSummary(null);
    } finally {
      setPreviewLoading(false);
    }
  }, [requestBody]);

  const handleExport = useCallback(
    async (fmt: ExportFormat, fallbackExt: string) => {
      setExportingFmt(fmt);
      setExportError(null);
      try {
        const response = await fetch(
          `${getApiBase()}/api/character-builder/export/${fmt}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
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
        const filename = match ? match[1] : `character.${fallbackExt}`;
        const mediaType = response.headers.get('Content-Type') ?? 'text/plain';
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
        setExportingFmt(null);
      }
    },
    [requestBody],
  );

  return (
    <div className="space-y-4">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Spec Character
      </h3>

      {presetsError && (
        <div className="text-xs text-red-400 bg-red-950/30 p-2.5 rounded border border-red-900/50">
          Failed to load presets: {presetsError}
        </div>
      )}

      <div>
        <label
          htmlFor="spec-preset-select"
          className="block text-xs font-semibold text-gray-300 mb-1"
        >
          Preset
        </label>
        <select
          id="spec-preset-select"
          value={selectedPresetId}
          onChange={handlePresetChange}
          className="w-full bg-gray-700 border-none text-gray-200 rounded px-2 py-1.5 text-sm focus:ring-1 focus:ring-blue-400"
        >
          <option value="">Defaults (no preset)</option>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        {selectedPreset && (
          <div className="mt-1.5 text-xs text-gray-400 space-y-0.5">
            <p>{selectedPreset.description}</p>
            <p className="text-gray-400">{selectedPreset.limitations}</p>
          </div>
        )}
      </div>

      <SpecSlider
        id="spec-stature-slider"
        label="Stature (m)"
        min={1.2}
        max={2.3}
        step={0.01}
        value={params.stature_m}
        displayValue={`${params.stature_m.toFixed(2)} m`}
        onChange={(v) => setParams((prev) => ({ ...prev, stature_m: v }))}
      />
      <SpecSlider
        id="spec-mass-slider"
        label="Mass (kg)"
        min={30}
        max={200}
        step={1}
        value={params.mass_kg}
        displayValue={`${params.mass_kg.toFixed(0)} kg`}
        onChange={(v) => setParams((prev) => ({ ...prev, mass_kg: v }))}
      />
      <SpecSlider
        id="spec-trunk-scale-slider"
        label="Trunk Length Scale"
        min={0.7}
        max={1.4}
        step={0.01}
        value={params.trunk_scale}
        displayValue={params.trunk_scale.toFixed(2)}
        onChange={(v) => setParams((prev) => ({ ...prev, trunk_scale: v }))}
      />
      <SpecSlider
        id="spec-arm-scale-slider"
        label="Arm Length Scale"
        min={0.7}
        max={1.4}
        step={0.01}
        value={params.arm_scale}
        displayValue={params.arm_scale.toFixed(2)}
        onChange={(v) => setParams((prev) => ({ ...prev, arm_scale: v }))}
      />
      <SpecSlider
        id="spec-shoulder-scale-slider"
        label="Shoulder Breadth Scale"
        min={0.7}
        max={1.4}
        step={0.01}
        value={params.shoulder_scale}
        displayValue={params.shoulder_scale.toFixed(2)}
        onChange={(v) => setParams((prev) => ({ ...prev, shoulder_scale: v }))}
      />

      <div>
        <label
          htmlFor="spec-club-select"
          className="block text-xs font-semibold text-gray-300 mb-1"
        >
          Club
        </label>
        <select
          id="spec-club-select"
          value={params.club}
          onChange={(e) =>
            setParams((prev) => ({ ...prev, club: e.target.value }))
          }
          className="w-full bg-gray-700 border-none text-gray-200 rounded px-2 py-1.5 text-sm focus:ring-1 focus:ring-blue-400"
        >
          <option value="driver">Driver</option>
          <option value="iron7">7 Iron</option>
        </select>
      </div>

      <button
        type="button"
        onClick={() => void handlePreview()}
        disabled={previewLoading}
        className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 text-white rounded py-2 text-sm font-semibold transition-colors"
      >
        {previewLoading ? 'Previewing...' : 'Preview'}
      </button>
      {previewError && (
        <div className="text-xs text-red-400 bg-red-950/30 p-2.5 rounded border border-red-900/50">
          {previewError}
        </div>
      )}

      {summary && preview && (
        <div className="border-t border-gray-700 pt-4">
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Preview Result
          </h4>
          <div className="text-xs text-gray-300 space-y-1 mb-2">
            <div>
              Total mass:{' '}
              <span className="font-mono text-blue-400">
                {summary.total_mass_kg.toFixed(2)} kg
              </span>
            </div>
            <div>
              Bodies: <span className="font-mono">{summary.bodies}</span>
            </div>
            <div>
              Joints: <span className="font-mono">{summary.joints}</span>
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-700/50">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 font-semibold">
                  <th className="pb-1.5">Body</th>
                  <th className="pb-1.5">Mass</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/40">
                {preview.bodies.map((b) => (
                  <tr key={b.name} className="text-gray-300">
                    <td className="py-1.5 font-medium">{b.name}</td>
                    <td className="py-1.5 font-mono">
                      {b.mass_kg.toFixed(2)} kg
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="border-t border-gray-700 pt-4">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Export
        </h4>
        <div className="flex flex-wrap gap-2">
          {EXPORT_BUTTONS.map(({ fmt, label, ext }) => (
            <button
              key={fmt}
              type="button"
              onClick={() => void handleExport(fmt, ext)}
              disabled={exportingFmt !== null}
              className="text-xs bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-400 text-gray-200 px-2.5 py-1.5 rounded transition-colors"
            >
              {exportingFmt === fmt ? 'Exporting…' : label}
            </button>
          ))}
        </div>
        {exportError && (
          <div className="mt-2 text-xs text-red-400 bg-red-950/30 p-2.5 rounded border border-red-900/50">
            {exportError}
          </div>
        )}
      </div>
    </div>
  );
}
