/**
 * Spine Viewer page (SPN-15c, #12815): web counterpart of the PyQt6 viewer.
 *
 * Opens a hash-verified spine study receipt through the API and shows the
 * per-level load heatmap. The scientific status and distribution source are
 * always visible above the heatmap; unqualified data is never shown as
 * qualified.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  fetchSpineReceipts,
  fetchSpineStudy,
  SPINE_QUANTITIES,
  type SpineStudyPayload,
} from '@/api/spineViewer';
import { SpineHeatmap } from '@/components/spine/SpineHeatmap';
import { SpineStatusBanner } from '@/components/spine/SpineStatusBanner';

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function SpineViewerPage() {
  const [receipts, setReceipts] = useState<string[]>([]);
  const [selected, setSelected] = useState('');
  const [quantity, setQuantity] = useState<string>(SPINE_QUANTITIES[0]);
  const [study, setStudy] = useState<SpineStudyPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    fetchSpineReceipts()
      .then((list) => active && setReceipts(list))
      .catch((e: unknown) => active && setError(messageOf(e)));
    return () => {
      active = false;
    };
  }, []);

  const openStudy = useCallback(async () => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setStudy(null);
    try {
      setStudy(await fetchSpineStudy(selected));
    } catch (e: unknown) {
      setError(messageOf(e));
    } finally {
      setLoading(false);
    }
  }, [selected]);

  const heatmap = study?.heatmaps[quantity];

  return (
    <div className="flex min-h-0 flex-1 flex-col text-gray-100">
      <div className="border-b border-gray-700 bg-gray-800 px-6 py-4">
        <h1 className="text-lg font-semibold">Spine Viewer</h1>
        <p className="mt-1 text-xs text-gray-400">
          Per-level spinal loads from hash-verified study receipts
        </p>
      </div>
      <div className="flex flex-wrap items-end gap-4 px-6 py-4">
        <label className="text-xs text-gray-300">
          Study receipt
          <select
            className="ml-2 rounded bg-gray-700 px-2 py-1 text-gray-100"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            <option value="">Select a receipt</option>
            {receipts.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={openStudy}
          disabled={!selected || loading}
          className="rounded bg-gray-700 px-3 py-1 text-xs hover:bg-gray-600 disabled:opacity-50"
        >
          Open Study
        </button>
        <label className="text-xs text-gray-300">
          Quantity
          <select
            className="ml-2 rounded bg-gray-700 px-2 py-1 text-gray-100"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
          >
            {SPINE_QUANTITIES.map((q) => (
              <option key={q} value={q}>
                {q}
              </option>
            ))}
          </select>
        </label>
      </div>
      {error && (
        <div
          role="alert"
          className="border-y border-red-800 bg-red-900/30 px-6 py-3 text-sm text-red-300"
        >
          {error}
        </div>
      )}
      {study && <SpineStatusBanner study={study} />}
      {study && heatmap && (
        <div className="flex-1 overflow-y-auto p-6">
          <SpineHeatmap heatmap={heatmap} levels={study.levels} />
        </div>
      )}
    </div>
  );
}
