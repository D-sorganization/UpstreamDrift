/**
 * Dataset Generator Page — Controls for generating datasets, importing swing
 * data, managing generation parameters, and exploring features/plots/exports.
 */

import { useState, useCallback } from 'react';
import { useDatasetGenerator } from '@/api/useDatasetGenerator';
import type { FeatureInfo, PlotType, ExportFormat, DatasetControl } from '@/api/useDatasetGenerator';
export type {
  DatasetControl,
  DatasetLoadState,
  ExportFormat,
  FeatureInfo,
  GenerateResult,
  PlotType,
} from '@/api/useDatasetGenerator';

/**
 * DatasetGeneratorPage - Full dataset generation tool page.
 */
export function DatasetGeneratorPage() {
  const {
    features,
    plotTypes,
    exportFormats,
    controls,
    generateResult,
    loadState,
    error,
    generateDataset,
    importSwing,
    updateControl,
    exportDataset,
  } = useDatasetGenerator();

  const [sidebarTab, setSidebarTab] = useState<'features' | 'plots' | 'export'>('features');
  const [swingFilePath, setSwingFilePath] = useState('');
  const [controlValues, setControlValues] = useState<Record<string, unknown>>({});
  const [exportFormat, setExportFormat] = useState('csv');

  const isLoading = loadState === 'loading';

  const handleGenerate = useCallback(() => {
    generateDataset(controlValues);
  }, [controlValues, generateDataset]);

  const handleImportSwing = useCallback(() => {
    if (!swingFilePath.trim()) return;
    importSwing(swingFilePath.trim());
  }, [swingFilePath, importSwing]);

  const handleControlChange = useCallback((controlId: string, value: unknown) => {
    setControlValues((prev) => ({ ...prev, [controlId]: value }));
    updateControl(controlId, value);
  }, [updateControl]);

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Left Sidebar */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-gray-200">Dataset Generator</h2>
          <p className="text-xs text-gray-500 mt-1">Generate and import datasets</p>
        </div>

        {/* Sidebar Tabs */}
        <div className="flex border-b border-gray-700">
          {(['features', 'plots', 'export'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setSidebarTab(tab)}
              className={`flex-1 py-2 text-xs font-medium capitalize transition-colors ${
                sidebarTab === tab
                  ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="flex-1 p-4 overflow-y-auto">
          {/* Features Tab */}
          {sidebarTab === 'features' && (
            <div className="space-y-2">
              {features.length === 0 ? (
                <div className="text-xs text-gray-500 italic text-center py-4">No features available</div>
              ) : (
                features.map((f: FeatureInfo) => (
                  <div key={f.id} className="p-3 bg-gray-700/30 rounded-md border border-gray-600">
                    <div className="text-sm font-medium text-gray-200">{f.name}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{f.description}</div>
                    <span className="mt-1 inline-block px-1.5 py-0.5 bg-gray-600 text-gray-300 text-xs rounded">
                      {f.category}
                    </span>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Plots Tab */}
          {sidebarTab === 'plots' && (
            <div className="space-y-2">
              {plotTypes.length === 0 ? (
                <div className="text-xs text-gray-500 italic text-center py-4">No plot types available</div>
              ) : (
                plotTypes.map((pt: PlotType) => (
                  <div key={pt.id} className="p-3 bg-gray-700/30 rounded-md border border-gray-600">
                    <div className="text-sm font-medium text-gray-200">{pt.name}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{pt.description}</div>
                    {pt.axes.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {pt.axes.map((axis: string) => (
                          <span key={axis} className="px-1.5 py-0.5 bg-gray-600 text-gray-300 text-xs rounded">{axis}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {/* Export Tab */}
          {sidebarTab === 'export' && (
            <div className="space-y-2">
              {exportFormats.length === 0 ? (
                <div className="text-xs text-gray-500 italic text-center py-4">No export formats available</div>
              ) : (
                exportFormats.map((ef: ExportFormat) => (
                  <div key={ef.id} className="p-3 bg-gray-700/30 rounded-md border border-gray-600">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-200">{ef.name}</span>
                      <span className="px-1.5 py-0.5 bg-gray-600 text-gray-300 text-xs rounded font-mono">
                        .{ef.extension}
                      </span>
                    </div>
                    <div className="text-xs text-gray-400 mt-0.5">{ef.mime_type}</div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
          <h1 className="text-lg font-semibold text-gray-100">Dataset Generator</h1>
          <p className="text-xs text-gray-400 mt-1">
            {generateResult
              ? `Last: ${generateResult.name} (${generateResult.rows} rows)`
              : 'Configure parameters and generate a dataset'}
          </p>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-900/30 border-b border-red-800 px-6 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Content Area */}
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="max-w-2xl mx-auto space-y-6">
            {/* Generation Controls */}
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Generation Controls</h3>

              {controls.length > 0 ? (
                <div className="space-y-3">
                  {controls.map((ctrl: DatasetControl) => (
                    <div key={ctrl.id} className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-32 shrink-0">{ctrl.name}</label>
                      {ctrl.type === 'select' && ctrl.options ? (
                        <select
                          value={String(controlValues[ctrl.id] ?? ctrl.value)}
                          onChange={(e) => handleControlChange(ctrl.id, e.target.value)}
                          className="flex-1 bg-gray-700 text-gray-200 rounded px-2 py-1 text-xs border border-gray-600"
                        >
                          {ctrl.options.map((opt: string) => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      ) : ctrl.type === 'range' ? (
                        <input
                          type="range"
                          min={ctrl.min ?? 0}
                          max={ctrl.max ?? 100}
                          step={ctrl.step ?? 1}
                          value={Number(controlValues[ctrl.id] ?? ctrl.value)}
                          onChange={(e) => handleControlChange(ctrl.id, Number(e.target.value))}
                          className="flex-1"
                        />
                      ) : (
                        <input
                          type="text"
                          value={String(controlValues[ctrl.id] ?? ctrl.value ?? '')}
                          onChange={(e) => handleControlChange(ctrl.id, e.target.value)}
                          className="flex-1 bg-gray-700 text-gray-200 rounded px-2 py-1 text-xs border border-gray-600"
                        />
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-gray-500 italic">No controls available</div>
              )}

              <div className="mt-4 flex gap-2">
                <button
                  onClick={handleGenerate}
                  disabled={isLoading}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-xs rounded transition-colors"
                >
                  {isLoading ? 'Generating...' : 'Generate Dataset'}
                </button>
              </div>
            </div>

            {/* Import Swing Data */}
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Import Swing Data</h3>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={swingFilePath}
                  onChange={(e) => setSwingFilePath(e.target.value)}
                  placeholder="File path or data source..."
                  className="flex-1 bg-gray-700 text-gray-200 rounded px-3 py-1.5 text-xs border border-gray-600 focus:border-blue-500 focus:outline-none"
                  onKeyDown={(e) => { if (e.key === 'Enter') handleImportSwing(); }}
                />
                <button
                  onClick={handleImportSwing}
                  disabled={!swingFilePath.trim() || isLoading}
                  className="px-4 py-1.5 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-xs rounded transition-colors"
                >
                  Import
                </button>
              </div>
            </div>

            {/* Export Section */}
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Export</h3>
              <div className="flex gap-2 items-end">
                <div className="flex-1">
                  <label className="text-xs text-gray-400 block mb-1">Format</label>
                  <select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value)}
                    className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1.5 text-xs border border-gray-600"
                  >
                    {exportFormats.length > 0 ? (
                      exportFormats.map((ef: ExportFormat) => (
                        <option key={ef.id} value={ef.id}>{ef.name} (.{ef.extension})</option>
                      ))
                    ) : (
                      <>
                        <option value="csv">CSV (.csv)</option>
                        <option value="json">JSON (.json)</option>
                        <option value="hdf5">HDF5 (.h5)</option>
                      </>
                    )}
                  </select>
                </div>
                <button
                  onClick={() => exportDataset(exportFormat)}
                  disabled={!generateResult || isLoading}
                  className="px-4 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-xs rounded transition-colors"
                >
                  Export
                </button>
              </div>
            </div>

            {/* Result */}
            {generateResult && (
              <div className="bg-gray-800 rounded-lg border border-blue-500/30 p-4" data-testid="generate-result">
                <h3 className="text-sm font-semibold text-blue-300 mb-2">Generated Dataset</h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-gray-500">ID</span>
                    <div className="text-gray-200 font-mono">{generateResult.dataset_id}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Name</span>
                    <div className="text-gray-200 font-mono">{generateResult.name}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Rows</span>
                    <div className="text-gray-200 font-mono">{generateResult.rows}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Columns</span>
                    <div className="text-gray-200 font-mono">{generateResult.columns.length}</div>
                  </div>
                  <div className="col-span-2">
                    <span className="text-gray-500">Created</span>
                    <div className="text-gray-200 font-mono">{generateResult.created_at}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
