/**
 * Terrain Engine Page — Terrain presets, materials, types, and active terrain.
 *
 * Provides controls for loading terrain presets, querying terrain properties,
 * and viewing available materials and terrain types.
 */

import { useState, useCallback } from 'react';
import { useTerrain } from '@/api/useTerrain';
import type { TerrainPreset, TerrainMaterial, TerrainTypeInfo } from '@/api/useTerrain';

/**
 * TerrainPage - Full terrain engine tool page.
 */
export function TerrainPage() {
  const {
    presets,
    materials,
    terrainTypes,
    activeTerrain,
    queryResult,
    loadState,
    error,
    loadTerrain,
    queryTerrain,
  } = useTerrain();

  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [queryProperty, setQueryProperty] = useState('');
  const [sidebarTab, setSidebarTab] = useState<'presets' | 'materials' | 'types'>('presets');

  const handleLoadPreset = useCallback(() => {
    if (!selectedPreset) return;
    loadTerrain(selectedPreset);
  }, [selectedPreset, loadTerrain]);

  const handleQuery = useCallback(() => {
    if (!queryProperty.trim()) return;
    queryTerrain(queryProperty.trim());
  }, [queryProperty, queryTerrain]);

  const isLoading = loadState === 'loading';

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Left Sidebar */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-gray-200">Terrain Engine</h2>
          <p className="text-xs text-gray-500 mt-1">Load and configure terrain</p>
        </div>

        {/* Sidebar Tabs */}
        <div className="flex border-b border-gray-700">
          {(['presets', 'materials', 'types'] as const).map((tab) => (
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
          {/* Presets Tab */}
          {sidebarTab === 'presets' && (
            <div className="space-y-2">
              {presets.length === 0 ? (
                <div className="text-xs text-gray-500 italic text-center py-4">No presets available</div>
              ) : (
                presets.map((preset: TerrainPreset) => (
                  <button
                    key={preset.id}
                    onClick={() => setSelectedPreset(preset.id)}
                    className={`w-full text-left p-3 rounded-md transition-colors ${
                      selectedPreset === preset.id
                        ? 'bg-blue-600/30 border border-blue-500'
                        : 'bg-gray-700/30 border border-gray-600 hover:bg-gray-700/50'
                    }`}
                  >
                    <div className="text-sm font-medium text-gray-200">{preset.name}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{preset.description}</div>
                    <div className="text-xs text-gray-500 mt-1">Type: {preset.terrain_type}</div>
                  </button>
                ))
              )}
            </div>
          )}

          {/* Materials Tab */}
          {sidebarTab === 'materials' && (
            <div className="space-y-2">
              {materials.length === 0 ? (
                <div className="text-xs text-gray-500 italic text-center py-4">No materials available</div>
              ) : (
                materials.map((mat: TerrainMaterial) => (
                  <div key={mat.id} className="p-3 bg-gray-700/30 rounded-md border border-gray-600">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded-full border border-gray-500"
                        style={{ backgroundColor: mat.color }}
                      />
                      <span className="text-sm font-medium text-gray-200">{mat.name}</span>
                    </div>
                    <div className="text-xs text-gray-400 mt-1 space-y-0.5">
                      <div>Friction: {mat.friction_coefficient}</div>
                      <div>Restitution: {mat.restitution}</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Types Tab */}
          {sidebarTab === 'types' && (
            <div className="space-y-2">
              {terrainTypes.length === 0 ? (
                <div className="text-xs text-gray-500 italic text-center py-4">No terrain types available</div>
              ) : (
                terrainTypes.map((tt: TerrainTypeInfo) => (
                  <div key={tt.id} className="p-3 bg-gray-700/30 rounded-md border border-gray-600">
                    <div className="text-sm font-medium text-gray-200">{tt.name}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{tt.description}</div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* Load Button */}
        <div className="p-4 border-t border-gray-700">
          <button
            onClick={handleLoadPreset}
            disabled={!selectedPreset || isLoading}
            className="w-full py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm rounded-md transition-colors"
          >
            {isLoading ? 'Loading...' : 'Load Terrain'}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
          <h1 className="text-lg font-semibold text-gray-100">Terrain Engine</h1>
          <p className="text-xs text-gray-400 mt-1">
            {activeTerrain
              ? `Active: ${activeTerrain.name} (${activeTerrain.terrain_type})`
              : 'No terrain loaded'}
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
          {!activeTerrain && !isLoading && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-4xl mb-4 text-gray-600">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-16 h-16 mx-auto text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M3 15l5-7 4 4 4-6 5 9H3z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-400 mb-2">Load a Terrain</h3>
                <p className="text-sm text-gray-500 max-w-xs">
                  Select a terrain preset from the sidebar and click Load to begin.
                </p>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="flex items-center justify-center h-full">
              <div className="text-sm text-gray-400">Loading terrain...</div>
            </div>
          )}

          {activeTerrain && !isLoading && (
            <div className="space-y-6">
              {/* Active Terrain Info */}
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Active Terrain</h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-gray-500">Name</span>
                    <div className="text-gray-200 font-mono">{activeTerrain.name}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Type</span>
                    <div className="text-gray-200 font-mono">{activeTerrain.terrain_type}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Material</span>
                    <div className="text-gray-200 font-mono">{activeTerrain.material}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Loaded</span>
                    <div className="text-gray-200 font-mono">{activeTerrain.loaded_at}</div>
                  </div>
                </div>
              </div>

              {/* Properties */}
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Terrain Properties</h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <span className="text-gray-500">Dimensions</span>
                    <div className="text-gray-200 font-mono">{activeTerrain.properties.dimensions.join(' x ')}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Resolution</span>
                    <div className="text-gray-200 font-mono">{activeTerrain.properties.resolution}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Elevation Range</span>
                    <div className="text-gray-200 font-mono">{activeTerrain.properties.elevation_range.join(' - ')}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Slope Range</span>
                    <div className="text-gray-200 font-mono">{activeTerrain.properties.slope_range.join(' - ')}°</div>
                  </div>
                </div>
                {activeTerrain.properties.features.length > 0 && (
                  <div className="mt-3">
                    <span className="text-gray-500 text-xs">Features: </span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {activeTerrain.properties.features.map((f: string) => (
                        <span key={f} className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded">{f}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Query Panel */}
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Query Terrain</h3>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={queryProperty}
                    onChange={(e) => setQueryProperty(e.target.value)}
                    placeholder="Property name (e.g. elevation_profile)"
                    className="flex-1 bg-gray-700 text-gray-200 rounded px-3 py-1.5 text-xs border border-gray-600 focus:border-blue-500 focus:outline-none"
                    onKeyDown={(e) => { if (e.key === 'Enter') handleQuery(); }}
                  />
                  <button
                    onClick={handleQuery}
                    disabled={!queryProperty.trim()}
                    className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-xs rounded transition-colors"
                  >
                    Query
                  </button>
                </div>

                {queryResult && (
                  <div className="mt-3 bg-gray-700/50 p-3 rounded text-xs">
                    <div className="text-gray-400 mb-1">Query Result:</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div><span className="text-gray-500">Dimensions:</span> <span className="text-gray-200 font-mono">{queryResult.dimensions?.join(' x ') ?? 'N/A'}</span></div>
                      <div><span className="text-gray-500">Resolution:</span> <span className="text-gray-200 font-mono">{queryResult.resolution ?? 'N/A'}</span></div>
                      <div><span className="text-gray-500">Material:</span> <span className="text-gray-200 font-mono">{queryResult.material ?? 'N/A'}</span></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
