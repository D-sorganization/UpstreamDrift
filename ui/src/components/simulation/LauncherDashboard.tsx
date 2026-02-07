/**
 * LauncherDashboard — Tile-based grid dashboard for the Tauri launcher.
 *
 * This is the primary entry point for the application, displaying all
 * available tiles in a responsive grid layout matching the PyQt launcher.
 *
 * Features:
 *   - Tiles loaded from shared manifest API (DRY — single source of truth)
 *   - Model Explorer is first tile (simulation setup entry point)
 *   - Category sections: Physics Engines, Tools & Utilities
 *   - Status chips showing tile readiness
 *   - Logo/icon display for each tile
 *   - Help button prominently displayed
 *   - "Launch Simulation" button always visible (sticky bottom)
 */

import { HelpCircle, Loader2, AlertTriangle, Zap, Wrench, ExternalLink, RefreshCw } from 'lucide-react';
import type { LauncherTile } from '@/api/useLauncherManifest';
import type { ManifestLoadState } from '@/api/useLauncherManifest';

interface Props {
    tiles: LauncherTile[];
    loadState: ManifestLoadState;
    error: string | null;
    selectedTileId: string | null;
    onSelectTile: (tileId: string) => void;
    onLaunchTile: (tileId: string) => void;
    onShowHelp: () => void;
    onRefetch: () => void;
}

/** Map status values to display colors and labels */
function getStatusChip(status: string): { label: string; color: string } {
    switch (status) {
        case 'gui_ready':
            return { label: 'GUI Ready', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-600/50' };
        case 'engine_ready':
            return { label: 'Ready', color: 'bg-blue-500/20 text-blue-300 border-blue-600/50' };
        case 'utility':
            return { label: 'Utility', color: 'bg-purple-500/20 text-purple-300 border-purple-600/50' };
        case 'external':
            return { label: 'External', color: 'bg-amber-500/20 text-amber-300 border-amber-600/50' };
        case 'simulator':
            return { label: 'Simulator', color: 'bg-cyan-500/20 text-cyan-300 border-cyan-600/50' };
        default:
            return { label: status || 'Unknown', color: 'bg-gray-500/20 text-gray-300 border-gray-600/50' };
    }
}

/** Map category to an icon */
function CategoryIcon({ category }: { category: string }) {
    switch (category) {
        case 'physics_engine':
            return <Zap className="w-3.5 h-3.5" aria-hidden="true" />;
        case 'tool':
            return <Wrench className="w-3.5 h-3.5" aria-hidden="true" />;
        case 'external':
            return <ExternalLink className="w-3.5 h-3.5" aria-hidden="true" />;
        default:
            return null;
    }
}

function TileCard({
    tile,
    isSelected,
    onSelect,
    onLaunch,
}: {
    tile: LauncherTile;
    isSelected: boolean;
    onSelect: () => void;
    onLaunch: () => void;
}) {
    const status = getStatusChip(tile.status);

    return (
        <button
            id={`tile-${tile.id}`}
            onClick={onSelect}
            onDoubleClick={onLaunch}
            aria-label={`${tile.name} — ${tile.description}`}
            aria-pressed={isSelected}
            className={`
        group relative flex flex-col items-center p-4 rounded-xl border-2 transition-all duration-200
        hover:shadow-lg hover:shadow-blue-500/10 hover:-translate-y-0.5
        focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-900
        ${isSelected
                    ? 'border-blue-500 bg-blue-500/15 ring-1 ring-blue-500/40 shadow-md shadow-blue-500/20'
                    : 'border-gray-700 bg-gray-800/80 hover:border-gray-500'
                }
      `}
        >
            {/* Logo area */}
            <div className="w-16 h-16 rounded-lg bg-gray-700/50 flex items-center justify-center mb-3 group-hover:bg-gray-600/50 transition-colors">
                <img
                    src={`/api/launcher/logos/${tile.logo}`}
                    alt={`${tile.name} logo`}
                    className="w-12 h-12 object-contain"
                    onError={(e) => {
                        // Fallback to category icon if logo not found
                        (e.target as HTMLImageElement).style.display = 'none';
                        const parent = (e.target as HTMLImageElement).parentElement;
                        if (parent) {
                            parent.classList.add('text-gray-400');
                        }
                    }}
                />
                {/* Fallback icon rendered behind the img; visible if img fails */}
                <span className="absolute text-gray-400 opacity-0 group-[img-error]:opacity-100" aria-hidden="true">
                    <CategoryIcon category={tile.category} />
                </span>
            </div>

            {/* Name */}
            <h3 className="text-sm font-semibold text-white text-center mb-1 leading-tight">
                {tile.name}
            </h3>

            {/* Description */}
            <p className="text-xs text-gray-400 text-center mb-2 leading-relaxed line-clamp-2">
                {tile.description}
            </p>

            {/* Status chip */}
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border ${status.color}`}>
                <CategoryIcon category={tile.category} />
                {status.label}
            </span>

            {/* Capabilities badges — show on hover */}
            <div className="mt-2 flex flex-wrap gap-1 justify-center opacity-0 group-hover:opacity-100 transition-opacity max-h-0 group-hover:max-h-20 overflow-hidden">
                {tile.capabilities.slice(0, 3).map((cap) => (
                    <span
                        key={cap}
                        className="px-1.5 py-0.5 bg-gray-700/80 text-gray-400 text-[9px] rounded"
                    >
                        {cap.replace(/_/g, ' ')}
                    </span>
                ))}
                {tile.capabilities.length > 3 && (
                    <span className="px-1.5 py-0.5 text-gray-500 text-[9px]">
                        +{tile.capabilities.length - 3}
                    </span>
                )}
            </div>
        </button>
    );
}

export function LauncherDashboard({
    tiles,
    loadState,
    error,
    selectedTileId,
    onSelectTile,
    onLaunchTile,
    onShowHelp,
    onRefetch,
}: Props) {
    const engines = tiles.filter((t) => t.category === 'physics_engine');
    const toolsAndExternal = tiles.filter((t) => t.category === 'tool' || t.category === 'external');
    const selectedTile = tiles.find((t) => t.id === selectedTileId);

    if (loadState === 'loading') {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900" role="status" aria-label="Loading launcher">
                <div className="text-center">
                    <Loader2 className="w-10 h-10 animate-spin text-blue-400 mx-auto mb-4" aria-hidden="true" />
                    <p className="text-gray-400 text-sm">Loading launcher tiles...</p>
                </div>
            </div>
        );
    }

    if (loadState === 'error') {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900" role="alert">
                <div className="text-center max-w-md">
                    <AlertTriangle className="w-10 h-10 text-red-400 mx-auto mb-4" aria-hidden="true" />
                    <h2 className="text-lg font-semibold text-white mb-2">Failed to Load</h2>
                    <p className="text-sm text-gray-400 mb-4">{error || 'Could not connect to the launcher API.'}</p>
                    <button
                        onClick={onRefetch}
                        className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400"
                    >
                        <RefreshCw className="w-4 h-4" aria-hidden="true" />
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-screen bg-gray-900">
            {/* Header */}
            <header className="flex items-center justify-between px-6 py-4 bg-gray-800/80 border-b border-gray-700 flex-shrink-0">
                <div>
                    <h1 className="text-xl font-bold text-white">Golf Modeling Suite</h1>
                    <p className="text-xs text-gray-500 mt-0.5">
                        {tiles.length} tiles · {engines.length} engines · {toolsAndExternal.length} tools
                    </p>
                </div>
                <button
                    id="help-button"
                    onClick={onShowHelp}
                    aria-label="Help"
                    className="flex items-center gap-2 px-3 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 rounded-lg border border-blue-600/40 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                    <HelpCircle className="w-5 h-5" aria-hidden="true" />
                    <span className="text-sm font-medium">Help</span>
                </button>
            </header>

            {/* Scrollable tile grid */}
            <main className="flex-1 overflow-y-auto px-6 py-6" id="tile-grid-container">
                {/* Physics Engines */}
                {engines.length > 0 && (
                    <section aria-label="Physics Engines" className="mb-8">
                        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Zap className="w-4 h-4" aria-hidden="true" />
                            Physics Engines
                        </h2>
                        <div
                            className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4"
                            role="group"
                            aria-label="Physics engine tiles"
                            id="engine-tiles-grid"
                        >
                            {engines.map((tile) => (
                                <TileCard
                                    key={tile.id}
                                    tile={tile}
                                    isSelected={selectedTileId === tile.id}
                                    onSelect={() => onSelectTile(tile.id)}
                                    onLaunch={() => onLaunchTile(tile.id)}
                                />
                            ))}
                        </div>
                    </section>
                )}

                {/* Tools & Utilities */}
                {toolsAndExternal.length > 0 && (
                    <section aria-label="Tools and Utilities" className="mb-8">
                        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Wrench className="w-4 h-4" aria-hidden="true" />
                            Tools & Utilities
                        </h2>
                        <div
                            className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4"
                            role="group"
                            aria-label="Tool tiles"
                            id="tool-tiles-grid"
                        >
                            {toolsAndExternal.map((tile) => (
                                <TileCard
                                    key={tile.id}
                                    tile={tile}
                                    isSelected={selectedTileId === tile.id}
                                    onSelect={() => onSelectTile(tile.id)}
                                    onLaunch={() => onLaunchTile(tile.id)}
                                />
                            ))}
                        </div>
                    </section>
                )}
            </main>

            {/* Sticky bottom bar — Launch button always visible (fixes #1165) */}
            <footer
                className="flex items-center justify-between px-6 py-3 bg-gray-800/95 border-t border-gray-700 flex-shrink-0 backdrop-blur-sm"
                id="launch-footer"
            >
                <div className="text-sm text-gray-400">
                    {selectedTile ? (
                        <span>
                            Selected: <strong className="text-white">{selectedTile.name}</strong>
                            <span className="ml-2 text-xs text-gray-500">{selectedTile.description}</span>
                        </span>
                    ) : (
                        <span className="text-gray-500">Select a tile to launch</span>
                    )}
                </div>
                <button
                    id="launch-simulation-button"
                    onClick={() => selectedTile && onLaunchTile(selectedTile.id)}
                    disabled={!selectedTile}
                    aria-label={selectedTile ? `Launch ${selectedTile.name}` : 'Launch Simulation'}
                    className={`
            inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-sm transition-all
            focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-offset-2 focus:ring-offset-gray-900
            ${selectedTile
                            ? 'bg-green-600 hover:bg-green-500 text-white shadow-lg shadow-green-600/30 hover:shadow-green-500/40'
                            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        }
          `}
                >
                    <Zap className="w-4 h-4" aria-hidden="true" />
                    Launch Simulation
                </button>
            </footer>
        </div>
    );
}
