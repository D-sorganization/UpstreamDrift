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

import { useState, useMemo } from 'react';
import {
    Activity,
    AlertTriangle,
    ChevronRight,
    Download,
    ExternalLink,
    HelpCircle,
    Monitor,
    Loader2,
    MessageSquare,
    Play,
    RefreshCw,
    Search,
    Wrench,
    Zap,
} from 'lucide-react';
import { useNavigate, NavLink, Link } from 'react-router';
import type { LauncherTile } from '@/api/useLauncherManifest';
import type { ManifestLoadState } from '@/api/useLauncherManifest';
import type { LauncherWindowRecord } from '@/api/launcherWindowRegistry';
import type { EngineStatusResponse } from '@/api/generated/types';
import { canLaunchNativeWindow, resolveTileLaunchAction } from '@/api/webLaunch';
import { DIAGNOSTICS_BADGE_CLEARANCE } from '@/components/ui/DiagnosticsPanel';
import type { TileLaunchAction } from '@/api/webLaunch';

interface Props {
    tiles: LauncherTile[];
    engineStatuses?: EngineStatusResponse[];
    loadState: ManifestLoadState;
    error: string | null;
    selectedTileId: string | null;
    launchedWindows: LauncherWindowRecord[];
    onSelectTile: (tileId: string) => void;
    onLaunchTile: (tileId: string) => void;
    onFocusLaunchedTile: (tileId: string) => void;
    onShowHelp: () => void;
    onShowAbout?: () => void;
    onRefetch: () => void;
    /**
     * Whether native-window tiles may be launched from this environment
     * (Tauri or localhost API). Defaults to runtime detection; injectable
     * for tests (issue #7461).
     */
    nativeWindowAllowed?: boolean;
}

interface TileSection {
    id: string;
    title: string;
    ariaLabel: string;
    groupLabel: string;
    icon: typeof Zap;
    tiles: LauncherTile[];
}

/** Navigation button to routable pages */
function NavButton({
    to,
    label,
    icon: Icon,
    onClick,
}: {
    to: string;
    label: string;
    icon: React.ComponentType<{ className?: string; 'aria-hidden'?: boolean }>;
    onClick?: () => void;
}) {
    const navigate = useNavigate();
    return (
        <button
            onClick={() => {
                navigate(to);
                onClick?.();
            }}
            aria-label={label}
            className="touch-target flex items-center gap-2 px-3 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 rounded-lg border border-blue-600/40 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400"
        >
            <Icon className="w-5 h-5" aria-hidden={true} />
            <span className="text-sm font-medium hidden md:inline">{label}</span>
        </button>
    );
}

/** Map status values to display colors and labels */
function getStatusChip(status: string): { label: string; color: string } {
    switch (status) {
        case 'gui_ready':
            return { label: 'GUI Ready', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-600/50' };
        case 'engine_ready':
        case 'ready':
            return { label: 'Ready', color: 'bg-blue-500/20 text-blue-300 border-blue-600/50' };
        case 'experimental':
            return { label: 'Experimental', color: 'bg-amber-500/20 text-amber-300 border-amber-600/50' };
        case 'runtime_unavailable':
        case 'provider_unavailable':
            return { label: 'Unavailable', color: 'bg-rose-500/20 text-rose-300 border-rose-600/50' };
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

/** Badge shown for tiles that cannot be launched from this environment. */
function WebReachabilityBadge({ action }: { action: TileLaunchAction }) {
    if (action.kind !== 'blocked') {
        return null;
    }
    return (
        <span
            title={action.reason}
            className="mt-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border bg-gray-600/30 text-gray-300 border-gray-500/50"
        >
            <Monitor className="w-3 h-3" aria-hidden="true" />
            {action.badge}
        </span>
    );
}

function TileCard({
    tile,
    isSelected,
    launchAction,
    onSelect,
    onLaunch,
}: {
    tile: LauncherTile;
    isSelected: boolean;
    launchAction: TileLaunchAction;
    onSelect: () => void;
    onLaunch: () => void;
}) {
    const status = getStatusChip(tile.status);
    const blocked = launchAction.kind === 'blocked';

    return (
        <button
            id={`tile-${tile.id}`}
            onClick={onSelect}
            onDoubleClick={blocked ? undefined : onLaunch}
            aria-label={`${tile.name} — ${tile.description}`}
            aria-pressed={isSelected}
            className={`
        group relative flex flex-col items-center p-4 rounded-xl border transition-all duration-200
        bg-white/[0.07] backdrop-blur-md shadow-lg shadow-black/25
        hover:bg-white/[0.11] hover:shadow-xl hover:shadow-blue-500/20 hover:-translate-y-0.5
        focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-900
        ${isSelected
                    ? 'border-blue-400 bg-blue-500/20 ring-1 ring-blue-400/50 shadow-blue-500/25'
                    : 'border-white/10 hover:border-white/25'
                }
      `}
        >
            {/* Logo area */}
            <div className="w-16 h-16 rounded-lg bg-gray-950/35 border border-white/10 flex items-center justify-center mb-3 group-hover:bg-gray-900/50 transition-colors">
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
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${status.color}`}>
                <CategoryIcon category={tile.category} />
                {status.label}
            </span>

            {/* Web reachability badge (issue #7461) — honest affordance
                instead of a launch button that does nothing visible */}
            <WebReachabilityBadge action={launchAction} />

            {/* Capabilities badges — show on hover */}
            <div className="mt-2 flex flex-wrap gap-1 justify-center opacity-0 group-hover:opacity-100 transition-opacity max-h-0 group-hover:max-h-20 overflow-hidden">
                {tile.capabilities.slice(0, 3).map((cap) => (
                    <span
                        key={cap}
                        className="px-1.5 py-0.5 bg-gray-700/80 text-gray-400 text-xs rounded"
                    >
                        {cap.replace(/_/g, ' ')}
                    </span>
                ))}
                {tile.capabilities.length > 3 && (
                    <span className="px-1.5 py-0.5 text-gray-400 text-xs">
                        +{tile.capabilities.length - 3}
                    </span>
                )}
            </div>
        </button>
    );
}

function isAnalysisTile(tile: LauncherTile): boolean {
    const searchableText = [
        tile.id,
        tile.name,
        tile.description,
        tile.type,
        ...tile.capabilities,
    ].join(' ').toLowerCase();
    return /analysis|capture|viewer|mediapipe|openpose|c3d|video|data/.test(searchableText);
}

/**
 * Planned tools (#14262) have no working destination yet, so home never
 * offers them; they stay reachable from the Developer & Research workspace.
 */
const PLANNED_STATUS = 'planned';

function isHomeDestination(tile: LauncherTile): boolean {
    return tile.status !== PLANNED_STATUS;
}

function buildTileSections(tiles: LauncherTile[]): TileSection[] {
    const engines = tiles.filter((tile) => tile.category === 'physics_engine');
    const analysisTools = tiles.filter(
        (tile) => tile.category !== 'physics_engine' && isAnalysisTile(tile)
    );
    const utilities = tiles.filter(
        (tile) => tile.category !== 'physics_engine' && !isAnalysisTile(tile)
    );

    return [
        {
            id: 'core-engine-tiles-grid',
            title: 'Core Physics Engines',
            ariaLabel: 'Core Physics Engines',
            groupLabel: 'Core physics engine tiles',
            icon: Zap,
            tiles: engines,
        },
        {
            id: 'analysis-tool-tiles-grid',
            title: 'Analysis Tools',
            ariaLabel: 'Analysis Tools',
            groupLabel: 'Analysis tool tiles',
            icon: Activity,
            tiles: analysisTools,
        },
        {
            id: 'utility-tiles-grid',
            title: 'Utilities',
            ariaLabel: 'Utilities',
            groupLabel: 'Utility tiles',
            icon: Wrench,
            tiles: utilities,
        },
    ].filter((section) => section.tiles.length > 0);
}

interface EngineRow {
    id: string;
    engineKey: string;
    name: string;
    type: string;
    tileId: string;
}

const ENGINE_ROWS: EngineRow[] = [
    { id: 'mujoco', engineKey: 'mujoco', name: 'MuJoCo Engine', type: 'Rigid Body & Contact', tileId: 'mujoco_unified' },
    { id: 'drake', engineKey: 'drake', name: 'Drake Engine', type: 'Dynamics & Optimization', tileId: 'drake_golf' },
    { id: 'pinocchio', engineKey: 'pinocchio', name: 'Pinocchio Engine', type: 'Kinematics & Spatial Dynamics', tileId: 'pinocchio_golf' },
    { id: 'opensim', engineKey: 'opensim', name: 'OpenSim Engine', type: 'Musculoskeletal & Biomechanics', tileId: 'opensim_golf' },
    { id: 'myosuite', engineKey: 'myosuite', name: 'MyoSuite Engine', type: 'Muscle & Neural Control', tileId: 'myosim_suite' },
    { id: 'matlab', engineKey: 'matlab_2d', name: 'MATLAB / Simscape', type: 'Simscape Multibody 2D/3D', tileId: 'matlab_unified' },
];

function getEngineReadiness(
    row: EngineRow,
    engineStatuses?: EngineStatusResponse[]
): { status: 'Ready' | 'Experimental' | 'Not installed' | 'Needs MATLAB'; color: string; canOpen: boolean } {
    if (!engineStatuses || engineStatuses.length === 0) {
        return { status: 'Ready', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', canOpen: true };
    }
    const match = engineStatuses.find((e) =>
        e.name === row.engineKey ||
        (row.engineKey === 'myosuite' && (e.name === 'myosuite' || e.name === 'myosim')) ||
        (row.engineKey === 'matlab_2d' && (e.name === 'matlab_2d' || e.name === 'matlab_3d' || e.name === 'matlab'))
    );
    if (!match || !match.available) {
        if (row.id === 'matlab') {
            return { status: 'Needs MATLAB', color: 'bg-amber-500/20 text-amber-300 border-amber-500/40', canOpen: false };
        }
        return { status: 'Not installed', color: 'bg-rose-500/20 text-rose-300 border-rose-500/40', canOpen: false };
    }
    if (match.status === 'gui_ready' || match.status === 'ready' || match.status === 'engine_ready') {
        return { status: 'Ready', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', canOpen: true };
    }
    return { status: 'Experimental', color: 'bg-amber-500/20 text-amber-300 border-amber-500/40', canOpen: true };
}

const PIPELINE_STEPS = [
    { id: 'capture', step: '1', title: 'Capture', desc: 'Video & Mocap', route: '/workspaces/capture-analyze', prereq: 'Ready to start', ready: true },
    { id: 'estimate', step: '2', title: 'Estimate', desc: 'Pose & Joints', route: '/tools/video-analyzer', prereq: 'Needs a capture', ready: false },
    { id: 'model_match', step: '3', title: 'Model & Match', desc: 'Biomechanics', route: '/workspaces/model-match', prereq: 'Needs an estimate', ready: false },
    { id: 'optimize', step: '4', title: 'Optimize', desc: 'Forward Dynamics', route: '/workspaces/optimize-train', prereq: 'Needs matched model', ready: false },
    { id: 'compare', step: '5', title: 'Compare', desc: 'Results & Valid.', route: '/workspaces/results-compare', prereq: 'Needs simulation run', ready: false },
];

export function LauncherDashboard({
    tiles: manifestTiles,
    engineStatuses,
    loadState,
    error,
    selectedTileId,
    launchedWindows,
    onSelectTile,
    onLaunchTile,
    onFocusLaunchedTile,
    onShowHelp,
    onShowAbout,
    onRefetch,
    nativeWindowAllowed = canLaunchNativeWindow(),
}: Props) {
    const [searchQuery, setSearchQuery] = useState('');
    const tiles = manifestTiles.filter(isHomeDestination);
    const engines = tiles.filter((t) => t.category === 'physics_engine');
    const toolsAndExternal = tiles.filter((t) => t.category !== 'physics_engine');

    const filteredTiles = useMemo(() => {
        if (!searchQuery.trim()) return tiles;
        const q = searchQuery.toLowerCase().trim();
        return tiles.filter((tile) =>
            tile.name.toLowerCase().includes(q) ||
            tile.description.toLowerCase().includes(q) ||
            tile.id.toLowerCase().includes(q) ||
            tile.capabilities.some((c) => c.toLowerCase().includes(q))
        );
    }, [tiles, searchQuery]);

    const selectedTile = tiles.find((t) => t.id === selectedTileId);
    const tileSections = buildTileSections(filteredTiles);
    const selectedTileAction = selectedTile
        ? resolveTileLaunchAction(selectedTile, nativeWindowAllowed)
        : null;
    const selectedLaunchable = selectedTileAction !== null && selectedTileAction.kind !== 'blocked';

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
            {/* #14280: wrap instead of pushing actions off a phone screen; below md
                the actions are icon-only (each keeps its aria-label). */}
            <header className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 md:px-6 py-3 md:py-4 bg-gray-800/80 border-b border-gray-700 flex-shrink-0">
                <div>
                    <h1 className="heading-page">Golf Modeling Suite</h1>
                    <p className="text-xs text-gray-400 mt-0.5">
                        {tiles.length} tiles · {engines.length} engines · {toolsAndExternal.length} tools
                    </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                    <details className="relative">
                        <summary
                            role="button"
                            aria-label="Window list"
                            className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 bg-gray-700/40 hover:bg-gray-700/60 text-gray-200 rounded-lg border border-gray-600/60 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400"
                        >
                            <Monitor className="w-5 h-5" aria-hidden="true" />
                            <span className="text-sm font-medium hidden md:inline">Windows</span>
                            {launchedWindows.length > 0 && (
                                <span className="rounded-full bg-blue-500/30 px-1.5 text-xs text-blue-100">
                                    {launchedWindows.length}
                                </span>
                            )}
                        </summary>
                        <div
                            className="absolute right-0 z-20 mt-2 w-72 rounded-lg border border-gray-700 bg-gray-900 p-2 shadow-xl shadow-black/40"
                            role="menu"
                            aria-label="Launched windows"
                        >
                            {launchedWindows.length === 0 ? (
                                <p className="px-3 py-2 text-sm text-gray-400">No launched windows</p>
                            ) : (
                                launchedWindows.map((record) => (
                                    <div
                                        key={record.tileId}
                                        className="flex items-center justify-between gap-3 rounded-md px-3 py-2 hover:bg-white/5"
                                    >
                                        <div className="min-w-0">
                                            <p className="truncate text-sm font-medium text-white">{record.name}</p>
                                            <p className="text-xs text-gray-400">
                                                {record.launchCount} launch{record.launchCount === 1 ? '' : 'es'}
                                            </p>
                                        </div>
                                        <button
                                            type="button"
                                            className="shrink-0 rounded-md border border-blue-500/40 px-2 py-1 text-xs font-medium text-blue-200 hover:bg-blue-500/20 focus:outline-none focus:ring-2 focus:ring-blue-400"
                                            onClick={() => onFocusLaunchedTile(record.tileId)}
                                        >
                                            Focus
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    </details>
                    <NavButton to="/chat" label="Chat" icon={MessageSquare} />
                    <a
                        href="/capability-atlas/index.html"
                        aria-label="Capability Map"
                        className="touch-target flex items-center gap-2 px-3 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 rounded-lg border border-blue-600/40 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400"
                    >
                        <ExternalLink className="w-5 h-5" aria-hidden="true" />
                        <span className="text-sm font-medium hidden md:inline">Capability Map</span>
                    </a>
                    <button
                        id="help-button"
                        onClick={onShowHelp}
                        aria-label="Help"
                        className="touch-target flex items-center gap-2 px-3 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 rounded-lg border border-blue-600/40 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400"
                    >
                        <HelpCircle className="w-5 h-5" aria-hidden="true" />
                        <span className="text-sm font-medium hidden md:inline">Help</span>
                    </button>
                </div>
            </header>

            {/* Task Workspaces Bar (ORG-06, #10516) */}
            <nav
                aria-label="Task Workspaces"
                className="flex flex-wrap items-center gap-2 px-4 md:px-6 py-2.5 bg-gray-800/40 border-b border-gray-700/50 text-xs flex-shrink-0"
            >
                <span className="font-semibold text-gray-400 uppercase tracking-wider text-xs mr-1">Workspaces:</span>
                <NavLink
                    to="/workspaces/capture-analyze"
                    className="touch-target inline-flex items-center px-2.5 py-1 rounded bg-blue-600/10 hover:bg-blue-600/25 text-blue-300 border border-blue-500/20 whitespace-nowrap transition-colors"
                >
                    Capture & Analyze
                </NavLink>
                <NavLink
                    to="/workspaces/model-match"
                    className="touch-target inline-flex items-center px-2.5 py-1 rounded bg-blue-600/10 hover:bg-blue-600/25 text-blue-300 border border-blue-500/20 whitespace-nowrap transition-colors"
                >
                    Model & Match
                </NavLink>
                <NavLink
                    to="/workspaces/shot-course-lab"
                    className="touch-target inline-flex items-center px-2.5 py-1 rounded bg-blue-600/10 hover:bg-blue-600/25 text-blue-300 border border-blue-500/20 whitespace-nowrap transition-colors"
                >
                    Shot & Course Lab
                </NavLink>
                <NavLink
                    to="/workspaces/optimize-train"
                    className="touch-target inline-flex items-center px-2.5 py-1 rounded bg-blue-600/10 hover:bg-blue-600/25 text-blue-300 border border-blue-500/20 whitespace-nowrap transition-colors"
                >
                    Optimize & Train
                </NavLink>
                <NavLink
                    to="/workspaces/results-compare"
                    className="touch-target inline-flex items-center px-2.5 py-1 rounded bg-blue-600/10 hover:bg-blue-600/25 text-blue-300 border border-blue-500/20 whitespace-nowrap transition-colors"
                >
                    Results & Compare
                </NavLink>
                <NavLink
                    to="/workspaces/dev-research"
                    className="touch-target inline-flex items-center px-2.5 py-1 rounded bg-gray-700/30 hover:bg-gray-700/50 text-gray-300 border border-gray-600/30 whitespace-nowrap transition-colors"
                >
                    Developer & Research
                </NavLink>
            </nav>

            {/* Scrollable tile grid */}
            <main className="flex-1 overflow-y-auto px-3 sm:px-4 md:px-6 py-3 md:py-6" id="tile-grid-container">
                {/* Resume Panel (#14259) */}
                <section aria-label="Resume Work" className="mb-6 rounded-xl border border-token-border bg-gradient-to-r from-surface-muted to-surface p-4">
                    {launchedWindows.length > 0 ? (
                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <div>
                                <span className="text-xs font-semibold uppercase tracking-wider text-primary">Resume Session</span>
                                <h3 className="text-base font-semibold text-white">{launchedWindows[0].name}</h3>
                                <p className="text-xs text-token-muted">Active background session · {launchedWindows[0].launchCount} launch{launchedWindows[0].launchCount === 1 ? '' : 'es'}</p>
                            </div>
                            <button
                                type="button"
                                onClick={() => onFocusLaunchedTile(launchedWindows[0].tileId)}
                                className="inline-flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg text-sm font-medium transition-colors shadow-md shadow-primary/25"
                            >
                                <Play className="w-4 h-4" aria-hidden="true" />
                                Resume Session
                            </button>
                        </div>
                    ) : (
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                            <div>
                                <span className="text-xs font-semibold uppercase tracking-wider text-primary">Start Your Workflow</span>
                                <h3 className="text-base font-semibold text-white">Import Data to Begin Analysis</h3>
                                <p className="text-xs text-token-muted">Load recorded swing video or 3D optical motion capture files to initialize the pipeline.</p>
                            </div>
                            <div className="flex items-center gap-2">
                                <Link
                                    to="/tools/video-analyzer"
                                    aria-label="Import swing video"
                                    className="inline-flex items-center gap-1.5 px-3 py-2 bg-primary/30 hover:bg-primary/50 text-token-text border border-primary/40 rounded-lg text-xs font-medium transition-colors"
                                >
                                    <Download className="w-3.5 h-3.5" aria-hidden="true" />
                                    Import swing video
                                </Link>
                                <Link
                                    to="/tools/motion-capture"
                                    aria-label="Import C3D / OpenCap"
                                    className="inline-flex items-center gap-1.5 px-3 py-2 bg-surface-muted hover:bg-surface-raised text-token-text border border-token-border rounded-lg text-xs font-medium transition-colors"
                                >
                                    <Download className="w-3.5 h-3.5" aria-hidden="true" />
                                    Import C3D / OpenCap
                                </Link>
                            </div>
                        </div>
                    )}
                </section>

                {/* Biomechanical Modeling Pipeline Cards (#14259) */}
                <section aria-label="Biomechanical Modeling Pipeline" className="mb-8">
                    <div className="flex items-center justify-between mb-3">
                        <h2 className="text-sm font-semibold text-token-text uppercase tracking-wider flex items-center gap-2">
                            <Activity className="w-4 h-4 text-primary" aria-hidden="true" />
                            Biomechanical Modeling Pipeline
                        </h2>
                        <span className="text-xs text-token-muted hidden sm:inline">Capture → Estimate → Model & Match → Optimize → Compare</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                        {PIPELINE_STEPS.map((step) => (
                            <Link
                                key={step.id}
                                to={step.route}
                                className="group relative flex flex-col justify-between p-3.5 rounded-xl border border-token-border bg-surface hover:bg-surface-raised hover:border-primary/50 transition-all hover:-translate-y-0.5"
                            >
                                <div>
                                    <div className="flex items-center justify-between mb-1.5">
                                        <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-primary/20 text-primary">
                                            STEP {step.step}
                                        </span>
                                        <ChevronRight className="w-3.5 h-3.5 text-token-muted group-hover:text-primary transition-colors" aria-hidden="true" />
                                    </div>
                                    <h3 className="text-sm font-semibold text-white group-hover:text-primary transition-colors">{step.title}</h3>
                                    <p className="text-xs text-token-muted">{step.desc}</p>
                                </div>
                                <div className="mt-3 pt-2 border-t border-token-border flex items-center gap-1.5 text-xs text-token-muted">
                                    <span className={`w-1.5 h-1.5 rounded-full ${step.ready ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                                    <span>{step.prereq}</span>
                                </div>
                            </Link>
                        ))}
                    </div>
                </section>

                {/* Physics Engines & Truthful Readiness Table (#14259) */}
                <section aria-label="Physics Engines & Readiness" className="mb-8 rounded-xl border border-token-border bg-surface p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h2 className="text-sm font-semibold text-token-text uppercase tracking-wider flex items-center gap-2">
                            <Zap className="w-4 h-4 text-primary" aria-hidden="true" />
                            Physics Engines & Readiness
                        </h2>
                        <span className="text-xs text-token-muted">Live backend availability status</span>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm text-token-text">
                            <thead>
                                <tr className="border-b border-token-border text-xs font-semibold uppercase text-token-muted">
                                    <th scope="col" className="py-2 pr-4">Engine</th>
                                    <th scope="col" className="py-2 pr-4 hidden sm:table-cell">Specialization</th>
                                    <th scope="col" className="py-2 pr-4">Status</th>
                                    <th scope="col" className="py-2 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-token-border">
                                {ENGINE_ROWS.map((row) => {
                                    const readiness = getEngineReadiness(row, engineStatuses);
                                    return (
                                        <tr key={row.id} className="hover:bg-white/[0.02]">
                                            <td className="py-2.5 pr-4 font-medium text-white">{row.name}</td>
                                            <td className="py-2.5 pr-4 text-xs text-token-muted hidden sm:table-cell">{row.type}</td>
                                            <td className="py-2.5 pr-4">
                                                <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${readiness.color}`}>
                                                    {readiness.status}
                                                </span>
                                            </td>
                                            <td className="py-2.5 text-right">
                                                {readiness.canOpen ? (
                                                    <button
                                                        type="button"
                                                        onClick={() => onLaunchTile(row.tileId)}
                                                        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md bg-primary/30 text-token-text hover:bg-primary/50 border border-primary/40 transition-colors"
                                                    >
                                                        Open
                                                    </button>
                                                ) : (
                                                    <button
                                                        type="button"
                                                        onClick={onShowHelp}
                                                        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md bg-surface-muted text-token-text hover:bg-surface-raised border border-token-border transition-colors"
                                                    >
                                                        How to install
                                                    </button>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* Search & All Tools (#14259) */}
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <div className="relative flex-1 min-w-[240px] max-w-md">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-token-muted pointer-events-none" aria-hidden="true" />
                        <input
                            type="search"
                            role="searchbox"
                            aria-label="Search tools"
                            placeholder="Search all 79 tools and utilities..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-9 pr-3 py-2 bg-surface-raised border border-token-border rounded-lg text-sm text-white placeholder-token-muted focus:outline-none focus:ring-2 focus:ring-primary"
                        />
                    </div>
                    <span className="text-xs text-token-muted">
                        Showing {filteredTiles.length} of {tiles.length} tools
                    </span>
                </div>

                {tileSections.map((section) => {
                    const Icon = section.icon;
                    return (
                        <section key={section.id} aria-label={section.ariaLabel} className="mb-9">
                            <div className="mb-4 flex items-center gap-3">
                                <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-2 whitespace-nowrap">
                                    <Icon className="w-4 h-4" aria-hidden="true" />
                                    {section.title}
                                </h2>
                                <div
                                    className="h-px flex-1 bg-gradient-to-r from-white/20 via-white/10 to-transparent"
                                    aria-hidden="true"
                                />
                            </div>
                            <div
                                className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2 sm:gap-3 md:gap-4"
                                role="group"
                                aria-label={section.groupLabel}
                                id={section.id}
                            >
                                {section.tiles.map((tile) => (
                                    <TileCard
                                        key={tile.id}
                                        tile={tile}
                                        isSelected={selectedTileId === tile.id}
                                        launchAction={resolveTileLaunchAction(tile, nativeWindowAllowed)}
                                        onSelect={() => onSelectTile(tile.id)}
                                        onLaunch={() => onLaunchTile(tile.id)}
                                    />
                                ))}
                            </div>
                        </section>
                    );
                })}
            </main>

            {/* Sticky bottom bar — Launch button always visible (fixes #1165) */}
            <footer
                className={`flex items-center justify-between gap-3 pl-6 ${DIAGNOSTICS_BADGE_CLEARANCE} py-3 bg-gray-800/95 border-t border-gray-700 flex-shrink-0 backdrop-blur-sm`}
                id="launch-footer"
            >
                <div className="flex items-center gap-3 text-sm text-gray-400">
                    {selectedTile ? (
                        <span>
                            Selected: <strong className="text-white">{selectedTile.name}</strong>
                            <span className="ml-2 text-xs text-gray-400">
                                {selectedTileAction?.kind === 'blocked'
                                    ? selectedTileAction.reason
                                    : selectedTile.description}
                            </span>
                        </span>
                    ) : (
                        <span className="text-gray-400">Select a tile to launch</span>
                    )}
                    {onShowAbout && (
                        <button
                            id="about-link"
                            onClick={onShowAbout}
                            aria-label="About UpstreamDrift"
                            className="text-xs text-gray-400 underline-offset-2 hover:text-gray-300 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-400 rounded"
                        >
                            About
                        </button>
                    )}
                </div>
                <button
                    id="launch-simulation-button"
                    onClick={() => selectedTile && selectedLaunchable && onLaunchTile(selectedTile.id)}
                    disabled={!selectedTile || !selectedLaunchable}
                    aria-label={selectedTile ? `Launch ${selectedTile.name}` : 'Launch Simulation'}
                    title={
                        selectedTileAction?.kind === 'blocked'
                            ? selectedTileAction.reason
                            : undefined
                    }
                    className={`
            inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-sm transition-all
            focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-offset-2 focus:ring-offset-gray-900
            ${selectedTile && selectedLaunchable
                            ? 'bg-green-600 hover:bg-green-500 text-white shadow-lg shadow-green-600/30 hover:shadow-green-500/40'
                            : 'bg-gray-700 text-gray-400 cursor-not-allowed'
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
