/**
 * TDD Tests for LauncherDashboard component.
 *
 * Tests:
 *   - Tile grid rendering (all tiles appear)
 *   - Category sections (engines vs tools)
 *   - Status chips for all tile types (#1168)
 *   - Model Explorer is first tile
 *   - Launch button always visible (#1165)
 *   - Help button prominently displayed (#1170)
 *   - Tile selection and launch
 *   - Loading and error states
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen, fireEvent, within } from '@testing-library/dom';
import { MemoryRouter } from 'react-router-dom';
import { LauncherDashboard } from './LauncherDashboard';
import type { LauncherTile } from '@/api/useLauncherManifest';

const renderWithRouter = (ui: React.ReactElement) =>
    render(<MemoryRouter>{ui}</MemoryRouter>);

const MOCK_TILES: LauncherTile[] = [
    {
        id: 'model_explorer',
        name: 'Model Explorer',
        description: 'Browse models',
        category: 'tool',
        type: 'special_app',
        path: 'src/tools/urdf_generator/launch_urdf_generator.py',
        logo: 'urdf_icon.png',
        status: 'utility',
        capabilities: ['model_browsing', 'urdf_generation'],
        order: 1,
    },
    {
        id: 'mujoco_unified',
        name: 'MuJoCo',
        description: 'MuJoCo simulation',
        category: 'physics_engine',
        type: 'custom_humanoid',
        path: 'src/launchers/mujoco.py',
        logo: 'mujoco_humanoid.png',
        status: 'gui_ready',
        capabilities: ['rigid_body', 'contact'],
        order: 2,
        engine_type: 'mujoco',
    },
    {
        id: 'drake_golf',
        name: 'Drake',
        description: 'Drake dynamics',
        category: 'physics_engine',
        type: 'drake',
        path: 'src/engines/drake.py',
        logo: 'drake.png',
        status: 'gui_ready',
        capabilities: ['rigid_body', 'optimization'],
        order: 3,
        engine_type: 'drake',
    },
    {
        id: 'putting_green',
        name: 'Putting Green',
        description: 'putting simulation',
        category: 'physics_engine',
        type: 'putting_green',
        path: 'src/putting_green.py',
        logo: 'putting_green.png',
        status: 'simulator',
        capabilities: ['ball_physics'],
        order: 7,
        engine_type: 'putting_green',
    },
    {
        id: 'motion_capture',
        name: 'Motion Capture',
        description: 'C3D Viewer, OpenPose, MediaPipe',
        category: 'tool',
        type: 'special_app',
        path: 'src/launchers/motion_capture.py',
        logo: 'c3d_icon.png',
        status: 'utility',
        capabilities: ['c3d_viewer', 'openpose', 'mediapipe'],
        order: 9,
    },
    {
        id: 'matlab_unified',
        name: 'Matlab Models',
        description: 'Simscape models',
        category: 'external',
        type: 'special_app',
        path: 'src/launchers/matlab.py',
        logo: 'matlab_logo.png',
        status: 'external',
        capabilities: ['simscape_2d', 'simscape_3d'],
        order: 8,
    },
];

describe('LauncherDashboard', () => {
    const defaultProps = {
        tiles: MOCK_TILES,
        loadState: 'loaded' as const,
        error: null,
        selectedTileId: null,
        launchedWindows: [],
        onSelectTile: vi.fn(),
        onLaunchTile: vi.fn(),
        onFocusLaunchedTile: vi.fn(),
        onShowHelp: vi.fn(),
        onRefetch: vi.fn(),
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    // ────────────────────────────────────────────────────────────
    // Tile Grid Rendering (#1171)
    // ────────────────────────────────────────────────────────────
    describe('tile grid rendering (#1171)', () => {
    it('renders all tiles from the manifest', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);

            expect(screen.getByText('MuJoCo')).toBeInTheDocument();
            expect(screen.getByText('Drake')).toBeInTheDocument();
            expect(screen.getByText('Putting Green')).toBeInTheDocument();
            expect(screen.getByText('Model Explorer')).toBeInTheDocument();
            expect(screen.getByText('Motion Capture')).toBeInTheDocument();
            expect(screen.getByText('Matlab Models')).toBeInTheDocument();
        });

    it('renders category sections', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);

            expect(screen.getByRole('region', { name: /core physics engines/i })).toBeInTheDocument();
            expect(screen.getByRole('region', { name: /analysis tools/i })).toBeInTheDocument();
            expect(screen.getByRole('region', { name: /utilities/i })).toBeInTheDocument();
        });

    it('groups engines separately from tools', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);

            const engineGrid = screen.getByRole('group', { name: /core physics engine tiles/i });
            const analysisGrid = screen.getByRole('group', { name: /analysis tool tiles/i });
            const utilityGrid = screen.getByRole('group', { name: /utility tiles/i });

            expect(within(engineGrid).getByText('MuJoCo')).toBeInTheDocument();
            expect(within(analysisGrid).getByText('Motion Capture')).toBeInTheDocument();
            expect(within(utilityGrid).getByText('Model Explorer')).toBeInTheDocument();
            expect(within(utilityGrid).getByText('Matlab Models')).toBeInTheDocument();
        });

    it('uses glass card styling with hover elevation', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);

            const tile = document.getElementById('tile-mujoco_unified')!;
            expect(tile.className).toContain('backdrop-blur');
            expect(tile.className).toContain('shadow-black');
            expect(tile.className).toContain('hover:shadow-blue');
        });
    });

    // ────────────────────────────────────────────────────────────
    // Status Chips (#1168)
    // ────────────────────────────────────────────────────────────
    describe('status chips (#1168)', () => {
    it('shows GUI Ready for gui_ready engines', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getAllByText('GUI Ready').length).toBeGreaterThan(0);
        });

    it('shows Utility for special_app tools', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getAllByText('Utility').length).toBeGreaterThan(0);
        });

    it('shows Simulator for putting_green', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByText('Simulator')).toBeInTheDocument();
        });

    it('shows External for matlab', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByText('External')).toBeInTheDocument();
        });
    });

    // ────────────────────────────────────────────────────────────
    // Launch Button Accessibility (#1165)
    // ────────────────────────────────────────────────────────────
    describe('launch button always visible (#1165)', () => {
    it('renders the launch button', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByRole('button', { name: /launch simulation/i })).toBeInTheDocument();
        });

    it('launch button is in a sticky footer', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            const footer = document.getElementById('launch-footer');
            expect(footer).not.toBeNull();
            expect(footer?.classList.contains('flex-shrink-0')).toBe(true);
        });

    it('launch button is disabled when no tile selected', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} selectedTileId={null} />);
            expect(screen.getByRole('button', { name: /launch simulation/i })).toBeDisabled();
        });

    it('launch button is enabled when tile selected', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} selectedTileId="mujoco_unified" />);
            expect(screen.getByRole('button', { name: /launch mujoco/i })).not.toBeDisabled();
        });

    it('clicking launch button calls onLaunchTile', () => {
        const onLaunchTile = vi.fn();
        renderWithRouter(
            <LauncherDashboard
                {...defaultProps}
                selectedTileId="mujoco_unified"
                onLaunchTile={onLaunchTile}
            />
        );

            fireEvent.click(screen.getByRole('button', { name: /launch mujoco/i }));
            expect(onLaunchTile).toHaveBeenCalledWith('mujoco_unified');
        });
    });

    // ────────────────────────────────────────────────────────────
    // Help Button (#1170)
    // ────────────────────────────────────────────────────────────
    describe('help button (#1170)', () => {
    it('renders a prominent help button', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByRole('button', { name: /help/i })).toBeInTheDocument();
        });

    it('help button has visible text (not icon-only)', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            const helpBtn = screen.getByRole('button', { name: /help/i });
            expect(helpBtn.textContent).toContain('Help');
        });

    it('clicking help button calls onShowHelp', () => {
        const onShowHelp = vi.fn();
        renderWithRouter(<LauncherDashboard {...defaultProps} onShowHelp={onShowHelp} />);

            fireEvent.click(screen.getByRole('button', { name: /help/i }));
            expect(onShowHelp).toHaveBeenCalled();
        });
    });

    // ────────────────────────────────────────────────────────────
    // Tile Selection
    // ────────────────────────────────────────────────────────────
    describe('tile selection', () => {
    it('clicking a tile calls onSelectTile', () => {
        const onSelectTile = vi.fn();
        renderWithRouter(<LauncherDashboard {...defaultProps} onSelectTile={onSelectTile} />);

            const tile = document.getElementById('tile-mujoco_unified')!;
            fireEvent.click(tile);
            expect(onSelectTile).toHaveBeenCalledWith('mujoco_unified');
        });

    it('selected tile has aria-pressed=true', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} selectedTileId="drake_golf" />);
            const tile = document.getElementById('tile-drake_golf')!;
            expect(tile).toHaveAttribute('aria-pressed', 'true');
        });

    it('double-clicking a tile calls onLaunchTile', () => {
        const onLaunchTile = vi.fn();
        renderWithRouter(<LauncherDashboard {...defaultProps} onLaunchTile={onLaunchTile} />);

            const tile = document.getElementById('tile-mujoco_unified')!;
            fireEvent.doubleClick(tile);
            expect(onLaunchTile).toHaveBeenCalledWith('mujoco_unified');
        });

    it('shows selected tile name in footer', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} selectedTileId="mujoco_unified" />);
            const footer = document.getElementById('launch-footer');
            expect(footer?.textContent).toContain('MuJoCo');
        });
    });

    // ────────────────────────────────────────────────────────────
    // Loading & Error States
    // ────────────────────────────────────────────────────────────
    describe('loading and error states', () => {
    it('shows loading spinner', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} loadState="loading" tiles={[]} />);
            expect(screen.getByRole('status', { name: /loading/i })).toBeInTheDocument();
        });

    it('shows error with retry button', () => {
        renderWithRouter(
            <LauncherDashboard
                {...defaultProps}
                loadState="error"
                error="Connection refused"
                tiles={[]}
            />
        );
            expect(screen.getByRole('alert')).toBeInTheDocument();
            expect(screen.getByText('Connection refused')).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
        });

    it('clicking Retry calls onRefetch', () => {
        const onRefetch = vi.fn();
        renderWithRouter(
            <LauncherDashboard
                {...defaultProps}
                loadState="error"
                error="fail"
                tiles={[]}
                onRefetch={onRefetch}
            />
        );

            fireEvent.click(screen.getByRole('button', { name: /retry/i }));
            expect(onRefetch).toHaveBeenCalled();
        });
    });

    // ────────────────────────────────────────────────────────────
    // Missing Tiles Parity (#1162)
    // ────────────────────────────────────────────────────────────
    describe('all required tiles present (#1162)', () => {
    it('renders Motion Capture tile', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByText('Motion Capture')).toBeInTheDocument();
        });

    it('renders Matlab Models tile', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByText('Matlab Models')).toBeInTheDocument();
        });

    it('renders Model Explorer tile', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByText('Model Explorer')).toBeInTheDocument();
        });

    it('renders Putting Green tile', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByText('Putting Green')).toBeInTheDocument();
        });
    });

    // ────────────────────────────────────────────────────────────
    // Header
    // ────────────────────────────────────────────────────────────
    describe('header', () => {
    it('shows application title', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByText('Golf Modeling Suite')).toBeInTheDocument();
        });

        it('shows tile counts', () => {
        renderWithRouter(<LauncherDashboard {...defaultProps} />);
            expect(screen.getByText(/6 tiles/)).toBeInTheDocument();
            expect(screen.getByText(/3 engines/)).toBeInTheDocument();
        });
    });

    describe('launched window menu (#7221)', () => {
        it('shows an empty window list state', () => {
            renderWithRouter(<LauncherDashboard {...defaultProps} />);

            fireEvent.click(screen.getByRole('button', { name: /window list/i }));

            expect(screen.getByRole('menu', { name: /launched windows/i })).toBeInTheDocument();
            expect(screen.getByText('No launched windows')).toBeInTheDocument();
        });

        it('renders launched windows and focuses by tile id', () => {
            const onFocusLaunchedTile = vi.fn();
            renderWithRouter(
                <LauncherDashboard
                    {...defaultProps}
                    launchedWindows={[
                        {
                            tileId: 'model_explorer',
                            name: 'Model Explorer',
                            launchedAt: '2026-06-10T12:00:00.000Z',
                            focusedAt: '2026-06-10T12:00:00.000Z',
                            launchCount: 2,
                        },
                    ]}
                    onFocusLaunchedTile={onFocusLaunchedTile}
                />
            );

            fireEvent.click(screen.getByRole('button', { name: /window list/i }));
            const menu = screen.getByRole('menu', { name: /launched windows/i });
            fireEvent.click(within(menu).getByRole('button', { name: /focus/i }));

            expect(within(menu).getByText('Model Explorer')).toBeInTheDocument();
            expect(within(menu).getByText('2 launches')).toBeInTheDocument();
            expect(onFocusLaunchedTile).toHaveBeenCalledWith('model_explorer');
        });
    });
});
