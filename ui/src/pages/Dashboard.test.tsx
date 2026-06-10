/**
 * TDD Tests for Dashboard page.
 *
 * Tests:
 *   - Page renders the LauncherDashboard
 *   - Navigation to simulation page works
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import { fireEvent, screen, waitFor } from '@testing-library/dom';
import { MemoryRouter } from 'react-router-dom';
import { ToastProvider } from '@/components/ui/Toast';
import {
    LAUNCHER_WINDOW_REGISTRY_KEY,
    loadLauncherWindowRecords,
} from '@/api/launcherWindowRegistry';

function installMemoryStorage(): void {
    const values = new Map<string, string>();
    Object.defineProperty(window, 'localStorage', {
        configurable: true,
        value: {
            getItem: (key: string) => values.get(key) ?? null,
            setItem: (key: string, value: string) => {
                values.set(key, value);
            },
            removeItem: (key: string) => {
                values.delete(key);
            },
            clear: () => values.clear(),
        },
    });
}

// Mock the manifest hook
const mockManifest = {
    tiles: [
        {
            id: 'mujoco_unified',
            name: 'MuJoCo',
            description: 'MuJoCo simulation',
            category: 'physics_engine' as const,
            type: 'custom_humanoid',
            path: 'src/mujoco.py',
            logo: 'mujoco.png',
            status: 'gui_ready',
            capabilities: ['rigid_body'],
            order: 2,
            engine_type: 'mujoco',
        },
        {
            id: 'model_explorer',
            name: 'Model Explorer',
            description: 'Browse models',
            category: 'tool' as const,
            type: 'special_app',
            path: 'src/tools/urdf.py',
            logo: 'urdf_icon.png',
            status: 'utility',
            capabilities: ['model_browsing'],
            order: 1,
        },
    ],
    engines: [],
    tools: [],
    loadState: 'loaded' as const,
    error: null,
    manifest: null,
    launcherCsrfToken: 'test-token',
    launcherCsrfHeader: 'X-Launcher-CSRF-Token',
    refetch: vi.fn(),
};

vi.mock('@/api/useLauncherManifest', () => ({
    useLauncherManifest: vi.fn(() => mockManifest),
}));

// Now import after mocking
import { DashboardPage } from './Dashboard';

const renderWithRouter = (ui: React.ReactElement) =>
    render(
        <MemoryRouter>
            <ToastProvider>{ui}</ToastProvider>
        </MemoryRouter>
    );

describe('DashboardPage', () => {
    const originalFetch = global.fetch;

    beforeEach(() => {
        vi.clearAllMocks();
        installMemoryStorage();
        window.localStorage.removeItem(LAUNCHER_WINDOW_REGISTRY_KEY);
        global.fetch = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) });
    });

    afterEach(() => {
        global.fetch = originalFetch;
    });

    it('renders the dashboard with tiles', () => {
        renderWithRouter(<DashboardPage />);

        expect(screen.getByText('Golf Modeling Suite')).toBeInTheDocument();
        expect(screen.getByText('MuJoCo')).toBeInTheDocument();
        expect(screen.getByText('Model Explorer')).toBeInTheDocument();
    });

    it('renders help button', () => {
        renderWithRouter(<DashboardPage />);
        expect(screen.getByRole('button', { name: /help/i })).toBeInTheDocument();
    });

    it('renders launch button', () => {
        renderWithRouter(<DashboardPage />);
        expect(screen.getByRole('button', { name: /launch simulation/i })).toBeInTheDocument();
    });

    it('persists launched windows after a successful launch', async () => {
        renderWithRouter(<DashboardPage />);

        fireEvent.click(screen.getByRole('button', { name: /model explorer/i }));
        fireEvent.click(screen.getByRole('button', { name: /launch model explorer/i }));

        await waitFor(() => {
            expect(loadLauncherWindowRecords()).toEqual([
                expect.objectContaining({
                    tileId: 'model_explorer',
                    name: 'Model Explorer',
                    launchCount: 1,
                }),
            ]);
        });
    });
});
