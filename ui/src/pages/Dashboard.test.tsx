/// <reference types="vitest/globals" />
/**
 * TDD Tests for Dashboard page.
 *
 * Tests:
 *   - Page renders the LauncherDashboard
 *   - Navigation to simulation page works
 */

import { vi } from 'vitest';
import { render } from '@testing-library/react';
import { fireEvent, screen, waitFor } from '@testing-library/dom';
import { MemoryRouter } from 'react-router';
import { HelpPanel } from '@/components/ui/HelpPanel';
import { useUIStore } from '@/stores';
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

function GlobalHelpPanel() {
    const isOpen = useUIStore((state) => state.helpOpen);
    const setHelpOpen = useUIStore((state) => state.setHelpOpen);
    return (
        <HelpPanel
            isOpen={isOpen}
            onOpen={() => setHelpOpen(true)}
            onClose={() => setHelpOpen(false)}
        />
    );
}

// Now import after mocking
import { DashboardPage } from './Dashboard';

const renderWithRouter = (ui: React.ReactElement) =>
    render(
        <MemoryRouter>
            <ToastProvider>
                {ui}
                <GlobalHelpPanel />
            </ToastProvider>
        </MemoryRouter>
    );

describe('DashboardPage', () => {
    const originalFetch = global.fetch;

    beforeEach(() => {
        vi.clearAllMocks();
        useUIStore.getState().setHelpOpen(false);
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

    it('renders persistent workspace navigation sidebar matching workspaces (#14259)', () => {
        renderWithRouter(<DashboardPage />);
        expect(screen.getByRole('navigation', { name: 'Workspace Navigation' })).toBeInTheDocument();
    });

    it('renders help button', () => {
        renderWithRouter(<DashboardPage />);
        expect(screen.getByRole('button', { name: /help/i })).toBeInTheDocument();
    });

    it('opens the help panel from the dashboard button and restores focus on Escape', () => {
        renderWithRouter(<DashboardPage />);
        const helpButton = screen.getByRole('button', { name: /help/i });
        helpButton.focus();

        fireEvent.click(helpButton);

        const helpPanel = screen.getByRole('dialog', { name: 'Help panel' });
        expect(helpPanel).toBeInTheDocument();
        fireEvent.keyDown(helpPanel, { key: 'Escape' });

        expect(screen.queryByRole('dialog', { name: 'Help panel' })).not.toBeInTheDocument();
        expect(helpButton).toHaveFocus();
    });

    it('opens the help panel with F1', () => {
        renderWithRouter(<DashboardPage />);

        fireEvent.keyDown(window, { key: 'F1' });

        expect(screen.getByRole('dialog', { name: 'Help panel' })).toBeInTheDocument();
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
