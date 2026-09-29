/**
 * Dashboard Page — Main entry point for the Tauri launcher.
 *
 * Wraps the LauncherDashboard component with the manifest hook
 * and navigation logic.
 */

import { useState, useCallback, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router';
import { useLauncherManifest } from '@/api/useLauncherManifest';
import { resolveTileLaunchAction } from '@/api/webLaunch';
import { LauncherDashboard } from '@/components/simulation/LauncherDashboard';
import { AboutModal } from '@/components/ui/AboutModal';
import { OnboardingOverlay } from '@/components/ui/OnboardingOverlay';
import { useToast } from '@/components/ui/Toast';
import { apiFetch } from '@/api/fetch';
import {
    loadLauncherWindowRecords,
    persistLauncherWindowRecords,
    reconcileLauncherWindowRecords,
    recordLauncherWindowLaunch,
} from '@/api/launcherWindowRegistry';

export function DashboardPage() {
    const {
        tiles,
        launcherCsrfToken,
        launcherCsrfHeader,
        loadState,
        error,
        refetch,
    } = useLauncherManifest();
    const [selectedTileId, setSelectedTileId] = useState<string | null>(null);
    const [aboutOpen, setAboutOpen] = useState(false);
    const [launchedWindows, setLaunchedWindows] = useState(() => loadLauncherWindowRecords());
    const visibleLaunchedWindows = useMemo(
        () => reconcileLauncherWindowRecords(launchedWindows, tiles),
        [launchedWindows, tiles]
    );
    const { showInfo, showError } = useToast();
    const navigate = useNavigate();

    useEffect(() => {
        persistLauncherWindowRecords(visibleLaunchedWindows);
    }, [visibleLaunchedWindows]);

    const handleLaunchTile = useCallback(
        (tileId: string) => {
            const tile = tiles.find((t) => t.id === tileId);
            if (!tile) {
                showError('Tile not found');
                return;
            }

            // Resolve the tile's web launch contract (issue #7461):
            //   route → in-app navigation; native-window → backend POST
            //   (Tauri/localhost only); blocked → honest error, no dead POST.
            const action = resolveTileLaunchAction(tile);
            if (action.kind === 'navigate') {
                navigate(action.route);
                return;
            }
            if (action.kind === 'blocked') {
                showError(`${tile.name}: ${action.reason}`);
                return;
            }

            // Launch all engines/tools as subprocesses via the backend API
            showInfo(`Launching ${tile.name}...`);
            apiFetch<{ name?: string }>(`/api/launcher/launch/${tile.id}`, {
                method: 'POST',
                headers: launcherCsrfToken ? { [launcherCsrfHeader]: launcherCsrfToken } : {},
            })
                .then((data) => {
                    setLaunchedWindows(recordLauncherWindowLaunch(tile));
                    showInfo(`${data.name || tile.name} launched successfully`);
                })
                .catch((err) => {
                    showError(`Failed to launch ${tile.name}: ${err.message}`);
                });
        },
        [tiles, launcherCsrfToken, launcherCsrfHeader, showInfo, showError, navigate]
    );

    const handleFocusLaunchedTile = useCallback(
        (tileId: string) => {
            handleLaunchTile(tileId);
        },
        [handleLaunchTile]
    );

    const handleShowHelp = useCallback(() => {
        showInfo('Help system opening...');
        // Future: open HelpDialog component
    }, [showInfo]);

    return (
        <>
            <LauncherDashboard
                tiles={tiles}
                loadState={loadState}
                error={error}
                selectedTileId={selectedTileId}
                launchedWindows={visibleLaunchedWindows}
                onSelectTile={setSelectedTileId}
                onLaunchTile={handleLaunchTile}
                onFocusLaunchedTile={handleFocusLaunchedTile}
                onShowHelp={handleShowHelp}
                onShowAbout={() => setAboutOpen(true)}
                onRefetch={refetch}
            />
            <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
            <OnboardingOverlay />
        </>
    );
}
