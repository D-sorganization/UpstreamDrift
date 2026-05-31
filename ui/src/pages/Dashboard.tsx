/**
 * Dashboard Page — Main entry point for the Tauri launcher.
 *
 * Wraps the LauncherDashboard component with the manifest hook
 * and navigation logic.
 */

import { useState, useCallback } from 'react';
import { useLauncherManifest } from '@/api/useLauncherManifest';
import { LauncherDashboard } from '@/components/simulation/LauncherDashboard';
import { useToast } from '@/components/ui/Toast';
import { apiFetch } from '@/api/fetch';

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
    const { showInfo, showError } = useToast();

    const handleLaunchTile = useCallback(
        (tileId: string) => {
            const tile = tiles.find((t) => t.id === tileId);
            if (!tile) {
                showError('Tile not found');
                return;
            }

            // Launch all engines/tools as subprocesses via the backend API
            showInfo(`Launching ${tile.name}...`);
            apiFetch<{ name?: string }>(`/api/launcher/launch/${tile.id}`, {
                method: 'POST',
                headers: launcherCsrfToken ? { [launcherCsrfHeader]: launcherCsrfToken } : {},
            })
                .then((data) => {
                    showInfo(`${data.name || tile.name} launched successfully`);
                })
                .catch((err) => {
                    showError(`Failed to launch ${tile.name}: ${err.message}`);
                });
        },
        [tiles, launcherCsrfToken, launcherCsrfHeader, showInfo, showError]
    );

    const handleShowHelp = useCallback(() => {
        showInfo('Help system opening...');
        // Future: open HelpDialog component
    }, [showInfo]);

    return (
        <LauncherDashboard
            tiles={tiles}
            loadState={loadState}
            error={error}
            selectedTileId={selectedTileId}
            onSelectTile={setSelectedTileId}
            onLaunchTile={handleLaunchTile}
            onShowHelp={handleShowHelp}
            onRefetch={refetch}
        />
    );
}
