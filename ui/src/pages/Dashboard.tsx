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

export function DashboardPage() {
    const { tiles, loadState, error, refetch } = useLauncherManifest();
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
            fetch(`/api/launcher/launch/${tile.id}`, { method: 'POST' })
                .then((res) => {
                    if (!res.ok) {
                        return res.json().then((body) => {
                            throw new Error(body.detail || `HTTP ${res.status}`);
                        });
                    }
                    return res.json();
                })
                .then((data) => {
                    showInfo(`${data.name || tile.name} launched successfully`);
                })
                .catch((err) => {
                    showError(`Failed to launch ${tile.name}: ${err.message}`);
                });
        },
        [tiles, showInfo, showError]
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
