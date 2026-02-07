/**
 * Dashboard Page — Main entry point for the Tauri launcher.
 *
 * Wraps the LauncherDashboard component with the manifest hook
 * and navigation logic.
 */

import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLauncherManifest } from '@/api/useLauncherManifest';
import { LauncherDashboard } from '@/components/simulation/LauncherDashboard';
import { useToast } from '@/components/ui/Toast';

export function DashboardPage() {
    const navigate = useNavigate();
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

            // Physics engines navigate to the simulation page with the engine preselected
            if (tile.category === 'physics_engine' && tile.engine_type) {
                showInfo(`Opening ${tile.name} simulation...`);
                navigate(`/simulation?engine=${tile.engine_type}`);
                return;
            }

            // Tools and external apps are launched via the backend API
            showInfo(`Launching ${tile.name}...`);
            fetch(`/api/launcher/launch/${tile.id}`, { method: 'POST' }).catch(() => {
                showError(`Failed to launch ${tile.name}`);
            });
        },
        [tiles, navigate, showInfo, showError]
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
