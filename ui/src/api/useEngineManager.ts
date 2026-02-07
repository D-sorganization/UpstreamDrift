/**
 * Engine Manager Hook — Lazy Engine Loading
 *
 * Manages a registry of known physics engines with on-demand loading.
 * Engines start in an "idle" state and are only loaded when the user
 * explicitly requests it, preventing heavy resource consumption on startup.
 */

import { useState, useCallback } from 'react';
import type { EngineStatus } from './client';

export type EngineLoadState = 'idle' | 'loading' | 'loaded' | 'error';

export interface ManagedEngine {
    name: string;
    displayName: string;
    description: string;
    loadState: EngineLoadState;
    available: boolean;
    version?: string;
    capabilities: string[];
    error?: string;
}

/** Static registry of known engines — no backend call needed to display these. */
export const ENGINE_REGISTRY: Omit<ManagedEngine, 'loadState' | 'available' | 'error'>[] = [
    {
        name: 'mujoco',
        displayName: 'MuJoCo',
        description: 'High-performance physics for robotics and biomechanics',
        capabilities: ['rigid_body', 'contact', 'tendons', 'actuators'],
    },
    {
        name: 'drake',
        displayName: 'Drake',
        description: 'Optimization-based multibody dynamics',
        capabilities: ['rigid_body', 'optimization', 'contact'],
    },
    {
        name: 'pinocchio',
        displayName: 'Pinocchio',
        description: 'Rigid body dynamics algorithms',
        capabilities: ['rigid_body', 'inverse_kinematics'],
    },
    {
        name: 'opensim',
        displayName: 'OpenSim',
        description: 'Musculoskeletal modeling and biomechanics simulation',
        capabilities: ['musculoskeletal', 'inverse_kinematics', 'muscle_analysis'],
    },
    {
        name: 'myosuite',
        displayName: 'MyoSuite',
        description: 'Muscle-tendon control and neural activation',
        capabilities: ['musculoskeletal', 'muscle_control', 'neural_activation'],
    },
    {
        name: 'putting_green',
        displayName: 'Putting Green',
        description: 'Golf putting green with ball roll physics',
        capabilities: ['surface_modeling', 'ball_physics', 'terrain'],
    },
];

async function probeEngine(engineName: string): Promise<EngineStatus> {
    const response = await fetch(`/api/engines/${engineName}/probe`);
    if (!response.ok) {
        throw new Error(`Failed to probe engine: ${engineName}`);
    }
    return response.json();
}

async function loadEngine(engineName: string): Promise<EngineStatus> {
    const response = await fetch(`/api/engines/${engineName}/load`, {
        method: 'POST',
    });
    if (!response.ok) {
        throw new Error(`Failed to load engine: ${engineName}`);
    }
    return response.json();
}

export function useEngineManager() {
    const [engines, setEngines] = useState<ManagedEngine[]>(() =>
        ENGINE_REGISTRY.map((e) => ({
            ...e,
            loadState: 'idle' as EngineLoadState,
            available: true, // Assume available until probed
        }))
    );

    const getEngine = useCallback(
        (name: string) => engines.find((e) => e.name === name),
        [engines]
    );

    const loadedEngines = engines.filter((e) => e.loadState === 'loaded');

    const requestLoad = useCallback(async (engineName: string) => {
        // Set to loading
        setEngines((prev) =>
            prev.map((e) =>
                e.name === engineName ? { ...e, loadState: 'loading' as EngineLoadState, error: undefined } : e
            )
        );

        try {
            // First probe, then load
            const probeResult = await probeEngine(engineName);

            if (!probeResult.available) {
                setEngines((prev) =>
                    prev.map((e) =>
                        e.name === engineName
                            ? { ...e, loadState: 'error' as EngineLoadState, available: false, error: 'Engine not installed on this system' }
                            : e
                    )
                );
                return;
            }

            const loadResult = await loadEngine(engineName);

            setEngines((prev) =>
                prev.map((e) =>
                    e.name === engineName
                        ? {
                            ...e,
                            loadState: 'loaded' as EngineLoadState,
                            available: true,
                            version: loadResult.version,
                            capabilities: loadResult.capabilities || e.capabilities,
                        }
                        : e
                )
            );
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error loading engine';
            setEngines((prev) =>
                prev.map((e) =>
                    e.name === engineName ? { ...e, loadState: 'error' as EngineLoadState, error: message } : e
                )
            );
        }
    }, []);

    const unloadEngine = useCallback((engineName: string) => {
        setEngines((prev) =>
            prev.map((e) =>
                e.name === engineName
                    ? { ...e, loadState: 'idle' as EngineLoadState, error: undefined }
                    : e
            )
        );
    }, []);

    return {
        engines,
        loadedEngines,
        getEngine,
        requestLoad,
        unloadEngine,
    };
}
