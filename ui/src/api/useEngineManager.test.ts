import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useEngineManager, ENGINE_REGISTRY } from './useEngineManager';

// Mock fetch globally
const mockFetch = vi.fn() as Mock;
vi.stubGlobal('fetch', mockFetch);

describe('useEngineManager', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockFetch.mockReset();
    });

    describe('initialization', () => {
        it('initializes all engines from the static registry', () => {
            const { result } = renderHook(() => useEngineManager());

            expect(result.current.engines).toHaveLength(ENGINE_REGISTRY.length);
            expect(result.current.engines.map((e) => e.name)).toEqual(
                ENGINE_REGISTRY.map((e) => e.name)
            );
        });

        it('all engines start in idle state', () => {
            const { result } = renderHook(() => useEngineManager());

            result.current.engines.forEach((engine) => {
                expect(engine.loadState).toBe('idle');
            });
        });

        it('no engines are loaded on startup', () => {
            const { result } = renderHook(() => useEngineManager());

            expect(result.current.loadedEngines).toHaveLength(0);
        });

        it('all engines are assumed available initially', () => {
            const { result } = renderHook(() => useEngineManager());

            result.current.engines.forEach((engine) => {
                expect(engine.available).toBe(true);
            });
        });
    });

    describe('ENGINE_REGISTRY', () => {
        it('contains MuJoCo', () => {
            const mujoco = ENGINE_REGISTRY.find((e) => e.name === 'mujoco');
            expect(mujoco).toBeDefined();
            expect(mujoco!.displayName).toBe('MuJoCo');
        });

        it('contains Drake', () => {
            const drake = ENGINE_REGISTRY.find((e) => e.name === 'drake');
            expect(drake).toBeDefined();
            expect(drake!.displayName).toBe('Drake');
        });

        it('contains Putting Green', () => {
            const pg = ENGINE_REGISTRY.find((e) => e.name === 'putting_green');
            expect(pg).toBeDefined();
            expect(pg!.displayName).toBe('Putting Green');
        });

        it('each engine has capabilities', () => {
            ENGINE_REGISTRY.forEach((engine) => {
                expect(engine.capabilities.length).toBeGreaterThan(0);
            });
        });
    });

    describe('getEngine', () => {
        it('returns the engine by name', () => {
            const { result } = renderHook(() => useEngineManager());

            const mujoco = result.current.getEngine('mujoco');
            expect(mujoco).toBeDefined();
            expect(mujoco!.name).toBe('mujoco');
        });

        it('returns undefined for unknown engine', () => {
            const { result } = renderHook(() => useEngineManager());

            const unknown = result.current.getEngine('nonexistent');
            expect(unknown).toBeUndefined();
        });
    });

    describe('requestLoad — success path', () => {
        it('transitions engine to loading state', async () => {
            // Mock probe and load endpoints
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: false, capabilities: [] }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: true, version: '3.1.0', capabilities: ['rigid_body'] }),
                });

            const { result } = renderHook(() => useEngineManager());

            // Start loading — check immediate state transition
            act(() => {
                result.current.requestLoad('mujoco');
            });

            // Should be in loading state immediately
            expect(result.current.getEngine('mujoco')!.loadState).toBe('loading');
        });

        it('transitions engine to loaded state after successful load', async () => {
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: false, capabilities: [] }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: true, version: '3.1.0', capabilities: ['rigid_body', 'contact'] }),
                });

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                expect(result.current.getEngine('mujoco')!.loadState).toBe('loaded');
            });
        });

        it('updates version after load', async () => {
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: true, version: '3.1.0', capabilities: [] }),
                });

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                expect(result.current.getEngine('mujoco')!.version).toBe('3.1.0');
            });
        });

        it('adds engine to loadedEngines', async () => {
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: true, version: '3.1.0', capabilities: [] }),
                });

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                expect(result.current.loadedEngines).toHaveLength(1);
                expect(result.current.loadedEngines[0].name).toBe('mujoco');
            });
        });

        it('can load multiple engines', async () => {
            // Load mujoco
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: true, version: '3.1.0', capabilities: [] }),
                });

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            // Load drake
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'drake', available: true }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'drake', available: true, loaded: true, version: '1.0.0', capabilities: [] }),
                });

            await act(async () => {
                await result.current.requestLoad('drake');
            });

            await waitFor(() => {
                expect(result.current.loadedEngines).toHaveLength(2);
            });
        });
    });

    describe('requestLoad — error paths', () => {
        it('sets error state when probe says engine unavailable', async () => {
            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ name: 'pinocchio', available: false }),
            });

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('pinocchio');
            });

            await waitFor(() => {
                const engine = result.current.getEngine('pinocchio')!;
                expect(engine.loadState).toBe('error');
                expect(engine.available).toBe(false);
                expect(engine.error).toContain('not installed');
            });
        });

        it('sets error state when probe network call fails', async () => {
            mockFetch.mockRejectedValueOnce(new Error('Network error'));

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                const engine = result.current.getEngine('mujoco')!;
                expect(engine.loadState).toBe('error');
                expect(engine.error).toContain('Network error');
            });
        });

        it('sets error state when probe returns non-ok response', async () => {
            mockFetch.mockResolvedValueOnce({
                ok: false,
                status: 500,
            });

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                expect(result.current.getEngine('mujoco')!.loadState).toBe('error');
            });
        });

        it('clears previous error on retry', async () => {
            // First attempt: fail
            mockFetch.mockRejectedValueOnce(new Error('timeout'));

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                expect(result.current.getEngine('mujoco')!.loadState).toBe('error');
            });

            // Second attempt: succeed
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: true, version: '3.1.0', capabilities: [] }),
                });

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                const engine = result.current.getEngine('mujoco')!;
                expect(engine.loadState).toBe('loaded');
                expect(engine.error).toBeUndefined();
            });
        });
    });

    describe('unloadEngine', () => {
        it('resets engine to idle state', async () => {
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true, loaded: true, version: '3.1.0', capabilities: [] }),
                });

            const { result } = renderHook(() => useEngineManager());

            // Load first
            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                expect(result.current.getEngine('mujoco')!.loadState).toBe('loaded');
            });

            // Unload
            act(() => {
                result.current.unloadEngine('mujoco');
            });

            expect(result.current.getEngine('mujoco')!.loadState).toBe('idle');
            expect(result.current.loadedEngines).toHaveLength(0);
        });

        it('clears error on unload', async () => {
            mockFetch.mockRejectedValueOnce(new Error('fail'));

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            await waitFor(() => {
                expect(result.current.getEngine('mujoco')!.error).toBeDefined();
            });

            act(() => {
                result.current.unloadEngine('mujoco');
            });

            expect(result.current.getEngine('mujoco')!.error).toBeUndefined();
        });
    });

    describe('API calls', () => {
        it('calls probe endpoint first, then load endpoint', async () => {
            mockFetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', available: true }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ name: 'mujoco', loaded: true, version: '3.1.0', capabilities: [] }),
                });

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('mujoco');
            });

            expect(mockFetch).toHaveBeenCalledTimes(2);
            expect(mockFetch).toHaveBeenNthCalledWith(1, '/api/engines/mujoco/probe');
            expect(mockFetch).toHaveBeenNthCalledWith(2, '/api/engines/mujoco/load', { method: 'POST' });
        });

        it('does not call load if probe says unavailable', async () => {
            mockFetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ name: 'pinocchio', available: false }),
            });

            const { result } = renderHook(() => useEngineManager());

            await act(async () => {
                await result.current.requestLoad('pinocchio');
            });

            // Only probe should be called, not load
            expect(mockFetch).toHaveBeenCalledTimes(1);
            expect(mockFetch).toHaveBeenCalledWith('/api/engines/pinocchio/probe');
        });
    });
});
