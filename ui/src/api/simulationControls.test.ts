/**
 * Tests for the simulation controls API module (issue #7452).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  fetchCameraPresets,
  applyCameraPreset,
  controlRecording,
  buildTrajectoryCsv,
  buildTrajectoryJson,
  downloadTrajectory,
  FALLBACK_CAMERA_PRESETS,
} from './simulationControls';
import type { SimulationFrame } from './client';

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Bad Request',
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

const FRAMES: SimulationFrame[] = [
  { frame: 0, time: 0.0, state: { qpos: [1, 2], qvel: [3] } },
  { frame: 1, time: 0.1, state: { qpos: [4, 5], qvel: [6] } },
];

describe('simulationControls API', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('fetchCameraPresets', () => {
    it('GETs the presets endpoint and returns the preset list', async () => {
      const presets = [
        { preset: 'side', position: [3, 0, 1.5], target: [0, 0, 1], up: [0, 0, 1] },
        { preset: 'top', position: [0, 0, 5], target: [0, 0, 0], up: [0, 1, 0] },
      ];
      fetchMock.mockResolvedValueOnce(jsonResponse({ presets }));

      const result = await fetchCameraPresets();

      expect(fetchMock).toHaveBeenCalledWith(
        '/api/simulation/camera/presets',
        expect.objectContaining({}),
      );
      expect(result.map((p) => p.preset)).toEqual(['side', 'top']);
    });

    it('throws on a malformed payload', async () => {
      fetchMock.mockResolvedValueOnce(jsonResponse({ presets: 'nope' }));
      await expect(fetchCameraPresets()).rejects.toThrow(
        /Unexpected camera presets response/,
      );
    });

    it('exports a non-empty canonical fallback list', () => {
      expect(FALLBACK_CAMERA_PRESETS).toContain('side');
      expect(FALLBACK_CAMERA_PRESETS).toContain('follow_ball');
    });
  });

  describe('applyCameraPreset', () => {
    it('POSTs the preset to /api/simulation/camera', async () => {
      fetchMock.mockResolvedValueOnce(
        jsonResponse({
          preset: 'front',
          position: [0, 3, 1.5],
          target: [0, 0, 1],
          up: [0, 0, 1],
        }),
      );

      const result = await applyCameraPreset('front');

      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe('/api/simulation/camera');
      expect(init.method).toBe('POST');
      expect(JSON.parse(init.body as string)).toEqual({ preset: 'front' });
      expect(result.preset).toBe('front');
    });

    it('rejects an empty preset before hitting the network', async () => {
      await expect(applyCameraPreset('')).rejects.toThrow(/non-empty/);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('surfaces server validation errors', async () => {
      fetchMock.mockResolvedValueOnce(
        jsonResponse({ detail: 'Unknown camera preset: diagonal' }, false, 400),
      );
      await expect(applyCameraPreset('diagonal')).rejects.toThrow(
        /Unknown camera preset/,
      );
    });
  });

  describe('controlRecording', () => {
    it('POSTs start and returns recording state', async () => {
      fetchMock.mockResolvedValueOnce(
        jsonResponse({ recording: true, frame_count: 0, status: 'Recording started' }),
      );

      const result = await controlRecording('start');

      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe('/api/simulation/recording');
      expect(JSON.parse(init.body as string)).toEqual({ action: 'start' });
      expect(result.recording).toBe(true);
    });

    it('POSTs stop and returns the saved frame count', async () => {
      fetchMock.mockResolvedValueOnce(
        jsonResponse({ recording: false, frame_count: 42, status: 'Recording stopped' }),
      );

      const result = await controlRecording('stop');

      expect(result.recording).toBe(false);
      expect(result.frame_count).toBe(42);
    });
  });
});

describe('trajectory export builders', () => {
  it('builds CSV with flattened state columns', () => {
    const csv = buildTrajectoryCsv(FRAMES);
    const lines = csv.split('\n');

    expect(lines[0]).toBe('frame,time,qpos_0,qpos_1,qvel_0');
    expect(lines[1]).toBe('0,0,1,2,3');
    expect(lines[2]).toBe('1,0.1,4,5,6');
  });

  it('handles frames with differing state keys', () => {
    const frames: SimulationFrame[] = [
      { frame: 0, time: 0, state: { qpos: [1] } },
      { frame: 1, time: 0.1, state: { qpos: [2], extra: [9] } },
    ];
    const csv = buildTrajectoryCsv(frames);
    const lines = csv.split('\n');

    expect(lines[0]).toBe('frame,time,qpos_0,extra_0');
    expect(lines[1]).toBe('0,0,1,'); // missing key → empty cell
    expect(lines[2]).toBe('1,0.1,2,9');
  });

  it('throws on an empty frame buffer (CSV and JSON)', () => {
    expect(() => buildTrajectoryCsv([])).toThrow(/empty trajectory/);
    expect(() => buildTrajectoryJson([])).toThrow(/empty trajectory/);
  });

  it('builds JSON with frame_count and frames', () => {
    const parsed = JSON.parse(buildTrajectoryJson(FRAMES));
    expect(parsed.frame_count).toBe(2);
    expect(parsed.frames).toHaveLength(2);
    expect(parsed.frames[1].state.qpos).toEqual([4, 5]);
  });

  it('downloadTrajectory triggers an anchor download', () => {
    const createObjectURL = vi.fn(() => 'blob:fake');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL,
      revokeObjectURL,
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {});

    try {
      downloadTrajectory(FRAMES, 'csv');
      expect(createObjectURL).toHaveBeenCalledTimes(1);
      expect(click).toHaveBeenCalledTimes(1);
      expect(revokeObjectURL).toHaveBeenCalledWith('blob:fake');
    } finally {
      click.mockRestore();
      vi.unstubAllGlobals();
    }
  });
});
