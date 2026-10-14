/**
 * Tests for the BunkerShot3D workbench client (#9545, slice 3).
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BUNKER_WORKBENCH_PATH,
  compareBunkerDesigns,
  evaluateBunkerDesign,
  exploratoryLeader,
  rankedWinner,
} from './bunkerWorkbench';
import type { BunkerWorkbenchComparisonV1 } from './generated/types';

function response(ok: boolean, body: unknown, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Unprocessable Content',
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

function comparison(
  overrides: Partial<BunkerWorkbenchComparisonV1>,
): BunkerWorkbenchComparisonV1 {
  return {
    predictive: false,
    winner: null,
    exploratory_leader: null,
    objective_standing: { disposition: 'unavailable-uncalibrated', reasons: [], predictive: false },
    ...overrides,
  } as BunkerWorkbenchComparisonV1;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('bunker workbench requests', () => {
  it('posts the evaluation to the versioned route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response(true, { digest: 'd' }));
    vi.stubGlobal('fetch', fetchMock);

    await evaluateBunkerDesign({ handedness: 'right', design: { name: 'A' } });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${BUNKER_WORKBENCH_PATH}/evaluate`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      handedness: 'right',
      design: { name: 'A' },
    });
  });

  it('posts the comparison to the versioned route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response(true, { digest: 'd' }));
    vi.stubGlobal('fetch', fetchMock);

    await compareBunkerDesigns({
      handedness: 'right',
      left: { name: 'A' },
      right: { name: 'B' },
    });

    expect(fetchMock.mock.calls[0][0]).toContain(`${BUNKER_WORKBENCH_PATH}/compare`);
  });

  it('surfaces the message of a structured 422 refusal', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        response(
          false,
          { detail: { code: 'design_names_not_distinct', message: 'names must differ' } },
          422,
        ),
      ),
    );

    await expect(
      compareBunkerDesigns({ handedness: 'right', left: { name: 'A' }, right: { name: 'A' } }),
    ).rejects.toThrow('names must differ');
  });
});

describe('ranking guard (#9239)', () => {
  it('names a winner only on a predictive standing', () => {
    const supported = comparison({
      predictive: true,
      winner: 'A',
      objective_standing: { disposition: 'supported', reasons: [], predictive: true },
    });
    expect(rankedWinner(supported)).toBe('A');
    expect(exploratoryLeader(supported)).toBeNull();
  });

  it('never shows a winner on an unqualified objective, even if one is sent', () => {
    const unqualified = comparison({ predictive: false, winner: 'A' });
    expect(rankedWinner(unqualified)).toBeNull();
    expect(exploratoryLeader(unqualified)).toBe('A');
  });

  it('reports the exploratory leader as given', () => {
    expect(exploratoryLeader(comparison({ exploratory_leader: 'B' }))).toBe('B');
    expect(rankedWinner(comparison({ exploratory_leader: 'B' }))).toBeNull();
  });
});
