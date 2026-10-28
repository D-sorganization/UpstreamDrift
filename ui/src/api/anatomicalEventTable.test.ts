import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ANATOMICAL_EVENT_TABLE_PATH,
  fetchAnatomicalEventTable,
  type AnatomicalEventTableRequest,
} from './anatomicalEventTable';

const request = vi.hoisted(() => vi.fn());
vi.mock('./fetch', () => ({ apiFetch: request }));

const frames = { rotation_world: [], origin_world: [], calibration_id: 'cal-1' };
const payload: AnatomicalEventTableRequest = {
  time_s: [0, 0.01],
  events: { address: 0, impact: 1 },
  handedness: 'RH',
  event_source: 'detector-v1',
  receipt_path: 'receipts/swing.json',
  arms: { left: { humerus: frames, forearm: frames, hand: frames } },
};

beforeEach(() => {
  request.mockReset();
  request.mockResolvedValue({ schema: 'upstreamdrift.anatomical_event_table' });
});

describe('fetchAnatomicalEventTable (HRG-E4, #13175)', () => {
  it('posts to the existing biomechanics route', () => {
    expect(ANATOMICAL_EVENT_TABLE_PATH).toBe('/api/biomechanics/anatomical-event-table');
  });

  it('requests the json export format and forwards the signal', async () => {
    const controller = new AbortController();
    await fetchAnatomicalEventTable(payload, controller.signal);
    expect(request).toHaveBeenCalledExactlyOnceWith(ANATOMICAL_EVENT_TABLE_PATH, {
      method: 'POST',
      body: JSON.stringify({ ...payload, format: 'json' }),
      signal: controller.signal,
    });
  });

  it('never lets a caller request csv through the json client', async () => {
    const sneaky = { ...payload, format: 'csv' } as AnatomicalEventTableRequest;
    await fetchAnatomicalEventTable(sneaky);
    const body = JSON.parse(request.mock.calls[0][1].body as string);
    expect(body.format).toBe('json');
  });

  it('returns the document the transport resolves with', async () => {
    const doc = await fetchAnatomicalEventTable(payload);
    expect(doc).toEqual({ schema: 'upstreamdrift.anatomical_event_table' });
  });

  it('propagates a refused request', async () => {
    request.mockRejectedValue(new Error('events must be within time_s'));
    await expect(fetchAnatomicalEventTable(payload)).rejects.toThrow(/events must be/);
  });
});
