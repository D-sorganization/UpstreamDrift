import { describe, expect, it } from 'vitest';
import { buildSensitivityOptions, parseChannels, type VariabilityChoices } from './variabilityRequest';
import type { SensitivityCatalogue } from '@/api/loopNullSpace';

const CAT: SensitivityCatalogue = {
  stages: [{ id: 'torque', label: 'TORQUE' }],
  spectra: [{ id: 'white', label: 'WHITE' }],
  mechanisms: [{ id: 'H-NOISE-SIGNAL-DEPENDENT', label: 'M1', note: '' }],
  strategies: [
    { id: 'S1', name: 'free', mechanism: 'm', availability: 'implemented', reason: '' },
    { id: 'S2', name: 'imp', mechanism: 'm', availability: 'implemented', reason: '' },
  ],
  outcomes: [{ id: 'face_angle', label: 'face angle' }],
  default_outcome_priority: ['face_angle'],
  engines: [{ id: 'pinocchio', method: 'analytic', available: true, reason: '' }],
};

const choices = (over: Partial<VariabilityChoices> = {}): VariabilityChoices => ({
  disturbances: [
    {
      name: 'torque_add',
      stage: 'torque',
      spectrum: 'white',
      persistent: false,
      channels: parseChannels('shoulder=0.004, elbow=0.009'),
      units: 'N m',
      bandwidthHz: null,
    },
  ],
  mechanisms: [],
  strategies: null,
  outcomePriority: ['face_angle'],
  ...over,
});

describe('parseChannels', () => {
  it('reads name=variance pairs', () => {
    expect(parseChannels('shoulder=0.004, elbow = 0.009')).toEqual([
      ['shoulder', 0.004],
      ['elbow', 0.009],
    ]);
  });
  it.each(['', 'shoulder', 'a=x', 'a=1,a=2', 'a=-1', 'a=NaN'])('refuses %j', (text) => {
    expect(() => parseChannels(text)).toThrow();
  });
});

describe('buildSensitivityOptions', () => {
  it('places the amplitudes on a diagonal covariance and keeps catalogue ids', () => {
    const out = buildSensitivityOptions(choices(), CAT);
    expect(out.disturbance_spec[0].covariance).toEqual([
      [0.004, 0],
      [0, 0.009],
    ]);
    expect(out.disturbance_spec[0].spectrum).toBe('white');
    expect(out.strategy_set).toBeNull();
    expect(out.outcome_priority).toEqual(['face_angle']);
  });
  it('a persistent disturbance has no spectrum', () => {
    const base = choices().disturbances[0];
    const out = buildSensitivityOptions(
      choices({ disturbances: [{ ...base, persistent: true }] }),
      CAT,
    );
    expect(out.disturbance_spec[0].spectrum).toBeNull();
  });
  it.each([
    choices({ mechanisms: ['H-INVENTED'] }),
    choices({ strategies: ['S99'] }),
    choices({ outcomePriority: ['elbow_angle'] }),
    choices({ outcomePriority: [] }),
    choices({ disturbances: [] }),
  ])('refuses a value outside the catalogue', (bad) => {
    expect(() => buildSensitivityOptions(bad, CAT)).toThrow();
  });
});
