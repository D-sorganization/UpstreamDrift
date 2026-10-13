import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { OutputStatusBadges } from './OutputStatusBadges';
import { OUTPUT_POLICY_SCHEMA_VERSION, parseOutputPolicy } from './outputPolicy';

const FACE = {
  observable: false,
  reason: 'club markers collinear',
  observed_frames: 40,
  total_frames: 100,
  coverage: 0.4,
  gaps: [[10, 30]],
};

const PAYLOAD = {
  schema_version: OUTPUT_POLICY_SCHEMA_VERSION,
  outputs: [
    { name: 'clubhead_speed', status: 'advertised', label: 'advertised', reason: 'advertised with uncertainty' },
    { name: 'face_angle', status: 'withheld', label: 'NOT VALIDATED', reason: 'not validated' },
    { name: 'joint_loads', status: 'unavailable', label: 'UNAVAILABLE', reason: 'no loads in receipt' },
  ],
  face_observability: FACE,
};

const RECEIPT = {
  outputs: {
    clubhead_speed: { value: 42.5, uncertainty: 1.5, unit: 'm/s' },
    // A leaked value on a withheld output must never be rendered.
    face_angle: { value: 7.77, uncertainty: 0.1, unit: 'deg' },
  },
};

afterEach(cleanup);

describe('OutputStatusBadges (#10382 web viewer)', () => {
  it('shows an advertised badge with value and uncertainty', () => {
    render(<OutputStatusBadges payload={PAYLOAD} receipt={RECEIPT} />);
    expect(screen.getByTestId('output-status-clubhead_speed')).toHaveTextContent('ADVERTISED');
    expect(screen.getByTestId('output-value-clubhead_speed')).toHaveTextContent('42.5 ± 1.5 m/s');
  });

  it('shows a withheld badge with its reason and no number', () => {
    render(<OutputStatusBadges payload={PAYLOAD} receipt={RECEIPT} />);
    const row = screen.getByTestId('output-row-face_angle');
    expect(row).toHaveTextContent('NOT VALIDATED');
    expect(row).toHaveTextContent('not validated');
    expect(row.textContent).not.toMatch(/\d/);
    expect(screen.queryByTestId('output-value-face_angle')).toBeNull();
    expect(screen.queryByText(/7\.77/)).toBeNull();
  });

  it('shows an unavailable badge without a number', () => {
    render(<OutputStatusBadges payload={PAYLOAD} receipt={RECEIPT} />);
    const row = screen.getByTestId('output-row-joint_loads');
    expect(row).toHaveTextContent('UNAVAILABLE');
    expect(row).toHaveTextContent('no loads in receipt');
    expect(screen.queryByTestId('output-value-joint_loads')).toBeNull();
  });

  it('shows the face observability summary exactly once', () => {
    render(<OutputStatusBadges payload={PAYLOAD} receipt={RECEIPT} />);
    const summaries = screen.getAllByText(/Face-angle observability/);
    expect(summaries).toHaveLength(1);
    expect(summaries[0]).toHaveTextContent('unobservable (club markers collinear)');
    expect(summaries[0]).toHaveTextContent('40/100 frames (40.0%)');
    expect(summaries[0]).toHaveTextContent('[10, 30)');
  });

  it('renders a policy-error payload as all unavailable', () => {
    const failed = {
      ...PAYLOAD,
      outputs: ['clubhead_speed', 'face_angle'].map((name) => ({
        name,
        status: 'unavailable',
        label: 'UNAVAILABLE',
        reason: 'policy error: bad receipt',
      })),
      face_observability: null,
    };
    render(<OutputStatusBadges payload={failed} receipt={RECEIPT} />);
    expect(screen.getAllByText('UNAVAILABLE')).toHaveLength(2);
    expect(screen.queryByTestId('output-value-clubhead_speed')).toBeNull();
    expect(screen.getByText(/not audited/)).toBeInTheDocument();
  });

  it('fails closed to unavailable on an unknown schema_version', () => {
    render(<OutputStatusBadges payload={{ ...PAYLOAD, schema_version: 'output-policy/2' }} receipt={RECEIPT} />);
    expect(screen.getByRole('alert')).toHaveTextContent('output-policy/2');
    expect(screen.queryByTestId('output-value-clubhead_speed')).toBeNull();
    expect(screen.getAllByText('UNAVAILABLE').length).toBeGreaterThan(0);
  });

  it('fails closed when the payload is absent', () => {
    render(<OutputStatusBadges payload={undefined} receipt={RECEIPT} />);
    expect(screen.getAllByText('UNAVAILABLE').length).toBeGreaterThan(0);
  });
});

describe('parseOutputPolicy (DbC)', () => {
  it('rejects an unknown schema_version', () => {
    expect(() => parseOutputPolicy({ ...PAYLOAD, schema_version: 'output-policy/9' })).toThrow(
      /output-policy\/9/,
    );
  });

  it('rejects a non-object payload and malformed rows', () => {
    expect(() => parseOutputPolicy(null)).toThrow();
    expect(() => parseOutputPolicy({ schema_version: OUTPUT_POLICY_SCHEMA_VERSION, outputs: [{}] })).toThrow();
  });

  it('rejects a malformed face_observability instead of crashing the render', () => {
    expect(() => parseOutputPolicy({ ...PAYLOAD, face_observability: { observable: true } })).toThrow(
      /face_observability/,
    );
    expect(() =>
      parseOutputPolicy({ ...PAYLOAD, face_observability: { ...FACE, gaps: [[1]] } }),
    ).toThrow(/face_observability/);
    expect(parseOutputPolicy({ ...PAYLOAD, face_observability: null }).face_observability).toBeNull();
  });

  it('accepts the current schema', () => {
    expect(parseOutputPolicy(PAYLOAD).outputs).toHaveLength(3);
  });
});
