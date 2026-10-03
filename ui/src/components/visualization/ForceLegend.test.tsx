/**
 * Tests for ForceLegend component (ADR-0052, #11308).
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ForceLegend } from './ForceLegend';
import type { GlyphSetV1 } from '@/types/glyphs';

describe('ForceLegend', () => {
  it('returns null when glyphs or legend is absent', () => {
    const { container: c1 } = render(<ForceLegend glyphs={null} />);
    expect(c1).toBeEmptyDOMElement();

    const { container: c2 } = render(<ForceLegend glyphs={undefined} />);
    expect(c2).toBeEmptyDOMElement();
  });

  it('renders reference values, swatches, unavailable list and engine', () => {
    const glyphs: GlyphSetV1 = {
      schema_version: 'glyph-set-v1',
      time_s: 0.5,
      arrows: [],
      torque_arcs: [],
      legend: {
        force_reference_n: 500.0,
        force_reference_length_m: 0.5,
        torque_reference_nm: 25.0,
        torque_reference_radius_m: 0.08,
        kinds_present: ['joint_actuator', 'contact'],
        unavailable_labels: ['contact:foot_left'],
        engine: 'mujoco',
        source_labels: ['mujoco:actuator', 'mujoco:contact'],
      },
    };

    render(<ForceLegend glyphs={glyphs} />);

    expect(screen.getByText(/Force & Torque Legend/i)).toBeInTheDocument();
    expect(screen.getAllByText(/mujoco/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/500 N/i)).toBeInTheDocument();
    expect(screen.getByText(/25 N\*m/i)).toBeInTheDocument();
    expect(screen.getByText(/Actuator Torque/i)).toBeInTheDocument();
    expect(screen.getByText(/Contact Force/i)).toBeInTheDocument();
    expect(screen.getByText(/contact:foot_left/i)).toBeInTheDocument();
  });

  it('handles null reference values gracefully', () => {
    const glyphs: GlyphSetV1 = {
      schema_version: 'glyph-set-v1',
      time_s: 0.5,
      arrows: [],
      torque_arcs: [],
      legend: {
        force_reference_n: null,
        force_reference_length_m: null,
        torque_reference_nm: null,
        torque_reference_radius_m: null,
        kinds_present: ['gravity'],
        unavailable_labels: [],
        engine: 'pinocchio',
        source_labels: [],
      },
    };

    render(<ForceLegend glyphs={glyphs} />);
    expect(screen.getByText(/pinocchio/i)).toBeInTheDocument();
    expect(screen.getByText(/Gravity/i)).toBeInTheDocument();
    expect(screen.queryByText(/N\*m/i)).not.toBeInTheDocument();
  });
});
