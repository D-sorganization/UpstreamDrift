/**
 * Tests for VideoForceOverlay component (FTO-29, #11314).
 *
 * Verifies:
 * 1. Renders N polylines and N polygons for N arrows.
 * 2. Halo elements precede stroke/shaft elements in DOM order (painters-model).
 * 3. Applies vector-effect="non-scaling-stroke" for resolution-independent lines.
 * 4. Sizes viewBox from width/height props.
 * 5. Handles empty/null glyph sets gracefully.
 */

import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';

import {

  VideoForceOverlay,
  type ProjectedGlyphSetPayload,
} from './VideoForceOverlay';

const samplePayload: ProjectedGlyphSetPayload = {
  time_s: 0.1,
  image_size_px: [1920, 1080],
  arrows: [
    {
      start_px: [100, 200],
      end_px: [300, 200],
      polyline_px: [
        [100, 200],
        [300, 200],
      ],
      head_poly_px: [
        [300, 200],
        [280, 190],
        [280, 210],
      ],
      rgba: [1, 0, 0, 1],
      color_hex: '#E69F00',
      kind: 'reaction',
      label: 'contact:lead_foot',
      magnitude: 150.0,
      units: 'N',
      shaft_width_px: 3,
      halo_width_px: 5,
    },
    {
      start_px: [400, 500],
      end_px: [400, 300],
      polyline_px: [
        [400, 500],
        [400, 300],
      ],
      head_poly_px: [
        [400, 300],
        [390, 320],
        [410, 320],
      ],
      rgba: [0, 1, 0, 1],
      color_hex: '#009E73',
      kind: 'contact',
      label: 'contact:trail_foot',
      magnitude: 200.0,
      units: 'N',
      shaft_width_px: 3,
      halo_width_px: 5,
    },
  ],
  torque_arcs: [],
  legend: {
    engine: 'pinocchio',
    force_reference_n: 100.0,
    torque_reference_nm: 10.0,
    kinds_present: ['reaction', 'contact'],
    unavailable_labels: [],
    source_labels: ['c3d'],
  },
  receipt: {
    drawn: 2,
    skipped_behind_camera: 0,
    skipped_out_of_frame: 0,
    unavailable_labels: [],
  },
};

describe('VideoForceOverlay', () => {
  it('renders null when glyphs is null or hidden', () => {
    const { container, rerender } = render(
      <VideoForceOverlay glyphs={null} width={1920} height={1080} />,
    );
    expect(container.firstChild).toBeNull();

    rerender(
      <VideoForceOverlay
        glyphs={samplePayload}
        width={1920}
        height={1080}
        visible={false}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders N polylines and N polygons for N arrows; halo precedes stroke', () => {
    const { container } = render(
      <VideoForceOverlay glyphs={samplePayload} width={1920} height={1080} />,
    );

    const svg = container.querySelector('svg');
    expect(svg).toBeDefined();
    expect(svg?.getAttribute('viewBox')).toBe('0 0 1920 1080');

    // For 2 arrows, each has a halo polyline and a shaft polyline -> 4 polylines
    const polylines = container.querySelectorAll('polyline');
    expect(polylines.length).toBe(4);

    // Each arrow has a halo polygon and a head polygon -> 4 polygons
    const polygons = container.querySelectorAll('polygon');
    expect(polygons.length).toBe(4);

    // Halo must precede stroke for each arrow in DOM order
    // Arrow 0: polylines[0] is halo, polylines[1] is shaft
    expect(polylines[0].getAttribute('data-role')).toBe('halo');
    expect(polylines[1].getAttribute('data-role')).toBe('shaft');
    expect(polygons[0].getAttribute('data-role')).toBe('halo');
    expect(polygons[1].getAttribute('data-role')).toBe('head');

    // Arrow 1: polylines[2] is halo, polylines[3] is shaft
    expect(polylines[2].getAttribute('data-role')).toBe('halo');
    expect(polylines[3].getAttribute('data-role')).toBe('shaft');
    expect(polygons[2].getAttribute('data-role')).toBe('halo');
    expect(polygons[3].getAttribute('data-role')).toBe('head');

    // Vector-effect must be non-scaling-stroke
    expect(polylines[1].getAttribute('vector-effect')).toBe(
      'non-scaling-stroke',
    );
    expect(polygons[1].getAttribute('vector-effect')).toBe(
      'non-scaling-stroke',
    );
  });

  it('renders legend when showLegend is true', () => {
    const { getByTestId } = render(
      <VideoForceOverlay
        glyphs={samplePayload}
        width={1920}
        height={1080}
        showLegend={true}
      />,
    );

    const legend = getByTestId('force-overlay-legend');
    expect(legend).toBeDefined();
    expect(legend.textContent).toContain('pinocchio');
    expect(legend.textContent).toContain('100.0 N');
  });
});
