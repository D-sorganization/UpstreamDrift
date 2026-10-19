import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { useState } from 'react';

import { GripControls } from './GripControls';
import type { GripPicks } from './gripSelection';

const mockFetch = vi.fn();

function Harness({ onPicks }: { onPicks?: (p: GripPicks) => void }) {
  const [picks, setPicks] = useState<GripPicks>({});
  return (
    <GripControls
      picks={picks}
      onChange={(next) => {
        setPicks(next);
        onPicks?.(next);
      }}
    />
  );
}

describe('GripControls', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    vi.stubGlobal('fetch', mockFetch);
  });

  it('keeps the build button off until a physical pick is made', () => {
    render(<Harness />);
    const button = screen.getByRole('button', { name: 'Build Grip Revision' });
    expect(button).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/^Lead Thumb Offset/), {
      target: { value: '20' },
    });
    fireEvent.change(screen.getByLabelText(/^Visual Grip Strength/), {
      target: { value: '10' },
    });
    fireEvent.change(screen.getByLabelText(/^Lead Roll Offset/), {
      target: { value: '-5' },
    });
    fireEvent.change(screen.getByLabelText(/^Trail Roll Offset/), {
      target: { value: '8' },
    });
    expect(button).toBeDisabled(); // picture-only picks (thumb + HRG-A2) are not a revision
    fireEvent.change(screen.getByLabelText('Grip Style'), {
      target: { value: 'overlap' },
    });
    expect(button).toBeDisabled(); // unchanged default physical style is a no-op
    // HRG-GS-G1 (#13346): non-default grip_style is a physical pick and enables Build Grip Revision.
    fireEvent.change(screen.getByLabelText('Grip Style'), {
      target: { value: 'interlock' },
    });
    expect(button).toBeEnabled();
  });

  it('posts the picks and shows the new revision with a refit note', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: new Headers(),
      json: () =>
        Promise.resolve({
          spec_sha256: 'a'.repeat(64),
          physics_spec_sha256: 'b'.repeat(64),
          parent_physics_spec_sha256: 'c'.repeat(64),
          refit_required: true,
        }),
    });
    render(<Harness />);
    fireEvent.change(screen.getByLabelText('Grip Strength'), { target: { value: 'weak' } });
    fireEvent.click(screen.getByRole('button', { name: 'Build Grip Revision' }));
    await waitFor(() => expect(screen.getByTestId('grip-revision')).toHaveTextContent('Needs a refit'));
    const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/api/character-builder/grip');
    expect(JSON.parse(init.body as string)).toEqual({ preset: 'weak' });
  });

  it('shows the server message when the picks are refused', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 422,
      statusText: 'Unprocessable',
      headers: new Headers(),
      json: () => Promise.resolve({ detail: 'lead_v_azimuth_deg must be in [0.0, 90.0]' }),
    });
    render(<Harness />);
    fireEvent.change(screen.getByLabelText('Grip Strength'), { target: { value: 'weak' } });
    fireEvent.click(screen.getByRole('button', { name: 'Build Grip Revision' }));
    await waitFor(() => expect(screen.getByText(/lead_v_azimuth_deg/)).toBeInTheDocument());
  });

  it('resets a slider to the model value', () => {
    const seen: GripPicks[] = [];
    render(<Harness onPicks={(p) => seen.push(p)} />);
    fireEvent.change(screen.getByLabelText(/^Lead Palm Diagonal/), { target: { value: '55' } });
    expect(seen[seen.length - 1]?.lead_palm_diagonal).toBe(55);
    fireEvent.click(screen.getByRole('button', { name: /Reset Lead Palm Diagonal/ }));
    expect(seen[seen.length - 1]?.lead_palm_diagonal).toBeUndefined();
  });
});
