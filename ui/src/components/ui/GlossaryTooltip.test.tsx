/**
 * Tests for GlossaryTooltip component.
 *
 * See issue #3165.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor, act } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { GlossaryTooltip } from './GlossaryTooltip';
import { useGlossaryStore } from '@/stores/useGlossaryStore';
import { useUIStore } from '@/stores/useUIStore';

function mockFetchResponse(body: object, ok = true) {
  return vi.fn().mockResolvedValue({
    ok,
    json: async () => body,
  } as unknown as Response);
}

describe('GlossaryTooltip', () => {
  beforeEach(() => {
    act(() => {
      useGlossaryStore.getState().clear();
      useUIStore.getState().resetUI();
    });
    vi.restoreAllMocks();
  });

  it('fetches and shows the description on hover', async () => {
    const mock = mockFetchResponse({
      term_id: 'drag_coefficient',
      title: 'Drag coefficient',
      short: 'Dimensionless drag parameter',
      level: 'intermediate',
    });
    vi.stubGlobal('fetch', mock);

    render(
      <GlossaryTooltip termId="drag_coefficient">drag coefficient</GlossaryTooltip>,
    );

    // Before hover, description is absent.
    expect(screen.queryByRole('tooltip')).toBeNull();

    // Hover over the inline term.
    fireEvent.mouseEnter(screen.getByText('drag coefficient'));

    await waitFor(() => {
      expect(screen.getByRole('tooltip')).toBeInTheDocument();
      expect(screen.getByText('Dimensionless drag parameter')).toBeInTheDocument();
    });

    // Confirms the client hit the expected endpoint.
    expect(mock).toHaveBeenCalledWith(
      expect.stringContaining('/glossary/drag_coefficient?level=intermediate'),
    );
  });

  it('opens the HelpPanel scrolled to the term on click', async () => {
    vi.stubGlobal('fetch', mockFetchResponse({
      term_id: 'spin_rate',
      title: 'Spin rate',
      short: 'Ball rotation speed.',
      level: 'intermediate',
    }));

    const openHelpSpy = vi.spyOn(useUIStore.getState(), 'openHelpPanel');

    render(<GlossaryTooltip termId="spin_rate">spin rate</GlossaryTooltip>);

    fireEvent.click(screen.getByText('spin rate'));

    expect(openHelpSpy).toHaveBeenCalledWith('spin_rate');
    // Also reflected in store state.
    expect(useUIStore.getState().helpOpen).toBe(true);
    expect(useUIStore.getState().helpTopicId).toBe('spin_rate');
  });
});
