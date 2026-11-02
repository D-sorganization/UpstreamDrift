/**
 * Tests for OnboardingOverlay show/dismiss logic (issue #7459).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, waitFor, fireEvent } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { OnboardingOverlay } from './OnboardingOverlay';
import { onboardingPersistence } from '@/utils/onboardingStorage';
import type { OnboardingCopy } from '@/api/about';

const mockCopy: OnboardingCopy = {
  header: 'Welcome to UpstreamDrift',
  subtitle: 'Biomechanics and Robotics Platform',
  cards: [
    {
      id: 'quick_start',
      title: 'Quick Start',
      body: 'Launch your first physics model.',
      link_text: 'Read the Guide',
      link_url: 'https://example.com/guide',
    },
    {
      id: 'configurations',
      title: 'Configurations',
      body: 'Adjust themes and dependencies.',
      link_text: 'Report an Issue',
      link_url: 'https://example.com/issues',
    },
    {
      id: 'documentation',
      title: 'Documentation',
      body: 'Browse the user guide.',
      link_text: 'Open Documentation',
      link_url: 'https://example.com/docs',
    },
  ],
};

const fetchOnboardingCopy = vi.fn();
vi.mock('@/api/about', () => ({
  fetchOnboardingCopy: () => fetchOnboardingCopy(),
}));

describe('OnboardingOverlay', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    onboardingPersistence.reset();
    fetchOnboardingCopy.mockResolvedValue(mockCopy);
  });

  it('shows the onboarding cards on first run', async () => {
    render(<OnboardingOverlay />);
    await waitFor(() => {
      expect(screen.getByRole('region', { name: 'Welcome to UpstreamDrift' })).toBeInTheDocument();
    });
    expect(screen.getByText('Welcome to UpstreamDrift')).toBeInTheDocument();
    expect(screen.getByText('Quick Start')).toBeInTheDocument();
    expect(screen.getByText('Configurations')).toBeInTheDocument();
    expect(screen.getByText('Documentation')).toBeInTheDocument();
  });

  it('is non-blocking: no modal dialog, no backdrop, page stays usable (#14263)', async () => {
    const onHomeAction = vi.fn();
    render(
      <>
        <button onClick={onHomeAction}>Launch Simulation</button>
        <OnboardingOverlay />
      </>,
    );
    const panel = await screen.findByRole('region', { name: 'Welcome to UpstreamDrift' });

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(document.querySelector('[aria-modal="true"]')).toBeNull();
    expect(document.querySelector('.inset-0')).toBeNull();
    expect(panel.className).not.toMatch(/\binset-0\b/);

    fireEvent.click(screen.getByRole('button', { name: 'Launch Simulation' }));
    expect(onHomeAction).toHaveBeenCalledTimes(1);
    expect(panel).toBeInTheDocument();
  });

  it('keeps the stored dismissal key for migration (#14263)', () => {
    window.localStorage.setItem('upstreamdrift.onboarding.dismissed', 'true');
    const { container } = render(<OnboardingOverlay />);
    expect(fetchOnboardingCopy).not.toHaveBeenCalled();
    expect(container.innerHTML).toBe('');
  });

  it('does not show when already dismissed', () => {
    onboardingPersistence.dismiss();
    const { container } = render(<OnboardingOverlay />);
    expect(fetchOnboardingCopy).not.toHaveBeenCalled();
    expect(container.innerHTML).toBe('');
  });

  it('closing without the checkbox does not persist dismissal', async () => {
    render(<OnboardingOverlay />);
    await waitFor(() => {
      expect(screen.getByRole('region', { name: 'Welcome to UpstreamDrift' })).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole('button', { name: 'Get Started' }));
    expect(screen.queryByRole('region', { name: 'Welcome to UpstreamDrift' })).not.toBeInTheDocument();
    expect(onboardingPersistence.isDismissed()).toBe(false);
  });

  it('closing with "Don\'t show again" persists dismissal', async () => {
    render(<OnboardingOverlay />);
    await waitFor(() => {
      expect(screen.getByRole('region', { name: 'Welcome to UpstreamDrift' })).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: 'Get Started' }));
    expect(screen.queryByRole('region', { name: 'Welcome to UpstreamDrift' })).not.toBeInTheDocument();
    expect(onboardingPersistence.isDismissed()).toBe(true);
  });

  it('skips onboarding silently when the API is unreachable', async () => {
    fetchOnboardingCopy.mockRejectedValue(new Error('backend down'));
    const { container } = render(<OnboardingOverlay />);
    await waitFor(() => {
      expect(fetchOnboardingCopy).toHaveBeenCalled();
    });
    expect(container.innerHTML).toBe('');
  });
});

describe('onboardingPersistence adapter', () => {
  it('round-trips dismissal state', () => {
    onboardingPersistence.reset();
    expect(onboardingPersistence.isDismissed()).toBe(false);
    onboardingPersistence.dismiss();
    expect(onboardingPersistence.isDismissed()).toBe(true);
    onboardingPersistence.reset();
    expect(onboardingPersistence.isDismissed()).toBe(false);
  });
});
