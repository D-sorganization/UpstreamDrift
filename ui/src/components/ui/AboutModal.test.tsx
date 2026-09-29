/**
 * Tests for AboutModal (issue #7459).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, waitFor, fireEvent } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { AboutModal, FRONTEND_VERSION } from './AboutModal';
import type { AboutInfo } from '@/api/about';

const mockAbout: AboutInfo = {
  app_name: 'UpstreamDrift',
  app_version: '2.1.1',
  python_version: '3.12.1',
  platform: 'Linux 6.1',
  git_commit: 'abcdef1234567890',
  dependencies: { numpy: '2.1.0', mujoco: 'not installed' },
  links: {
    repository: 'https://github.com/D-sorganization/UpstreamDrift',
    report_bug: 'https://github.com/D-sorganization/UpstreamDrift/issues',
    user_guide:
      'https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/user_guide/getting_started.md',
  },
};

const fetchAboutInfo = vi.fn();
vi.mock('@/api/about', () => ({
  fetchAboutInfo: () => fetchAboutInfo(),
}));

describe('AboutModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchAboutInfo.mockResolvedValue(mockAbout);
  });

  it('renders nothing when closed', () => {
    const { container } = render(<AboutModal isOpen={false} onClose={() => {}} />);
    expect(container.innerHTML).toBe('');
    expect(fetchAboutInfo).not.toHaveBeenCalled();
  });

  it('shows backend and frontend versions when open', async () => {
    render(<AboutModal isOpen={true} onClose={() => {}} />);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    // Frontend version comes from package.json at build time
    expect(screen.getByText(FRONTEND_VERSION)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText('2.1.1')).toBeInTheDocument();
    });
    expect(screen.getByText('3.12.1')).toBeInTheDocument();
    expect(screen.getByText('abcdef123456')).toBeInTheDocument();
    // Installed deps shown; "not installed" deps hidden
    expect(screen.getByText('numpy')).toBeInTheDocument();
    expect(screen.queryByText('mujoco')).not.toBeInTheDocument();
  });

  it('shows the user guide and report bug links from the API', async () => {
    render(<AboutModal isOpen={true} onClose={() => {}} />);
    await waitFor(() => {
      expect(screen.getByText('2.1.1')).toBeInTheDocument();
    });
    const guide = screen.getByRole('link', { name: /user guide/i });
    expect(guide).toHaveAttribute('href', mockAbout.links.user_guide);
    const bug = screen.getByRole('link', { name: /report a bug/i });
    expect(bug).toHaveAttribute('href', mockAbout.links.report_bug);
  });

  it('shows an error message when the API is unreachable', async () => {
    fetchAboutInfo.mockRejectedValue(new Error('backend down'));
    render(<AboutModal isOpen={true} onClose={() => {}} />);
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('backend down');
    });
    // Frontend version still visible for bug reports
    expect(screen.getByText(FRONTEND_VERSION)).toBeInTheDocument();
  });

  it('calls onClose from the close button', async () => {
    const onClose = vi.fn();
    render(<AboutModal isOpen={true} onClose={onClose} />);
    fireEvent.click(screen.getByLabelText('Close about dialog'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
