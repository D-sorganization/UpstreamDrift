import { render, screen, fireEvent, within } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { MemoryRouter } from 'react-router';
import { ToolShell } from './ToolShell';

describe('ToolShell Component', () => {
  it('renders breadcrumb with Workspace / Tool and handles outside-router fallback', () => {
    const { unmount } = render(
      <MemoryRouter initialEntries={['/tools/video-analyzer']}>
        <ToolShell workspaceId="capture_analyze" toolName="Video Analyzer"><div>Routed</div></ToolShell>
      </MemoryRouter>,
    );
    const breadcrumb = screen.getByRole('navigation', { name: 'Breadcrumb' });
    expect(breadcrumb).toBeInTheDocument();
    expect(within(breadcrumb).getByText('Capture & Analyze')).toBeInTheDocument();
    expect(within(breadcrumb).getByText('Video Analyzer')).toHaveAttribute('aria-current', 'page');
    unmount();

    render(<ToolShell workspaceId="optimize_train" toolName="Swing Objective Lab"><div>Unrouted</div></ToolShell>);
    const unroutedBreadcrumb = screen.getByRole('navigation', { name: 'Breadcrumb' });
    expect(within(unroutedBreadcrumb).getByText('Optimize & Train')).toBeInTheDocument();
    expect(within(unroutedBreadcrumb).getByText('Swing Objective Lab')).toBeInTheDocument();
  });

  it('renders and omits primaryActionDisabledReason appropriately', () => {
    const { rerender } = render(
      <MemoryRouter>
        <ToolShell toolName="Video Analyzer" primaryActionDisabledReason="Select video first"><div>Body</div></ToolShell>
      </MemoryRouter>,
    );
    const reason = screen.getByTestId('primary-action-disabled-reason');
    expect(reason).toHaveAttribute('role', 'status');
    expect(reason).toHaveTextContent('Select video first');

    rerender(<MemoryRouter><ToolShell toolName="Video Analyzer"><div>Body</div></ToolShell></MemoryRouter>);
    expect(screen.queryByTestId('primary-action-disabled-reason')).toBeNull();
  });

  it('renders title, description, actions, and toggles collapsible nav rail', () => {
    render(
      <MemoryRouter>
        <ToolShell
          toolName="Video Analyzer"
          title="Video Analyzer Title"
          description="High-speed camera swing analysis"
          actions={<button type="button" data-testid="custom-action-btn">Analyze</button>}
          initialNavCollapsed={false}
        >
          <div>Body</div>
        </ToolShell>
      </MemoryRouter>,
    );
    expect(screen.getByRole('heading', { name: 'Video Analyzer Title' })).toBeInTheDocument();
    expect(screen.getByText('High-speed camera swing analysis')).toBeInTheDocument();
    expect(screen.getByTestId('custom-action-btn')).toBeInTheDocument();

    const toggle = screen.getByTestId('nav-rail-toggle');
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });

  it('renders custom navRailSteps and Next Step card callbacks', () => {
    const onStepClick = vi.fn();
    const handleNext = vi.fn();
    render(
      <MemoryRouter>
        <ToolShell
          toolName="Video Analyzer"
          navRailSteps={[{ id: '1', label: 'Upload Video', onClick: onStepClick }]}
          nextStep={{ title: 'Tour Matching', description: 'Match kinematics', actionLabel: 'Go', onAction: handleNext }}
          initialNavCollapsed={false}
        >
          <div>Body</div>
        </ToolShell>
      </MemoryRouter>,
    );
    fireEvent.click(screen.getByText(/1\. Upload Video/i));
    expect(onStepClick).toHaveBeenCalledTimes(1);

    expect(screen.getByTestId('next-step-card')).toBeInTheDocument();
    expect(screen.getByText('Tour Matching')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Go/i }));
    expect(handleNext).toHaveBeenCalledTimes(1);
  });
});
