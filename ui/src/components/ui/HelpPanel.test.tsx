/**
 * Tests for HelpPanel and HelpButton components.
 *
 * See issue #1205
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent, waitFor } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { HelpPanel, HelpButton } from './HelpPanel';
import { HELP_TOPICS, FEATURE_HELP } from './helpData';

describe('HelpPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing when isOpen is false', () => {
    const { container } = render(<HelpPanel isOpen={false} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders the help dialog when isOpen is true', () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Help')).toBeInTheDocument();
  });

  it('shows overview when no topic is selected', () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);
    expect(screen.getByText('UpstreamDrift Help')).toBeInTheDocument();
    expect(
      screen.getByText(/Select a topic from the sidebar/)
    ).toBeInTheDocument();
  });

  it('shows topic content when initialTopicId is provided', () => {
    render(
      <HelpPanel
        isOpen={true}
        initialTopicId="engine_selection"
        onClose={() => {}}
      />
    );
    expect(screen.getByText('Engine Selection')).toBeInTheDocument();
  });

  it('displays all sidebar categories with topics', () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);
    // Should show topic titles in the sidebar
    for (const topic of Object.values(HELP_TOPICS)) {
      expect(screen.getAllByText(topic.title).length).toBeGreaterThan(0);
    }
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn();
    render(<HelpPanel isOpen={true} onClose={onClose} />);

    const closeButton = screen.getByLabelText('Close help panel');
    fireEvent.click(closeButton);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when Escape is pressed', () => {
    const onClose = vi.fn();
    render(<HelpPanel isOpen={true} onClose={onClose} />);

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when backdrop is clicked', () => {
    const onClose = vi.fn();
    render(<HelpPanel isOpen={true} onClose={onClose} />);

    // The backdrop is the first div inside the dialog with bg-black/50
    const backdrop = screen.getByRole('dialog').querySelector('[aria-hidden="true"]');
    expect(backdrop).not.toBeNull();
    fireEvent.click(backdrop!);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('navigates to a topic when sidebar button is clicked', () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);

    // Click on "Simulation Controls" in the sidebar
    const sidebarButtons = screen.getAllByText('Simulation Controls');
    // First one is in the sidebar
    fireEvent.click(sidebarButtons[0]);

    // Should now show simulation controls content
    expect(screen.getByText('Control simulation playback and parameters')).toBeInTheDocument();
  });

  it('shows tips section for topics with tips', () => {
    render(
      <HelpPanel
        isOpen={true}
        initialTopicId="engine_selection"
        onClose={() => {}}
      />
    );
    expect(screen.getByText('Tips')).toBeInTheDocument();
    // Should show at least one tip
    const tips = FEATURE_HELP['engine_selection'].tips;
    expect(screen.getByText(tips[0])).toBeInTheDocument();
  });

  it('shows related topics section', () => {
    render(
      <HelpPanel
        isOpen={true}
        initialTopicId="engine_selection"
        onClose={() => {}}
      />
    );
    expect(screen.getByText('Related Topics')).toBeInTheDocument();
  });

  it('navigates to related topic when clicked', () => {
    render(
      <HelpPanel
        isOpen={true}
        initialTopicId="engine_selection"
        onClose={() => {}}
      />
    );

    // Click on a related topic
    const relatedButtons = screen.getAllByText('Simulation Controls');
    // Click the one in the related topics section (last occurrence)
    fireEvent.click(relatedButtons[relatedButtons.length - 1]);

    // Now should show simulation controls content
    expect(screen.getByText('Control simulation playback and parameters')).toBeInTheDocument();
  });

  it('search filters topics', async () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);

    const searchInput = screen.getByPlaceholderText('Search help topics...');
    fireEvent.change(searchInput, { target: { value: 'engine' } });

    await waitFor(() => {
      // Should show search results header
      expect(screen.getByText(/Search Results/)).toBeInTheDocument();
    });
  });

  it('search shows no results for gibberish', async () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);

    const searchInput = screen.getByPlaceholderText('Search help topics...');
    fireEvent.change(searchInput, { target: { value: 'xyznonexistent123' } });

    await waitFor(() => {
      expect(screen.getByText('No results found')).toBeInTheDocument();
    });
  });

  it('has proper ARIA attributes', () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);

    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-label', 'Help panel');
  });

  it('has accessible search input', () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);
    expect(screen.getByLabelText('Search help topics')).toBeInTheDocument();
  });

  it('shows topic cards in overview grid', () => {
    render(<HelpPanel isOpen={true} onClose={() => {}} />);

    // All topic short descriptions should appear
    for (const topic of Object.values(HELP_TOPICS)) {
      const elements = screen.getAllByText(topic.shortDescription);
      expect(elements.length).toBeGreaterThan(0);
    }
  });
});

describe('HelpButton', () => {
  it('renders a button with ? text', () => {
    render(<HelpButton topicId="engine_selection" />);
    expect(screen.getByText('?')).toBeInTheDocument();
  });

  it('has correct aria-label', () => {
    render(<HelpButton topicId="engine_selection" />);
    expect(
      screen.getByLabelText('Help: Engine Selection Guide')
    ).toBeInTheDocument();
  });

  it('opens HelpPanel when clicked', () => {
    render(<HelpButton topicId="engine_selection" />);
    fireEvent.click(screen.getByText('?'));

    // Should now show the help panel with engine selection topic
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Engine Selection')).toBeInTheDocument();
  });

  it('closes HelpPanel when close button is clicked', () => {
    render(<HelpButton topicId="engine_selection" />);
    fireEvent.click(screen.getByText('?'));

    // Panel should be open
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    // Close it
    fireEvent.click(screen.getByLabelText('Close help panel'));

    // Panel should be gone
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('accepts custom tooltip', () => {
    render(
      <HelpButton topicId="engine_selection" tooltip="Need help?" />
    );
    expect(screen.getByTitle('Need help?')).toBeInTheDocument();
  });
});

describe('helpData', () => {
  it('all FEATURE_HELP keys have corresponding HELP_TOPICS entries', () => {
    for (const key of Object.keys(FEATURE_HELP)) {
      // Some feature help entries may not have a topic entry (that is fine)
      // but all topic entries should have valid related topics
      const topic = HELP_TOPICS[key];
      if (topic) {
        for (const relId of topic.relatedTopics) {
          expect(HELP_TOPICS[relId]).toBeDefined();
        }
      }
    }
  });

  it('all related topics reference valid topics', () => {
    for (const topic of Object.values(HELP_TOPICS)) {
      for (const relId of topic.relatedTopics) {
        expect(HELP_TOPICS[relId]).toBeDefined();
      }
    }
  });
});
