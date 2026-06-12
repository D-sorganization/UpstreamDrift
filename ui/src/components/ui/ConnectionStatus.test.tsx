/**
 * ConnectionStatus tests (#7435): the `lost` state must offer an actionable
 * Reconnect affordance rather than a dead-end banner.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ConnectionStatus } from './ConnectionStatus';

describe('ConnectionStatus', () => {
  it('renders the status text with an accessible label', () => {
    render(<ConnectionStatus status="connected" />);
    const region = screen.getByRole('status');
    expect(region).toHaveAttribute(
      'aria-label',
      'Connection status: Connected',
    );
    expect(screen.getByText('Connected')).toBeInTheDocument();
  });

  it('shows a Reconnect button when lost and onReconnect is provided', () => {
    const onReconnect = vi.fn();
    render(<ConnectionStatus status="lost" onReconnect={onReconnect} />);
    const btn = screen.getByRole('button', { name: 'Reconnect' });
    fireEvent.click(btn);
    expect(onReconnect).toHaveBeenCalledTimes(1);
  });

  it('omits the Reconnect button when lost but no handler is given', () => {
    render(<ConnectionStatus status="lost" />);
    expect(screen.queryByRole('button', { name: 'Reconnect' })).toBeNull();
  });

  it('does not show Reconnect for non-lost statuses even with a handler', () => {
    render(<ConnectionStatus status="connected" onReconnect={() => {}} />);
    expect(screen.queryByRole('button', { name: 'Reconnect' })).toBeNull();
  });
});
