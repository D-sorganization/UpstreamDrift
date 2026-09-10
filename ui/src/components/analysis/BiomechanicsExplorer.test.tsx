import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { BiomechanicsExplorer } from './BiomechanicsExplorer';
import { apiFetch } from '@/api/fetch';

vi.mock('@/api/fetch', () => ({ apiFetch: vi.fn() }));
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Line: () => null, XAxis: () => null, YAxis: () => null,
  CartesianGrid: () => null, Tooltip: () => null, Legend: () => null, ReferenceLine: () => null,
}));

describe('BiomechanicsExplorer', () => {
  it('loads backend channels and exposes provenance, missing capabilities and controls', async () => {
    const result = { times: [0, 1], channels: { x_factor: { values: [0, null], unit: 'deg', definition: 'Projected Separation', frame: 'world' } }, source: 'kinematics', unavailable: { shaft_twist_velocity: 'Missing Shaft Orientation' }, events: {} };
    vi.mocked(apiFetch).mockResolvedValue(result);
    render(<BiomechanicsExplorer />);
    fireEvent.click(screen.getByRole('button', { name: 'Load Recorded Biomechanics' }));
    await waitFor(() => expect(screen.getByLabelText('Select x_factor')).toBeInTheDocument());
    expect(screen.getByText(/Missing Shaft Orientation/)).toBeInTheDocument();
    expect(screen.getByText(/kinematics/)).toBeInTheDocument();
    expect(screen.getByLabelText('Color for x_factor')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Export CSV' })).toBeEnabled();
  });
});
