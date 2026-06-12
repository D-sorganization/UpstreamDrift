import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { Button } from './Button';
import { Input } from './Input';
import { Select } from './Select';
import { Badge } from './Badge';
import { Card } from './Card';

describe('Button (#7420)', () => {
  it('renders primary blue variant by default', () => {
    render(<Button>Go</Button>);
    const btn = screen.getByRole('button', { name: 'Go' });
    expect(btn.className).toContain('bg-blue-600');
    expect(btn).toHaveAttribute('type', 'button');
  });

  it('applies variant and size classes', () => {
    render(
      <Button variant="danger" size="lg">
        Delete
      </Button>
    );
    const btn = screen.getByRole('button', { name: 'Delete' });
    expect(btn.className).toContain('bg-red-600');
    expect(btn.className).toContain('px-4 py-2');
  });

  it('disables and applies disabled styling', () => {
    render(<Button disabled>Off</Button>);
    const btn = screen.getByRole('button', { name: 'Off' });
    expect(btn).toBeDisabled();
    expect(btn.className).toContain('disabled:cursor-not-allowed');
  });

  it('merges a custom className', () => {
    render(<Button className="w-full">Wide</Button>);
    expect(screen.getByRole('button', { name: 'Wide' }).className).toContain('w-full');
  });
});

describe('Input/Select (#7420)', () => {
  it('Input renders dense size variant', () => {
    render(<Input inputSize="sm" placeholder="name" />);
    const el = screen.getByPlaceholderText('name');
    expect(el.className).toContain('px-2 py-1');
    expect(el.className).toContain('focus-visible:ring-blue-400');
  });

  it('Select renders options and focus ring', () => {
    render(
      <Select aria-label="pick">
        <option value="a">A</option>
      </Select>
    );
    const el = screen.getByLabelText('pick');
    expect(el.className).toContain('focus-visible:ring-blue-400');
    expect(screen.getByRole('option', { name: 'A' })).toBeInTheDocument();
  });
});

describe('Badge/Card (#7420)', () => {
  it('Badge uses amber for warning tone (#7421 semantic)', () => {
    render(<Badge tone="warning">Heads up</Badge>);
    expect(screen.getByText('Heads up').className).toContain('bg-amber-900');
  });

  it('Card is dense when requested', () => {
    render(<Card dense>body</Card>);
    expect(screen.getByText('body').className).toContain('p-2');
  });
});
