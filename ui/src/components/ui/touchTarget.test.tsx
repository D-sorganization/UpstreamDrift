import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { render, screen } from '@testing-library/react';
import { Button } from './Button';
import { Input } from './Input';
import { Select } from './Select';

/**
 * Touch targets (#14281). jsdom cannot evaluate `(pointer: coarse)`, so the
 * contract is pinned in two halves: the shared primitives carry the
 * `touch-target` utility, and `index.css` defines that utility plus a
 * coarse-pointer base rule that lifts hand-rolled native controls to 44px.
 */
const INDEX_CSS = readFileSync(resolve(__dirname, '../../index.css'), 'utf-8');

describe('touch-target primitives (#14281)', () => {
  it('Button, Input and Select carry the touch-target utility', () => {
    render(
      <>
        <Button>Go</Button>
        <Input aria-label="Name" />
        <Select aria-label="Engine">
          <option>mujoco</option>
        </Select>
      </>
    );
    expect(screen.getByRole('button', { name: 'Go' })).toHaveClass('touch-target');
    expect(screen.getByLabelText('Name')).toHaveClass('touch-target');
    expect(screen.getByLabelText('Engine')).toHaveClass('touch-target');
  });
});

describe('touch-target CSS contract (#14281, #14267)', () => {
  it('defines a coarse-pointer touch-target utility at 44px', () => {
    const utility = INDEX_CSS.match(/@utility touch-target\s*\{([\s\S]*?)\n\}/);
    expect(utility?.[1]).toMatch(/@media \(pointer: coarse\)/);
    expect(utility?.[1]).toMatch(/min-height: 44px/);
    expect(utility?.[1]).toMatch(/min-width: 44px/);
    expect(utility?.[1]).toMatch(/touch-action: manipulation/);
  });

  it('lifts native buttons, selects, text inputs and checkbox labels on coarse pointers', () => {
    const base = INDEX_CSS.match(/\/\* Coarse-pointer base \(#14281[\s\S]*?\n\}\n/);
    expect(base).not.toBeNull();
    for (const selector of ['button', 'select', 'input', 'label:has(']) {
      expect(base?.[0]).toContain(selector);
    }
    expect(base?.[0]).toMatch(/min-height: 44px/);
  });

  it('adapts range slider hit targets and thumb dimensions on coarse pointers (#14267)', () => {
    const base = INDEX_CSS.match(/\/\* Coarse-pointer base \(#14281[\s\S]*?\n\}\n/);
    expect(base?.[0]).toMatch(/input\[type='range'\]/);
    expect(base?.[0]).toMatch(/touch-action: pan-x/);
    expect(base?.[0]).toMatch(/webkit-slider-thumb/);
    expect(base?.[0]).toMatch(/moz-range-thumb/);
    expect(base?.[0]).toMatch(/width: 28px/);
    expect(base?.[0]).toMatch(/height: 28px/);
  });

  it('ensures navigation links meet coarse-pointer touch targets (#14267)', () => {
    const base = INDEX_CSS.match(/\/\* Coarse-pointer base \(#14281[\s\S]*?\n\}\n/);
    expect(base?.[0]).toContain('nav a');
    expect(base?.[0]).toContain("[role='navigation'] a");
  });
});
