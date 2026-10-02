import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { KeypointOverlay } from './KeypointOverlay';

describe('Source Image Landmark Overlay', () => {
  it('projects normalized observations to image pixels and labels unknown visibility', () => {
    const {container}=render(<KeypointOverlay points={{wrist:{x:0.5,y:0.25,visibility:null}}} coordinates="normalized_image_xy" width={320} height={240} showVisibility />);
    const point=container.querySelector('circle');
    expect(point).toHaveAttribute('cx','160'); expect(point).toHaveAttribute('cy','60');
    expect(screen.getByText('wrist: Visibility Unknown')).toBeInTheDocument();
  });
  it('preserves pixel-space points for the video analyzer', () => {
    const {container}=render(<KeypointOverlay points={{wrist:{x:20,y:30,visibility:null}}} coordinates="image_pixels" width={320} height={240} />);
    expect(container.querySelector('circle')).toHaveAttribute('cx','20');
  });
});
