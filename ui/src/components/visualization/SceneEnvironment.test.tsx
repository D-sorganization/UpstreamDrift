import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

const { hdrLoader } = vi.hoisted(() => ({
  hdrLoader: { load: vi.fn() },
}));

vi.mock('@react-three/drei', () => ({
  Environment: ({ files }: { files: string }) => {
    hdrLoader.load(files);
    return null;
  },
}));

import { SceneEnvironment } from './SceneEnvironment';

describe('SceneEnvironment', () => {
  beforeEach(() => {
    hdrLoader.load.mockReset();
    hdrLoader.load.mockImplementation(() => {
      throw new Error('HDR load rejected');
    });
  });

  it('keeps the scene lit when the local HDR loader rejects', () => {
    const { container } = render(<SceneEnvironment />);

    expect(hdrLoader.load).toHaveBeenCalledWith('/hdr/studio_small_03_1k.hdr');
    expect(container.querySelector('[name="hdr-fallback-ambient"]')).toBeInTheDocument();
    expect(container.querySelector('[name="hdr-fallback-key"]')).toBeInTheDocument();
    expect(screen.queryByTestId('error-boundary-fallback')).not.toBeInTheDocument();
  });
});