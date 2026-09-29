import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';

// Mock react-three/fiber
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
    <div data-testid="canvas-mock" {...props}>
      {children}
    </div>
  ),
  useFrame: vi.fn(),
}));

// Mock react-three/drei
vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls-mock" />,
  Grid: () => <div data-testid="grid-mock" />,
  Environment: () => <div data-testid="environment-mock" />,
}));

import { CharacterBuilderPage } from './CharacterBuilder';

describe('CharacterBuilderPage', () => {
  const mockFetch = vi.fn();
  let clickSpy: MockInstance;
  let setAttributeSpy: MockInstance;

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
    vi.stubGlobal('fetch', mockFetch);

    // Mock URL.createObjectURL and URL.revokeObjectURL
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:mock-url'),
      revokeObjectURL: vi.fn(),
    });

    clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    setAttributeSpy = vi.spyOn(HTMLAnchorElement.prototype, 'setAttribute').mockImplementation(() => {});
  });

  afterEach(() => {
    clickSpy.mockRestore();
    setAttributeSpy.mockRestore();
  });

  it('renders the Character Builder heading and core sliders', () => {
    render(<CharacterBuilderPage />);

    expect(screen.getByText('Character Builder')).toBeInTheDocument();
    expect(screen.getByLabelText(/Height/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Weight/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Build Type/i)).toBeInTheDocument();
  });

  it('has correct default slider ranges and values', () => {
    render(<CharacterBuilderPage />);

    const heightSlider = screen.getByLabelText(/Height/i) as HTMLInputElement;
    expect(heightSlider.value).toBe('1.8');
    expect(heightSlider.min).toBe('1.5');
    expect(heightSlider.max).toBe('2.1');

    const weightSlider = screen.getByLabelText(/Weight/i) as HTMLInputElement;
    expect(weightSlider.value).toBe('80');
    expect(weightSlider.min).toBe('40');
    expect(weightSlider.max).toBe('150');

    const buildSelect = screen.getByLabelText(/Build Type/i) as HTMLSelectElement;
    expect(buildSelect.value).toBe('Average');
  });

  it('updates display values and segment breakdown when inputs change', () => {
    render(<CharacterBuilderPage />);

    const heightSlider = screen.getByLabelText(/Height/i) as HTMLInputElement;
    const weightSlider = screen.getByLabelText(/Weight/i) as HTMLInputElement;
    const buildSelect = screen.getByLabelText(/Build Type/i) as HTMLSelectElement;

    // Initially, segment table has average/default proportions
    expect(screen.getByText('25.1 cm')).toBeInTheDocument(); // Head length: 1.8 * 0.1395 = 0.2511m (25.1cm)
    expect(screen.getByText('59.0 cm')).toBeInTheDocument(); // Trunk length: 1.8 * 0.328 = 0.5904m (59.0cm)

    fireEvent.change(heightSlider, { target: { value: '2.00' } });
    fireEvent.change(weightSlider, { target: { value: '100' } });
    fireEvent.change(buildSelect, { target: { value: 'Athletic' } });

    // Verify updating slider values
    expect(heightSlider.value).toBe('2');
    expect(weightSlider.value).toBe('100');
    expect(buildSelect.value).toBe('Athletic');

    // Head length for height 2.0m: 2.0 * 0.1395 = 0.279m = 27.9cm
    expect(screen.getByText('27.9 cm')).toBeInTheDocument();
  });

  it('sends POST request and triggers file download when Generate URDF is clicked', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve('<urdf></urdf>'),
    });

    render(<CharacterBuilderPage />);

    const generateBtn = screen.getByRole('button', { name: /Generate URDF/i });
    expect(generateBtn).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(generateBtn);
    });

    // Check fetch parameters
    expect(mockFetch).toHaveBeenCalledWith('/api/character-builder/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        height_m: 1.8,
        mass_kg: 80,
        build_type: 'average',
      }),
    });
  });
});
