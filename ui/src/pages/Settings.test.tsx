/**
 * Tests for the Settings page (#7457): load on mount, save via PUT,
 * immediate apply (theme PUT + font scale CSS var), and localStorage cache.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import { fireEvent, screen, waitFor } from '@testing-library/dom';
import { MemoryRouter } from 'react-router';
import { ToastProvider } from '@/components/ui/Toast';
import type { WebSettings } from '@/api/settingsClient';
import { SETTINGS_CACHE_KEY } from '@/api/settingsClient';
import { useSimulationStore, DEFAULT_PARAMETERS } from '@/stores';

const SERVER_SETTINGS: WebSettings = {
  appearance: { theme_id: 'Dark', font_scale: 1.0 },
  notifications: { toast_duration_ms: 4000, verbosity: 'all' },
  simulation_defaults: { default_engine: 'mujoco', duration: 3.0, timestep: 0.002 },
};

const THEME_LIST = {
  themes: {
    Dark: { name: 'Dark', is_builtin: true, colors: {} },
    Light: { name: 'Light', is_builtin: true, colors: {} },
  },
};

const ACTIVE_THEME = {
  name: 'Light',
  is_builtin: true,
  colors: { bg: '#ffffff', text: '#000000' },
};

const apiFetchMock = vi.fn();

/** jsdom's localStorage can be unavailable; install an in-memory shim. */
function installMemoryStorage(): void {
  const values = new Map<string, string>();
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => {
        values.set(key, value);
      },
      removeItem: (key: string) => {
        values.delete(key);
      },
      clear: () => values.clear(),
    },
  });
}

vi.mock('@/api/fetch', () => ({
  apiFetch: (...args: unknown[]) => apiFetchMock(...args),
}));

// Import after mocking so the page module picks up the mock.
import { SettingsPage } from './Settings';

function mockApiRoutes() {
  apiFetchMock.mockImplementation((path: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    if (path === '/api/v1/settings' && method === 'GET') {
      return Promise.resolve(structuredClone(SERVER_SETTINGS));
    }
    if (path === '/api/v1/settings' && method === 'PUT') {
      return Promise.resolve(JSON.parse(String(init?.body)));
    }
    if (path === '/api/v1/themes/' && method === 'GET') {
      return Promise.resolve(structuredClone(THEME_LIST));
    }
    if (path === '/api/v1/themes/active' && method === 'PUT') {
      return Promise.resolve({ success: true, message: 'ok' });
    }
    if (path === '/api/v1/themes/active' && method === 'GET') {
      return Promise.resolve(structuredClone(ACTIVE_THEME));
    }
    return Promise.reject(new Error(`Unexpected call: ${method} ${path}`));
  });
}

function findCall(method: string, path: string): [string, RequestInit | undefined] | undefined {
  return apiFetchMock.mock.calls.find(
    ([p, init]) => p === path && ((init as RequestInit | undefined)?.method ?? 'GET') === method,
  ) as [string, RequestInit | undefined] | undefined;
}

const renderPage = () =>
  render(
    <MemoryRouter>
      <ToastProvider>
        <SettingsPage />
      </ToastProvider>
    </MemoryRouter>,
  );

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    installMemoryStorage();
    window.localStorage.clear();
    document.documentElement.style.cssText = '';
    document.documentElement.style.fontSize = '';
    useSimulationStore.setState({
      parameters: { ...DEFAULT_PARAMETERS },
      hasRun: false,
      parametersTouched: false,
      defaultsHydrated: false,
    });
    mockApiRoutes();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it('loads settings from the server on mount', async () => {
    renderPage();

    await waitFor(() => {
      expect(findCall('GET', '/api/v1/settings')).toBeDefined();
    });
    const duration = (await screen.findByLabelText(/duration \(s\)/i)) as HTMLInputElement;
    expect(duration.value).toBe('3');
  });

  it('populates the theme dropdown from the theme list API', async () => {
    renderPage();

    const select = (await screen.findByLabelText(/theme/i)) as HTMLSelectElement;
    await waitFor(() => {
      const options = Array.from(select.options).map((o) => o.value);
      expect(options).toContain('Dark');
      expect(options).toContain('Light');
    });
  });

  it('saves via PUT and applies theme + font scale immediately', async () => {
    renderPage();
    await screen.findByLabelText(/duration \(s\)/i);

    fireEvent.change(screen.getByLabelText(/theme/i), { target: { value: 'Light' } });
    fireEvent.change(screen.getByLabelText(/font scale/i), { target: { value: '1.5' } });
    fireEvent.click(screen.getByRole('button', { name: /save (settings|changes)/i }));

    await waitFor(() => {
      const putCall = findCall('PUT', '/api/v1/settings');
      expect(putCall).toBeDefined();
      const body = JSON.parse(String(putCall?.[1]?.body)) as WebSettings;
      expect(body.appearance.theme_id).toBe('Light');
      expect(body.appearance.font_scale).toBe(1.5);
    });

    // Theme round-trips through the theme manager and tokens are re-applied.
    await waitFor(() => {
      const themePut = findCall('PUT', '/api/v1/themes/active');
      expect(themePut).toBeDefined();
      expect(JSON.parse(String(themePut?.[1]?.body))).toEqual({ name: 'Light' });
      expect(
        document.documentElement.style.getPropertyValue('--theme-bg'),
      ).toBe('#ffffff');
    });

    // Font scale applied as a root CSS var.
    expect(
      document.documentElement.style.getPropertyValue('--app-font-scale'),
    ).toBe('1.5');

    // localStorage refreshed as a cache.
    const cached = JSON.parse(
      window.localStorage.getItem(SETTINGS_CACHE_KEY) ?? 'null',
    ) as WebSettings;
    expect(cached.appearance.font_scale).toBe(1.5);
  });

  it('hydrates simulation defaults on save without clobbering in-session edits', async () => {
    renderPage();
    await screen.findByLabelText(/duration \(s\)/i);

    // Simulate an in-session user edit before save.
    useSimulationStore.getState().setParameters({ duration: 9.9 });

    fireEvent.change(screen.getByLabelText(/duration \(s\)/i), { target: { value: '5' } });
    fireEvent.click(screen.getByRole('button', { name: /save (settings|changes)/i }));

    await waitFor(() => {
      expect(findCall('PUT', '/api/v1/settings')).toBeDefined();
    });

    // The in-session change wins (#7424 guard).
    expect(useSimulationStore.getState().parameters.duration).toBe(9.9);
  });

  // ------------------------------------------------------------------
  // #8892 — unsaved edits can no longer be discarded silently
  // ------------------------------------------------------------------

  it('disables Save while pristine and enables it once an edit is made', async () => {
    renderPage();
    await screen.findByLabelText(/duration \(s\)/i);

    const save = () =>
      screen.getByRole('button', { name: /save (settings|changes)/i }) as HTMLButtonElement;
    await waitFor(() => expect(save().disabled).toBe(true));
    expect(screen.getByTestId('settings-dirty-state').textContent).toMatch(/all changes saved/i);

    fireEvent.change(screen.getByLabelText(/font scale/i), { target: { value: '1.5' } });

    await waitFor(() => expect(save().disabled).toBe(false));
    expect(save().textContent).toMatch(/save changes/i);
    expect(screen.getByTestId('settings-dirty-state').textContent).toMatch(/unsaved changes/i);
  });

  it('goes clean again when an edit is reverted to the loaded value', async () => {
    renderPage();
    const scale = (await screen.findByLabelText(/font scale/i)) as HTMLInputElement;
    const original = scale.value;

    fireEvent.change(scale, { target: { value: '1.5' } });
    await waitFor(() =>
      expect(screen.getByTestId('settings-dirty-state').textContent).toMatch(/unsaved changes/i),
    );

    fireEvent.change(scale, { target: { value: original } });
    await waitFor(() =>
      expect(screen.getByTestId('settings-dirty-state').textContent).toMatch(/all changes saved/i),
    );
  });

  it('confirms before the back arrow discards unsaved edits', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    try {
      renderPage();
      await screen.findByLabelText(/duration \(s\)/i);

      // Pristine: leaving must not prompt.
      fireEvent.click(screen.getByLabelText(/back to dashboard/i));
      expect(confirmSpy).not.toHaveBeenCalled();

      fireEvent.change(screen.getByLabelText(/font scale/i), { target: { value: '1.5' } });
      await waitFor(() =>
        expect(screen.getByTestId('settings-dirty-state').textContent).toMatch(/unsaved changes/i),
      );

      fireEvent.click(screen.getByLabelText(/back to dashboard/i));
      expect(confirmSpy).toHaveBeenCalledTimes(1);
      // Answering "stay" keeps the edit.
      expect((screen.getByLabelText(/font scale/i) as HTMLInputElement).value).toBe('1.5');
    } finally {
      confirmSpy.mockRestore();
    }
  });

  it('registers a beforeunload guard only while dirty', async () => {
    const addSpy = vi.spyOn(window, 'addEventListener');
    const removeSpy = vi.spyOn(window, 'removeEventListener');
    try {
      renderPage();
      await screen.findByLabelText(/duration \(s\)/i);

      const beforeUnloadAdds = () =>
        addSpy.mock.calls.filter(([name]) => name === 'beforeunload').length;
      expect(beforeUnloadAdds()).toBe(0);

      fireEvent.change(screen.getByLabelText(/font scale/i), { target: { value: '1.5' } });
      await waitFor(() => expect(beforeUnloadAdds()).toBeGreaterThan(0));

      // Saving clears the flag, which unregisters the guard.
      fireEvent.click(screen.getByRole('button', { name: /save (settings|changes)/i }));
      await waitFor(() =>
        expect(
          removeSpy.mock.calls.filter(([name]) => name === 'beforeunload').length,
        ).toBeGreaterThan(0),
      );
    } finally {
      addSpy.mockRestore();
      removeSpy.mockRestore();
    }
  });

  it('is clean again after a successful save', async () => {
    renderPage();
    await screen.findByLabelText(/duration \(s\)/i);

    fireEvent.change(screen.getByLabelText(/font scale/i), { target: { value: '1.5' } });
    await waitFor(() =>
      expect(screen.getByTestId('settings-dirty-state').textContent).toMatch(/unsaved changes/i),
    );

    fireEvent.click(screen.getByRole('button', { name: /save (settings|changes)/i }));

    await waitFor(() =>
      expect(screen.getByTestId('settings-dirty-state').textContent).toMatch(/all changes saved/i),
    );
  });

  it('shows an error banner when settings cannot be loaded', async () => {
    apiFetchMock.mockImplementation((path: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      if (path === '/api/v1/settings' && method === 'GET') {
        return Promise.reject(new Error('offline'));
      }
      if (path === '/api/v1/themes/' && method === 'GET') {
        return Promise.resolve(structuredClone(THEME_LIST));
      }
      return Promise.reject(new Error(`Unexpected call: ${method} ${path}`));
    });

    renderPage();

    expect(
      await screen.findByText(/could not load settings/i),
    ).toBeInTheDocument();
  });

  it('surfaces a save failure as an error toast', async () => {
    renderPage();
    await screen.findByLabelText(/duration \(s\)/i);

    // Save is disabled while pristine since #8892, so dirty the form first.
    fireEvent.change(screen.getByLabelText(/font scale/i), { target: { value: '1.25' } });
    await waitFor(() =>
      expect(screen.getByTestId('settings-dirty-state').textContent).toMatch(
        /unsaved changes/i,
      ),
    );

    apiFetchMock.mockImplementation((path: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      if (path === '/api/v1/settings' && method === 'PUT') {
        return Promise.reject(new Error('500 boom'));
      }
      return Promise.resolve({});
    });

    fireEvent.click(screen.getByRole('button', { name: /save (settings|changes)/i }));

    expect(await screen.findByText(/500 boom/i)).toBeInTheDocument();
  });
});

describe('useSimulationStore.hydrateDefaults', () => {
  beforeEach(() => {
    useSimulationStore.setState({
      parameters: { ...DEFAULT_PARAMETERS },
      hasRun: false,
      parametersTouched: false,
      defaultsHydrated: false,
    });
  });

  it('applies defaults on a fresh session', () => {
    useSimulationStore.getState().hydrateDefaults({ duration: 7, timestep: 0.01 });
    const state = useSimulationStore.getState();
    expect(state.parameters.duration).toBe(7);
    expect(state.parameters.timestep).toBe(0.01);
    expect(state.defaultsHydrated).toBe(true);
  });

  it('does not clobber user-edited parameters', () => {
    useSimulationStore.getState().setParameters({ duration: 12 });
    useSimulationStore.getState().hydrateDefaults({ duration: 7, timestep: 0.01 });
    expect(useSimulationStore.getState().parameters.duration).toBe(12);
  });

  it('only hydrates once per session', () => {
    useSimulationStore.getState().hydrateDefaults({ duration: 7 });
    useSimulationStore.setState((s) => ({
      ...s,
      parametersTouched: false,
    }));
    useSimulationStore.getState().hydrateDefaults({ duration: 8 });
    expect(useSimulationStore.getState().parameters.duration).toBe(7);
  });
});
