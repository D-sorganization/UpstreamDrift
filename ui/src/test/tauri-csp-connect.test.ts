/**
 * CSP must allow the ephemeral supervised backend port (#14964).
 */
import { describe, it, expect } from 'vitest';
import tauriConf from '../../src-tauri/tauri.conf.json';

describe('Tauri CSP connect-src (#14964)', () => {
  it('allows arbitrary localhost ports for supervised backend', () => {
    const csp = tauriConf.app.security.csp;
    expect(csp).toContain('http://localhost:*');
    expect(csp).toContain('ws://localhost:*');
    expect(csp).toContain('http://127.0.0.1:*');
    expect(csp).toContain('ws://127.0.0.1:*');
  });
});
