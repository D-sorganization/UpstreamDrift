import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { BACKEND_PORT } from './backend';

describe('Vite development proxy', () => {
  const viteConfig = readFileSync(resolve(process.cwd(), 'vite.config.ts'), 'utf8');

  it('uses the canonical backend port for REST and WebSocket traffic', () => {
    expect(viteConfig).toContain(`target: 'http://localhost:${BACKEND_PORT}'`);
    expect(viteConfig).toContain(`target: 'ws://localhost:${BACKEND_PORT}'`);
    expect(viteConfig).toContain('ws: true');
  });
});
