import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './e2e',
  testMatch: 'mobile-baseline.spec.ts',
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: { baseURL: 'http://127.0.0.1:4173', headless: true, viewport: { width: 390, height: 844 },
    launchOptions: { args: ['--proxy-server=http://127.0.0.1:4174', '--proxy-bypass-list=<-loopback>',
      '--disable-quic', '--disable-background-networking', '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1'] } },
  webServer: { command: 'node e2e/fixture-server.mjs', url: 'http://127.0.0.1:4173/', reuseExistingServer: false, timeout: 30_000 },
});
