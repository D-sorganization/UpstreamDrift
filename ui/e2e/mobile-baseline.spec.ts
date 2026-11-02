import { expect, test } from '@playwright/test';

const viewportCases = [
  { width: 390, height: 844 },
  { width: 768, height: 1024 },
];
for (const viewport of viewportCases) {
  test(`loads the Dashboard at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    const pageErrors: string[] = [];
    const externalRequests: string[] = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    page.on('request', (request) => {
      if (new URL(request.url()).origin !== 'http://127.0.0.1:4173') externalRequests.push(request.url());
    });
    await page.setViewportSize(viewport);
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Core Physics Engines' })).toBeVisible();
    await expect(page.getByRole('button', { name: /Model Explorer/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Help' })).toBeVisible();
    expect(pageErrors).toEqual([]);
    expect(externalRequests).toEqual([]);
  });
}
test('missing built assets return 404 instead of the SPA document', async ({ page }) => {
  const response = await page.goto('/__missing_baseline_asset__.js');
  expect(response?.status()).toBe(404);
});
test('Chromium denies external HTTP and records the HTTPS CONNECT attempt', async ({ page }) => {
  const policy = async () => {
    const response = await page.goto('/__network-policy');
    expect(response?.ok()).toBeTruthy();
    return JSON.parse(await page.locator('body').innerText()).requests;
  };
  await page.goto('/');
  expect((await policy()).some((request: { target: string; blocked: boolean }) =>
    request.target.startsWith('http://127.0.0.1:4173/') && !request.blocked)).toBeTruthy();
  const response = await page.goto('http://example.invalid/');
  expect(response?.status()).toBe(403);
  expect(response?.headers()['x-browser-network-policy']).toBe('blocked');
  const afterHttp = await policy();
  expect(afterHttp.some((request: { method: string; target: string; blocked: boolean }) =>
    request.method === 'GET' && request.target === 'http://example.invalid/' && request.blocked)).toBeTruthy();
  await page.goto('https://example.invalid/').catch(() => undefined);
  const afterHttps = await policy();
  expect(afterHttps.some((request: { method: string; target: string; blocked: boolean }) =>
    request.method === 'CONNECT' && request.target === 'example.invalid:443' && request.blocked)).toBeTruthy();
});
