import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { request as httpRequest } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { loadReports, validateReports, writeReceipt } from './validate-lighthouse-reports.mjs';

const url = 'http://127.0.0.1:4173/';
const rejectReport = (candidate, pattern) => assert.throws(() => validateReports([candidate, report(), report()], url), pattern);
function report(overrides = {}) {
  return {
    requestedUrl: url,
    finalUrl: url,
    lighthouseVersion: '12.6.1',
    environment: { networkUserAgent: 'Mozilla/5.0 Chrome/123.0.0.0' },
    runtimeError: null,
    configSettings: { formFactor: 'mobile', throttlingMethod: 'simulate',
      screenEmulation: { mobile: true, width: 390, height: 844, disabled: false } },
    categories: { performance: { score: 0.8 }, accessibility: { score: 0.9 }, 'best-practices': { score: 1 }, seo: { score: 0.7 } },
    audits: { 'network-requests': { details: { items: [{ url }, { url: `${url}assets/app.js` }] } } },
    ...overrides,
  };
}
test('accepts three complete mobile reports for the local fixture origin', () => {
  const result = validateReports([report(), report(), report()], url);
  assert.equal(result.reportCount, 3);
  assert.equal(result.lighthouseVersion, '12.6.1');
  assert.equal(result.configSettings.formFactor, 'mobile');
});

test('rejects incomplete, terminated, or stale run sets', () => {
  for (const reports of [[], [report(), report()], [report(), report(), report(), report()]]) {
    assert.throws(() => validateReports(reports, url), /expected 3 reports/i);
  }
});
test('rejects a report for a different URL or non-mobile collection', () => {
  rejectReport(report({ requestedUrl: 'https://example.test/' }), /expected URL/i);
  rejectReport(report({ configSettings: { formFactor: 'desktop' } }), /mobile/i);
});

test('requires active mobile emulation and blocks requests outside the fixture origin', () => {
  for (const screenEmulation of [{ mobile: false, width: 390, height: 844, disabled: false },
    { mobile: true, width: 390, height: 844, disabled: true }]) {
    const invalid = report({ configSettings: { ...report().configSettings, screenEmulation } });
    rejectReport(invalid, /mobile/i);
  }
});

test('rejects Lighthouse runtime errors and missing or invalid category scores', () => {
  rejectReport(report({ runtimeError: { code: 'NO_FCP' } }), /runtime error/i);
  rejectReport(report({ categories: { performance: { score: Number.NaN } } }), /score/i);
});
test('rejects app requests outside the deterministic fixture origin', () => {
  const external = report({ audits: { 'network-requests': { details: { items: [{ url }, { url: 'https://fonts.example.test/font.woff2' }] } } } });
  rejectReport(external, /external request/i);
});
test('egress proxy records blocked methods and targets', async (t) => {
  const { createBrowserProxy } = await import('./browser-egress-proxy.mjs');
  const proxy = createBrowserProxy();
  await new Promise((resolve) => proxy.listen(0, '127.0.0.1', resolve));
  t.after(() => proxy.close());
  const response = await new Promise((resolve, reject) => {
    const request = httpRequest({ hostname: '127.0.0.1', port: proxy.address().port,
      path: 'http://example.invalid/' }, (value) => { value.resume(); resolve(value); });
    request.on('error', reject);
    request.end();
  });
  assert.equal(response.statusCode, 403);
  assert.equal(response.headers['x-browser-network-policy'], 'blocked');
  assert.deepEqual(proxy.policyRequests(), [{ method: 'GET', target: 'http://example.invalid/', blocked: true }]);
});

test('requires a complete filesystem manifest and all referenced reports', async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'lighthouse-receipt-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await assert.rejects(loadReports(directory, url), /manifest is missing or invalid/i);
  const reportPaths = [1, 2, 3].map((i) => path.join(directory, `dashboard-run-${i}.report.json`));
  const manifest = (paths) => writeFile(path.join(directory, 'manifest.json'), JSON.stringify(paths.map((jsonPath) => ({ url, jsonPath }))));
  await Promise.all(reportPaths.slice(0, 2).map((file) => writeFile(file, JSON.stringify(report()))));
  await manifest(reportPaths.slice(0, 2));
  await assert.rejects(loadReports(directory, url), /expected 3 reports/i);
  await writeFile(reportPaths[2], JSON.stringify(report()));
  await manifest([...reportPaths.slice(0, 2), path.join(directory, 'missing.json')]);
  await assert.rejects(loadReports(directory, url), /manifest reports/i);
  await Promise.all(reportPaths.map((file) => writeFile(file, JSON.stringify(report()))));
  await manifest(reportPaths);
  assert.equal((await writeReceipt(directory, url)).reportCount, 3);
  assert.equal(JSON.parse(await readFile(path.join(directory, 'baseline-receipt.json'))).reportCount, 3);
  await writeFile(path.join(directory, 'unexpected.json'), '{}');
  await assert.rejects(loadReports(directory, url), /manifest reports/i);
});

test('ignores raw lhci collect output (lhr-*.json) that the manifest does not name', async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'lighthouse-receipt-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const names = [1, 2, 3].map((i) => `dashboard-run-${i}.report.json`);
  await Promise.all(names.map((name) => writeFile(path.join(directory, name), JSON.stringify(report()))));
  await Promise.all([1, 2, 3].map((i) => writeFile(path.join(directory, `lhr-170000000000${i}.json`), JSON.stringify(report()))));
  await writeFile(path.join(directory, 'manifest.json'), JSON.stringify(names.map((jsonPath) => ({ url, jsonPath }))));
  assert.equal((await loadReports(directory, url)).reportCount, 3);
});
