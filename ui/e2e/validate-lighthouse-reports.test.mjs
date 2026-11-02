import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { request } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { loadReport, validateReport, writeReceipt } from './validate-lighthouse-reports.mjs';

const url = 'http://127.0.0.1:4173/';
const categories = ['performance', 'accessibility', 'best-practices', 'seo'];
const report = (overrides = {}) => ({
  requestedUrl: url, finalUrl: url, lighthouseVersion: '12.6.1',
  environment: { networkUserAgent: 'Chrome/123.0.0.0' }, runtimeError: null,
  configSettings: { formFactor: 'mobile', throttlingMethod: 'simulate',
    screenEmulation: { mobile: true, width: 390, height: 844, disabled: false } },
  categories: Object.fromEntries(categories.map((name) => [name, { score: 0.8 }])),
  audits: { 'network-requests': { details: { items: [{ url }, { url: `${url}assets/app.js` }] } } },
  ...overrides,
});
const reject = (value, pattern) => assert.throws(() => validateReport(value, url), pattern);

test('writes a sanitized numeric receipt from one complete local mobile report', async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'lighthouse-receipt-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(path.join(dir, 'manifest.json'), JSON.stringify([{ url, jsonPath: 'report.json' }]));
  await writeFile(path.join(dir, 'report.json'), JSON.stringify(report()));
  const receipt = await writeReceipt(dir, url);
  assert.equal(receipt.reportCount, 1);
  assert.equal(receipt.viewport.width, 390);
  assert.deepEqual(Object.keys(receipt.scores).sort(), categories.sort());
  assert.ok(Object.values(receipt.scores).every(Number.isFinite));
  const saved = JSON.parse(await readFile(path.join(dir, 'baseline-receipt.json')));
  assert.deepEqual(saved, receipt);
  assert.equal('audits' in saved, false);
  assert.equal('requestedUrl' in saved, false);
});

test('rejects malformed, errored, non-mobile, non-numeric, and external reports', () => {
  const desktop = report({ configSettings: { formFactor: 'desktop' } });
  const cases = [
    [report({ finalUrl: 'https://example.test/' }), /URL/i],
    [report({ runtimeError: { code: 'NO_FCP' } }), /runtime error/i],
    [desktop, /mobile/i],
    [report({ categories: { performance: { score: NaN } } }), /score/i],
    [report({ audits: { 'network-requests': { details: {
      items: [{ url: 'https://cdn.example.test/font' }],
    } } } }), /external request/i],
  ];
  for (const [value, pattern] of cases) reject(value, pattern);
});

test('requires one safe manifest entry and its complete report', async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'lighthouse-input-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await assert.rejects(loadReport(dir, url), /manifest/i);
  const manifest = (entries) => writeFile(path.join(dir, 'manifest.json'), JSON.stringify(entries));
  await manifest([{ url, jsonPath: '../outside.json' }]);
  await assert.rejects(loadReport(dir, url), /outside/i);
  await manifest([{ url, jsonPath: 'missing.json' }]);
  await assert.rejects(loadReport(dir, url), /report/i);
  await writeFile(path.join(dir, 'report.json'), JSON.stringify(report()));
  await manifest([{ url, jsonPath: 'report.json' }, { url, jsonPath: 'report.json' }]);
  await assert.rejects(loadReport(dir, url), /one report/i);
});

test('proxy blocks an external HTTP target and records the attempt', async (t) => {
  const { createBrowserProxy } = await import('./browser-egress-proxy.mjs');
  const proxy = createBrowserProxy();
  await new Promise((resolve) => proxy.listen(0, '127.0.0.1', resolve));
  t.after(() => proxy.close());
  const response = await new Promise((resolve, rejectRequest) => {
    const req = request({ hostname: '127.0.0.1', port: proxy.address().port,
      path: 'http://example.invalid/' }, (res) => { res.resume(); resolve(res); });
    req.on('error', rejectRequest); req.end();
  });
  assert.equal(response.statusCode, 403);
  assert.deepEqual(proxy.policyRequests(), [
    { method: 'GET', target: 'http://example.invalid/', blocked: true },
  ]);
});
