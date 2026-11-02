import { appendFile, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const count = 3;
const version = '12.6.1';
const categories = ['performance', 'accessibility', 'best-practices', 'seo'];
const fail = (message) => { throw new Error(`Lighthouse baseline invalid: ${message}`); };

export function validateReports(reports, expectedUrl) {
  if (!Array.isArray(reports) || reports.length !== count) fail(`expected ${count} reports, found ${reports?.length ?? 0}`);
  const origin = new URL(expectedUrl).origin;
  const browser = reports[0]?.environment?.networkUserAgent;
  if (typeof browser !== 'string' || !/Chrome\/[\d.]+/.test(browser)) fail('Chrome version is missing');
  const scores = Object.fromEntries(categories.map((category) => [category, []]));
  for (const report of reports) {
    if (report.requestedUrl !== expectedUrl || report.finalUrl !== expectedUrl) fail(`expected URL ${expectedUrl}`);
    if (report.lighthouseVersion !== version) fail(`expected Lighthouse ${version}`);
    if (report.runtimeError) fail(`runtime error ${report.runtimeError.code ?? 'unknown'}`);
    if (report.environment?.networkUserAgent !== browser) fail('browser changed between runs');
    const settings = report.configSettings;
    if (settings?.formFactor !== 'mobile' || settings.screenEmulation?.mobile !== true ||
      settings.screenEmulation?.disabled !== false || settings.screenEmulation?.width !== 390 ||
      settings.screenEmulation?.height !== 844 || settings.throttlingMethod !== 'simulate') {
      fail('active 390x844 mobile emulation or simulated throttling is missing');
    }
    for (const category of categories) {
      const score = report.categories?.[category]?.score;
      if (!Number.isFinite(score) || score < 0 || score > 1) fail(`invalid ${category} score`);
      scores[category].push(score);
    }
    const requests = report.audits?.['network-requests']?.details?.items;
    if (!Array.isArray(requests)) fail('network request audit is missing');
    for (const request of requests) {
      let url;
      try { url = new URL(request.url); } catch { fail('network request URL is invalid'); }
      if (url.origin !== origin) fail(`external request ${url.origin}`);
    }
  }
  const medianScores = Object.fromEntries(categories.map((category) => [
    category, [...scores[category]].sort((a, b) => a - b)[1],
  ]));
  return {
    reportCount: count,
    sourceRevision: process.env.GITHUB_SHA ?? 'local',
    runnerOs: process.env.RUNNER_OS ?? process.platform,
    nodeVersion: process.version,
    url: expectedUrl,
    viewport: { width: 390, height: 844 },
    configSettings: {
      formFactor: 'mobile',
      screenEmulation: { mobile: true, width: 390, height: 844, disabled: false },
      throttlingMethod: 'simulate',
    },
    browserNetworkPolicy: 'loopback-only proxy',
    lighthouseVersion: version,
    chromeVersion: browser.match(/Chrome\/([\d.]+)/)?.[1],
    medianScores,
    scores,
  };
}

export async function loadReports(directory, expectedUrl) {
  const root = path.resolve(directory);
  let manifest;
  try { manifest = JSON.parse(await readFile(path.join(root, 'manifest.json'), 'utf8')); }
  catch { fail('filesystem manifest is missing or invalid'); }
  if (!Array.isArray(manifest) || manifest.length !== count) fail(`expected ${count} reports, found ${manifest?.length ?? 0}`);
  const paths = manifest.map((entry) => {
    if (entry.url !== expectedUrl || typeof entry.jsonPath !== 'string') fail('manifest URL or report path is invalid');
    const target = path.resolve(root, entry.jsonPath);
    const relative = path.relative(root, target);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative) || path.dirname(relative) !== '.') {
      fail('manifest report path is outside the artifact directory');
    }
    return target;
  });
  if (new Set(paths).size !== count) fail('manifest repeats a report path');
  const names = await readdir(root);
  const reports = names.filter((name) => name.endsWith('.json') && !['manifest.json', 'baseline-receipt.json'].includes(name));
  const expectedNames = paths.map((reportPath) => path.basename(reportPath));
  if (reports.length !== count || reports.some((name) => !expectedNames.includes(name))) {
    fail(`expected exactly ${count} manifest reports, found ${reports.length}`);
  }
  const values = await Promise.all(paths.map(async (reportPath) => {
    try { return JSON.parse(await readFile(reportPath, 'utf8')); }
    catch { fail('manifest references a missing or invalid report file'); }
  }));
  return validateReports(values, expectedUrl);
}

export async function writeReceipt(directory, expectedUrl) {
  const receipt = await loadReports(directory, expectedUrl);
  await writeFile(path.join(directory, 'baseline-receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`);
  const rows = categories.map((category) => `| ${category} | ${receipt.medianScores[category].toFixed(3)} |`);
  const summary = [
    '### Lighthouse mobile baseline', '', `- Revision: \`${receipt.sourceRevision}\``,
    `- Chrome: \`${receipt.chromeVersion}\``, `- Lighthouse: \`${version}\``,
    `- Runs: ${count} at 390×844`, '', '| Category | Median score |', '| --- | ---: |', ...rows,
    '', 'Scores are recorded without a pass threshold.', '',
  ].join('\n');
  if (process.env.GITHUB_STEP_SUMMARY) await appendFile(process.env.GITHUB_STEP_SUMMARY, summary);
  process.stdout.write(summary);
  return receipt;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  writeReceipt(process.argv[2] ?? '.lighthouseci', process.argv[3] ?? 'http://127.0.0.1:4173/')
    .catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
}
