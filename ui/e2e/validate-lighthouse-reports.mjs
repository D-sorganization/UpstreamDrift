import { appendFile, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const version = '12.6.1';
const categories = ['performance', 'accessibility', 'best-practices', 'seo'];
const fail = (message) => { throw new Error(`Lighthouse baseline invalid: ${message}`); };

export function validateReport(report, expectedUrl) {
  if (!report || report.requestedUrl !== expectedUrl || report.finalUrl !== expectedUrl) fail('unexpected report URL');
  if (report.lighthouseVersion !== version) fail(`expected Lighthouse ${version}`);
  if (report.runtimeError) fail(`runtime error ${report.runtimeError.code ?? 'unknown'}`);
  const browser = report.environment?.networkUserAgent?.match(/Chrome\/([\d.]+)/)?.[1];
  if (!browser) fail('Chrome version is missing');
  const settings = report.configSettings;
  if (settings?.formFactor !== 'mobile' || settings.throttlingMethod !== 'simulate' ||
      settings.screenEmulation?.mobile !== true || settings.screenEmulation.disabled !== false ||
      settings.screenEmulation.width !== 390 || settings.screenEmulation.height !== 844) {
    fail('active 390x844 mobile emulation is missing');
  }
  const scores = Object.fromEntries(categories.map((name) => {
    const score = report.categories?.[name]?.score;
    if (!Number.isFinite(score) || score < 0 || score > 1) fail(`invalid ${name} score`);
    return [name, score];
  }));
  const origin = new URL(expectedUrl).origin;
  const requests = report.audits?.['network-requests']?.details?.items;
  if (!Array.isArray(requests)) fail('network request audit is missing');
  for (const item of requests) {
    let target;
    try { target = new URL(item.url); } catch { fail('invalid network request URL'); }
    if (target.origin !== origin) fail(`external request ${target.origin}`);
  }
  return {
    sourceRevision: process.env.GITHUB_SHA ?? 'local',
    runnerOs: process.env.RUNNER_OS ?? process.platform,
    nodeVersion: process.version,
    lighthouseVersion: version,
    chromeVersion: browser,
    reportCount: 1,
    url: expectedUrl,
    viewport: { width: 390, height: 844 },
    scores,
  };
}

export async function loadReport(directory, expectedUrl) {
  const root = path.resolve(directory);
  let manifest;
  try { manifest = JSON.parse(await readFile(path.join(root, 'manifest.json'), 'utf8')); }
  catch { fail('filesystem manifest is missing or invalid'); }
  if (!Array.isArray(manifest) || manifest.length !== 1 || manifest[0].url !== expectedUrl ||
      typeof manifest[0].jsonPath !== 'string') fail('expected one report for the fixture URL');
  const file = path.resolve(root, manifest[0].jsonPath);
  const relative = path.relative(root, file);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative) || path.dirname(relative) !== '.') {
    fail('manifest report path is outside the artifact directory');
  }
  const jsonFiles = (await readdir(root)).filter((name) => name.endsWith('.json') && name !== 'manifest.json');
  if (jsonFiles.length !== 1 || jsonFiles[0] !== path.basename(file)) fail('expected one manifest report file');
  let report;
  try { report = JSON.parse(await readFile(file, 'utf8')); }
  catch { fail('manifest report is missing or invalid'); }
  return validateReport(report, expectedUrl);
}

export async function writeReceipt(directory, expectedUrl) {
  const receipt = await loadReport(directory, expectedUrl);
  await writeFile(path.join(directory, 'baseline-receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`);
  const summary = `### Lighthouse mobile baseline\n\n- Revision: \`${receipt.sourceRevision}\`\n- Chrome: \`${receipt.chromeVersion}\`\n- Lighthouse: \`${version}\`\n- Run: 390×844 (scores are informational)\n\n${categories.map((name) => `- ${name}: ${receipt.scores[name].toFixed(3)}`).join('\n')}\n`;
  if (process.env.GITHUB_STEP_SUMMARY) await appendFile(process.env.GITHUB_STEP_SUMMARY, summary);
  process.stdout.write(summary);
  return receipt;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  writeReceipt(process.argv[2] ?? '.lighthouseci', process.argv[3] ?? 'http://127.0.0.1:4173/')
    .catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
}
