import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createBrowserProxy } from './browser-egress-proxy.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../dist');
const host = '127.0.0.1';
const port = Number(process.env.PORT ?? 4173);
const proxyPort = 4174;
const proxy = createBrowserProxy();
const tile = (id, name, category, order, engine_type) => ({
  id,
  name,
  description: `${name} fixture`,
  category,
  type: engine_type ?? 'special_app',
  path: 'fixture',
  logo: 'fixture.svg',
  status: category === 'physics_engine' ? 'gui_ready' : 'utility',
  capabilities: ['browser_smoke'],
  order,
  default_launch: 'tab',
});
const manifest = {
  version: '1.0.0',
  description: 'Deterministic browser-test launcher manifest',
  tiles: [
    tile('model_explorer', 'Model Explorer', 'tool', 1),
    tile('mujoco_unified', 'MuJoCo', 'physics_engine', 2, 'custom_humanoid'),
    tile('drake_golf', 'Drake', 'physics_engine', 3, 'drake'),
    tile('canonical_core_estimation', 'Canonical-Core Estimation', 'biomechanics', 4),
  ],
};
const contentTypes = { '.css': 'text/css; charset=utf-8', '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml' };
const sendJson = (response, status, body) => { response.writeHead(status, { 'content-type': 'application/json; charset=utf-8' }); response.end(JSON.stringify(body)); };
const sendLogo = (response) => {
  response.writeHead(200, { 'content-type': 'image/svg+xml' });
  response.end('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="12" fill="#60a5fa"/></svg>');
}
const server = createServer((request, response) => {
  const url = new URL(request.url ?? '/', `http://${host}:${port}`);
  if (url.pathname === '/api/launcher/manifest') return sendJson(response, 200, manifest);
  if (url.pathname === '/__network-policy') return sendJson(response, 200, { requests: proxy.policyRequests() });
  if (url.pathname.startsWith('/api/launcher/logos/')) return sendLogo(response);
  if (url.pathname.startsWith('/api/')) return sendJson(response, 404, { detail: 'No fixture for this API request' });
  let decodedPath;
  try {
    decodedPath = decodeURIComponent(url.pathname);
  } catch {
    response.writeHead(400);
    response.end();
    return;
  }
  const candidate = path.resolve(root, `.${decodedPath}`);
  const safePath = path.relative(root, candidate);
  const filePath = safePath && !safePath.startsWith('..') && !path.isAbsolute(safePath)
    ? candidate
    : path.join(root, 'index.html');
  const exists = existsSync(filePath) && statSync(filePath).isFile();
  if (!exists && path.extname(url.pathname)) {
    response.writeHead(404);
    response.end();
    return;
  }
  const resolvedFile = exists ? filePath : path.join(root, 'index.html');
  response.writeHead(200, {
    'content-type': contentTypes[path.extname(resolvedFile)] ?? 'application/octet-stream',
  });
  createReadStream(resolvedFile).pipe(response);
});
server.listen(port, host, () => proxy.listen(proxyPort, host, () => process.stdout.write('fixture server ready\n')));
