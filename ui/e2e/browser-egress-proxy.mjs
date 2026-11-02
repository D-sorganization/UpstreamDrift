import { createServer, request as forwardRequest } from 'node:http';

export function createBrowserProxy(allowedOrigin = 'http://127.0.0.1:4173') {
  const requests = [];
  const block = (response) => { response.writeHead(403, { 'x-browser-network-policy': 'blocked' }); response.end(); };
  const proxy = createServer((request, response) => {
    let target;
    try { target = new URL(request.url ?? ''); } catch { requests.push({ method: request.method, target: request.url, blocked: true }); block(response); return; }
    const blocked = target.origin !== allowedOrigin || target.username || target.password;
    requests.push({ method: request.method, target: request.url, blocked });
    if (blocked) { block(response); return; }
    const upstream = forwardRequest(target, { method: request.method, headers: request.headers }, (result) => {
      response.writeHead(result.statusCode ?? 502, result.headers);
      result.pipe(response);
    });
    upstream.on('error', () => { response.writeHead(502); response.end(); });
    request.pipe(upstream);
  });
  proxy.on('connect', (request, socket) => { requests.push({ method: 'CONNECT', target: request.url, blocked: true }); socket.end('HTTP/1.1 403 Forbidden\r\nX-Browser-Network-Policy: blocked\r\nConnection: close\r\n\r\n'); });
  proxy.policyRequests = () => requests;
  return proxy;
}
