import { createServer as createHttpServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createSimulation, reset, step, observe } from './dist/simulation.js';
import { renderFrame } from './dist/renderer.js';

const projectDirectory = path.dirname(fileURLToPath(import.meta.url));
const staticDirectory = path.join(projectDirectory, 'dist');
const resolutions = new Set(['160x90', '320x180', '640x360']);
const loopbackHosts = new Set(['127.0.0.1', 'localhost', '[::1]']);
const bodyLimit = 1024 * 1024;

function requestError(message, status = 400) {
  return Object.assign(new Error(message), { status });
}

function json(response, status, payload) {
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  response.end(JSON.stringify(payload));
}

function object(value, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw requestError(`${name} must be an object.`);
  }
  return value;
}

function readJson(request) {
  const contentType = (request.headers['content-type'] || '').split(';')[0].trim();
  if (contentType !== 'application/json') {
    return Promise.reject(requestError('Use Content-Type: application/json.', 415));
  }
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let exceeded = false;
    request.on('data', chunk => {
      if (exceeded) return;
      size += chunk.length;
      if (size > bodyLimit) {
        exceeded = true;
        chunks.length = 0;
        reject(requestError('Request body exceeds 1 MiB.', 413));
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => {
      if (exceeded) return;
      try {
        resolve(object(JSON.parse(Buffer.concat(chunks).toString('utf8')), 'Request body'));
      } catch (error) {
        reject(error.status ? error : requestError('Request body is not valid JSON.'));
      }
    });
    request.on('error', reject);
    request.on('aborted', () => reject(requestError('Request was aborted.')));
  });
}

function validateOrigin(request, port) {
  let host;
  try {
    host = new URL(`http://${request.headers.host || ''}`);
  } catch {
    throw requestError('A local Host header is required.', 403);
  }
  if (!loopbackHosts.has(host.hostname) || Number(host.port || 80) !== port) {
    throw requestError('This server accepts local hostnames only.', 403);
  }
  if (request.headers.origin) {
    let origin;
    try { origin = new URL(request.headers.origin); } catch {
      throw requestError('Untrusted request origin.', 403);
    }
    if (origin.protocol !== 'http:' || origin.host !== host.host) {
      throw requestError('Cross-origin requests are not permitted.', 403);
    }
  }
}

function validateResolution(value) {
  if (!Array.isArray(value) || value.length !== 2 || !resolutions.has(value.join('x')) ||
      !value.every(Number.isInteger)) {
    throw requestError('resolution must be [160,90], [320,180], or [640,360].');
  }
  return [...value];
}

function validateReset(body) {
  if (body.seed !== undefined && (!Number.isInteger(body.seed) || body.seed < 0 || body.seed > 4294967295)) {
    throw requestError('seed must be an integer from 0 to 4294967295.');
  }
  if (body.scenario !== undefined && body.scenario !== 'tdm') {
    throw requestError('The only scenario is tdm.');
  }
  if (body.teamSize !== undefined && (!Number.isInteger(body.teamSize) || body.teamSize < 1 || body.teamSize > 8)) {
    throw requestError('teamSize must be an integer from 1 to 8.');
  }
  if (body.scoreLimit !== undefined && (!Number.isInteger(body.scoreLimit) || body.scoreLimit < 1 || body.scoreLimit > 100)) {
    throw requestError('scoreLimit must be an integer from 1 to 100.');
  }
  if (body.resolution !== undefined) validateResolution(body.resolution);
}

function validateAction(body) {
  const action = body.action === undefined ? {} : object(body.action, 'action');
  const supported = new Set(['forward', 'strafe', 'turn', 'pitch', 'fire', 'reload']);
  for (const [key, value] of Object.entries(action)) {
    if (!supported.has(key)) throw requestError(`Unknown action field: ${key}`);
    if (key === 'fire' || key === 'reload') {
      if (typeof value !== 'boolean') throw requestError(`${key} must be boolean.`);
    } else if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw requestError(`${key} must be a finite number.`);
    }
    if ((key === 'forward' || key === 'strafe') && Math.abs(value) > 1) {
      throw requestError(`${key} must be between -1 and 1.`);
    }
  }
  const ticks = body.ticks === undefined ? 1 : body.ticks;
  if (!Number.isInteger(ticks) || ticks < 1 || ticks > 600) {
    throw requestError('ticks must be an integer from 1 to 600.');
  }
  if (body.includePixels !== undefined && typeof body.includePixels !== 'boolean') {
    throw requestError('includePixels must be boolean.');
  }
  return { action, ticks };
}

/** Each server has one independent, headless environment shared by its HTTP clients. */
export function createServer() {
  const simulation = createSimulation({ seed: 42, scenario: 'tdm', teamSize: 4, scoreLimit: 25 });
  let resolution = [320, 180];

  function frame() {
    const [width, height] = resolution;
    const rgba = renderFrame(simulation, { width, height });
    const rgb = Buffer.allocUnsafe(width * height * 3);
    for (let pixel = 0; pixel < width * height; pixel++) {
      rgb[pixel * 3] = rgba[pixel * 4];
      rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
      rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
    }
    return { width, height, format: 'rgb8', data: rgb.toString('base64') };
  }

  function packet(transition = {}, includePixels = false) {
    const observation = observe(simulation);
    const output = {
      observation,
      reward: transition.reward ?? 0,
      terminated: observation.terminated,
      truncated: observation.truncated,
      info: { events: transition.events || [] },
    };
    if (includePixels) output.pixels = frame();
    return output;
  }

  const server = createHttpServer(async (request, response) => {
    try {
      validateOrigin(request, server.address().port);
      const rawPath = (request.url || '/').split('?')[0];
      let decodedPath;
      try { decodedPath = decodeURIComponent(rawPath); } catch {
        throw requestError('Invalid URL encoding.');
      }
      if (decodedPath.includes('\\') || decodedPath.includes('\0') || decodedPath.split('/').includes('..')) {
        throw requestError('Invalid path.', 403);
      }
      const url = new URL(request.url, `http://${request.headers.host}`);
      if (decodedPath.startsWith('/api/')) {
        if (decodedPath === '/api/reset' && request.method === 'POST') {
          const body = await readJson(request);
          validateReset(body);
          reset(simulation, { seed: body.seed ?? 42, scenario: 'tdm', teamSize: body.teamSize ?? 4, scoreLimit: body.scoreLimit ?? 25 });
          if (body.resolution !== undefined) resolution = validateResolution(body.resolution);
          json(response, 200, packet());
        } else if (decodedPath === '/api/step' && request.method === 'POST') {
          const body = await readJson(request);
          const { action, ticks } = validateAction(body);
          const transition = step(simulation, action, ticks);
          json(response, 200, packet(transition, body.includePixels === true));
        } else if (decodedPath === '/api/observe' && request.method === 'GET') {
          json(response, 200, packet({}, url.searchParams.get('pixels') === '1'));
        } else if (decodedPath === '/api/frame' && request.method === 'GET') {
          json(response, 200, frame());
        } else {
          json(response, 404, { error: 'Unknown API endpoint or method.' });
        }
        return;
      }
      if (request.method !== 'GET' && request.method !== 'HEAD') {
        json(response, 405, { error: 'Method not allowed.' });
        return;
      }
      const file = path.resolve(staticDirectory, '.' + (decodedPath === '/' ? '/index.html' : decodedPath));
      if (!file.startsWith(staticDirectory + path.sep)) throw requestError('Invalid path.', 403);
      let data;
      try { data = await readFile(file); } catch (error) {
        if (['ENOENT', 'EISDIR', 'ENOTDIR'].includes(error.code)) {
          json(response, 404, { error: 'File not found.' });
          return;
        }
        throw error;
      }
      const mime = {
        '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
        '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8',
        '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon',
      }[path.extname(file)] || 'application/octet-stream';
      response.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
      response.end(request.method === 'HEAD' ? undefined : data);
    } catch (error) {
      if (!response.headersSent && !response.destroyed) {
        const expected = error.status || error instanceof TypeError || error instanceof RangeError;
        json(response, error.status || (expected ? 400 : 500), {
          error: expected ? error.message : 'Internal server error.',
        });
        if (!expected) console.error(error);
      }
    }
  });
  server.requestTimeout = 10000;
  server.headersTimeout = 10000;
  return server;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 4173);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT must be from 1 to 65535.');
  const server = createServer();
  server.listen(port, '127.0.0.1', () => {
    console.log(`Local: http://127.0.0.1:${port}`);
    console.log('Agent API: independent headless environment at /api/reset, /api/step, /api/observe');
  });
  const shutdown = () => server.close(() => process.exit(0));
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}
