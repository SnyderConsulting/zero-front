import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createServer } from '../server.mjs';

async function withServer(run) {
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  try { await run(base, server); } finally {
    await new Promise(resolve => server.close(resolve));
  }
}

async function post(base, route, body, headers = {}) {
  return fetch(base + route, {
    method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body),
  });
}

test('team deathmatch resets its teams and scores, and deterministic stepping returns RGB', async () => {
  await withServer(async base => {
    const setup = { seed: 73, teamSize: 3, scoreLimit: 10, resolution: [160, 90] };
    const sequence = { action: { forward: 0.6, turn: 0.2 }, ticks: 30, includePixels: true };
    const initialResponse = await post(base, '/api/reset', setup);
    assert.equal(initialResponse.status, 200);
    const initial = await initialResponse.json();
    assert.equal(initial.observation.scenario, 'tdm');
    assert.equal(initial.observation.teamSize, 3);
    assert.equal(initial.observation.scoreLimit, 10);
    assert.deepEqual(initial.observation.scores, [0, 0]);
    assert.equal(initial.observation.ammo, 20);
    const first = await (await post(base, '/api/step', sequence)).json();
    assert.equal(first.observation.time, 0.5);
    assert.equal(first.pixels.format, 'rgb8');
    assert.equal(Buffer.from(first.pixels.data, 'base64').length, 160 * 90 * 3);
    assert.ok(!('entities' in first.observation));
    assert.ok(!('map' in first.observation));
    await post(base, '/api/reset', setup);
    const second = await (await post(base, '/api/step', sequence)).json();
    assert.deepEqual(second, first);
    const lightweight = await (await fetch(base + '/api/observe')).json();
    assert.ok(!('pixels' in lightweight));
    const frame = await (await fetch(base + '/api/frame')).json();
    assert.deepEqual(frame, first.pixels);
    const defaults = await (await post(base, '/api/reset', {})).json();
    assert.equal(defaults.observation.teamSize, 4);
    assert.equal(defaults.observation.scoreLimit, 25);
    assert.deepEqual(defaults.observation.scores, [0, 0]);
  });
});

test('bad input and foreign browser origins do not mutate the episode', async () => {
  await withServer(async base => {
    for (const body of [
      { ticks: 0 }, { ticks: 601 }, { ticks: 1.5 }, { action: { forward: 2 } },
      { action: { fire: 'yes' } }, { action: { teleport: true } }, { includePixels: 'yes' },
    ]) assert.equal((await post(base, '/api/step', body)).status, 400);
    assert.equal((await post(base, '/api/reset', { resolution: [4000, 2000] })).status, 400);
    assert.equal((await post(base, '/api/reset', { seed: -1 })).status, 400);
    assert.equal((await post(base, '/api/reset', { scenario: 'missing' })).status, 400);
    assert.equal((await post(base, '/api/reset', { scenario: 'range' })).status, 400);
    for (const teamSize of [0, 9, 1.5, '4']) {
      assert.equal((await post(base, '/api/reset', { teamSize })).status, 400);
    }
    for (const scoreLimit of [0, 101, 2.5, '25']) {
      assert.equal((await post(base, '/api/reset', { scoreLimit })).status, 400);
    }
    assert.equal((await post(base, '/api/step', { ticks: 1 }, { Origin: 'https://example.com' })).status, 403);
    const malformed = await fetch(base + '/api/step', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{',
    });
    assert.equal(malformed.status, 400);
    const state = await (await fetch(base + '/api/observe')).json();
    assert.equal(state.observation.time, 0);
  });
});

test('server serves modules and denies paths outside the static directory', async () => {
  await withServer(async (base, server) => {
    const module = await fetch(base + '/simulation.js');
    assert.equal(module.status, 200);
    assert.match(module.headers.get('content-type'), /javascript/);
    assert.equal((await fetch(base + '/missing-file')).status, 404);
    const traversalStatus = await new Promise((resolve, reject) => {
      http.get({ hostname: '127.0.0.1', port: server.address().port, path: '/%2e%2e/server.mjs' }, response => {
        response.resume(); resolve(response.statusCode);
      }).on('error', reject);
    });
    assert.equal(traversalStatus, 403);
  });
});
