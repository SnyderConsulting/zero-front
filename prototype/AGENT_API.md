# Agent API

Zero Front is a team deathmatch prototype: **4 vs 4 by default, first team to 25 eliminations**. Team size is configurable from 1 to 8 players per side. The human/API player is on team 0 (blue), with scripted teammates against team 1 (red). Eliminated actors respawn after 2 seconds with 1.2 seconds of spawn protection. A match also ends after 180 simulated seconds.

All living actors regenerate 50 HP per simulated second after 3 seconds without taking damage, capped at 100 HP. New damage restarts that delay. Observation health retains fractional precision; the human HUD rounds up to a whole number. Regeneration advances only when the simulation steps.

The browser and local HTTP server use the same deterministic 60 Hz simulation. They run **independent matches**: HTTP steps do not move the visible browser player. Each HTTP server has one shared match; launch separate server processes on different ports for separate environments. There is no network multiplayer.

## Browser

The page exposes `window.FPSLab`. Select stepped mode before programmatic control:

```js
FPSLab.setMode('stepped');
FPSLab.reset({ seed: 42, teamSize: 4, scoreLimit: 25 });
FPSLab.step({ forward: 1, turn: 0.3, fire: true }, 6);
FPSLab.observe({ pixels: true });
```

`setMode('human')` resumes human play; `setMode('paused')` pauses it. This API controls the visible match in both browser play and the standalone game's renderer. The HTTP server described below runs a separate local match.

Visible matches record replays by default. After the round ends, the results screen shows team scores and per-player statistics; **Watch replay** opens the top-down timeline. See [README.md](./README.md#round-results-and-replay) for playback, pan, zoom, and player-follow controls. `FPSLab.reset({ recordReplay: false })` disables recording for programmatic runs; later resets preserve that setting unless explicitly changed. Starting a match through the human UI enables it again.

## Local learned controllers

The standalone setup can assign a model to each bot team. Direct JavaScript users can attach the same policy to individual actors:

```js
import {createSimulation, stepAgents, setController} from './dist/simulation.js';
import {validateModel, createPolicyController, policyObservation} from './dist/policy.js';

const sim = createSimulation({seed: 42, teamSize: 4});
const controller = createPolicyController(validateModel(parsedModelJSON));
setController(sim, 'red-1', controller);
stepAgents(sim, {}, 6);
const input = policyObservation(sim, 'red-1'); // Visible RGB bytes + six own-status scalars.
```

`setController(sim, id, 'scripted')` can also automate the player for headless all-bot matches; `null` restores that actor's default. Explicit `stepAgents` actions take precedence over attached controllers. Reset clears controllers, so reattach them after reset. The model controller caches each actor's action for six simulation ticks and refreshes on respawn. All bot roles use the same weapon spread regardless of whether their actions are scripted, learned, or supplied externally; a controller attached to `player` also uses bot spread.

The versioned `arena-pixel-policy` JSON format supports `cnn-v1` and `cnn-v2`: 64×36 semantic RGB, two convolution layers, one hidden layer, and six categorical action heads. V2 adds finer turn actions, increasing the output from 18 to 22 logits while retaining the same image input and six-tick cadence. Personal status is normalized health, ammo, reloading, spawn protection, pitch, and alive state. `policyObservation` renders from the selected actor with relative ally/enemy colors and normal wall occlusion; it does not expose world coordinates or hidden actor locations. The JSON model contains weights, observation/action contracts, and training metadata, never executable code. See [training/README.md](./training/README.md) for exact architecture and commands.

`scriptedAction(sim, id)` is a non-mutating teacher diagnostic for collecting imitation labels. The teacher knows hidden enemy locations for navigation; its labels and all replay/diagnostic state must stay out of the student's observation input. The HTTP adapter remains a separate environment and does not load models.

## Local replay recording

Direct simulation users can opt in without a browser or renderer:

```js
import { createSimulation, step } from './dist/simulation.js';
import { sampleReplay } from './dist/replay.js';

const sim = createSimulation({ seed: 42, teamSize: 4, recordReplay: true });
step(sim, { forward: 1 }, 600);
const frame = sampleReplay(sim.replay, 5); // Simulation seconds; clamps to recorded range.
// frame: { time, tick, scores, actors, events }
// sim.replay.summary is populated when the round ends.
```

Recording is off by default in `createSimulation` and the HTTP adapter. The HTTP API does not expose replay history. `sim.replay` contains a copied map and roster, seed/layout/settings, typed-array snapshots, full event history, and a completed-round `summary`. It captures tick 0, every 60 Hz tick (including all ticks inside a batched step), and the terminal tick. Actor samples include position, facing, health, ammo, alive state, kills/deaths, shots/hits, reload, respawn, and spawn-protection timers. `sampleReplay` interpolates movement and facing, keeps death/respawn transitions discrete, and returns events from the preceding 0.35 seconds without future events.

`summary` contains `winner`, `scores`, `duration`, `reason` (`score` or `time`), map/settings metadata, `players`, and team totals. Player statistics include kills, deaths, shots, hits, and `accuracy` as a percentage from 0 to 100. Snapshots use Float32 values; a full 180-second, 16-player round needs less than 10 MiB for frame and score buffers, plus event history. Recording does not change simulation randomness or stepping.

History stays in local memory for the current round; reset replaces it and preserves the recording option unless explicitly set to `false`. There is no disk persistence or upload. Replay data is omniscient diagnostic data and is not included in `observe()` or pixel observations; keep it out of visual-only policy inputs.

## Local HTTP

Run `npm run web` or `node server.mjs` in this directory (`npm start` opens the standalone game). The HTTP server requires Node.js 20+ and no dependencies. The default address is `http://127.0.0.1:4173`; use the `PORT` environment variable to change it. The server binds to loopback and rejects foreign browser origins.

| Endpoint | Input | Result |
| --- | --- | --- |
| `POST /api/reset` | `{"seed":42,"teamSize":4,"scoreLimit":25,"resolution":[320,180]}` | Starts a new team deathmatch |
| `POST /api/step` | `{"action":{"forward":1,"turn":0.3,"fire":true},"ticks":6,"includePixels":true}` | Advances the player and all bots by 6 ticks |
| `GET /api/observe` | None | Reads current player status and team scores |
| `GET /api/observe?pixels=1` | None | Reads status with an RGB frame |
| `GET /api/frame` | None | Reads the RGB image object alone |

POST bodies require `Content-Type: application/json` and must be at most 1 MiB. Bad inputs return HTTP 400, unsupported content types 415, and oversized bodies 413.

HTTP reset defaults to seed 42, team size 4, and score limit 25. Seeds are unsigned 32-bit integers; team size is an integer from 1 to 8; score limit is an integer from 1 to 100. The only scenario is `tdm`, so the `scenario` field can be omitted. Resolution can be `[160,90]`, `[320,180]`, or `[640,360]`; the initial default is `[320,180]`, and resets preserve it when omitted.

Actions contain any subset of:

| Field | Range | Meaning |
| --- | --- | --- |
| `forward` | -1 to 1 | Backward/forward movement |
| `strafe` | -1 to 1 | Left/right movement |
| `turn` | -12 to 12 | Yaw velocity in radians per second |
| `pitch` | -6 to 6 | Pitch velocity in radians per second, camera bounded to ±0.45 radians |
| `fire` | Boolean | Hold the trigger |
| `reload` | Boolean | Request a reload |

Missing fields mean no input. `ticks` is an integer from 1 to 600, default 1. The action is held for that interval, stopping when the match ends. Actors continue to play and respawn during stepping. Player death alone does not end the match. Invalid actions are rejected before time advances.

Reset, step, and observe return:

```js
{
  observation: { /* player status, scores, match status */ },
  reward: 0,
  terminated: false,
  truncated: false,
  info: { events: [] },
  pixels: { width: 320, height: 180, format: 'rgb8', data: 'BASE64...' }
}
```

`observation` includes health, ammo, kills, deaths, shots, hits, alive status, respawn timer, spawn protection, team, team size, score limit, `scores: [blue, red]`, elapsed time, and match status. `winner` is `null` during play, `0` or `1` for the winning team, or `'draw'`. Enemy positions and the world map are omitted.

`pixels` is included only when requested. Base64-decoded data contains exactly `width * height * 3` uncompressed RGB bytes, top-to-bottom row-major order. Browser pixel output may use typed arrays; this encoding describes HTTP responses. The RGB renderer respects occlusion. Diagnostic `info.events` should be excluded from policy input when evaluating visual-only agents.

Reward is summed over the requested step interval. `terminated` means the score limit was reached; `truncated` means the 180-second time cap was reached. At the time cap, the leading score wins or equal scores draw. Reset after either flag. Read-only observations return zero transition reward.

## Python example

With the local server running:

```sh
python examples/random_agent.py --team-size 4 --score-limit 25 --seed 42
python examples/random_agent.py --team-size 2 --snapshot frame.ppm
```

This standard-library-only script demonstrates reset, stepping, and optional RGB export to PPM. It is a random policy, not a trained agent.

For repeatability, retain the seed, team size, score limit, resolution, action sequence, and source version. The seed determines the procedural arena as well as bot randomness. Five layout families vary the overall structure: room networks, rings, warehouses, courtyards, and offset corridors. Footprints, room counts, route graphs, and cover vary within families. Geometry uses reflection or 180-degree symmetry for team balance. Generation uses a separate random stream from combat. Agent reset keeps deterministic seed behavior; the human New match button chooses a fresh seed from a different family than the preceding match. The HTTP adapter is a lightweight interface, not a Gymnasium wrapper or secure competition sandbox.
