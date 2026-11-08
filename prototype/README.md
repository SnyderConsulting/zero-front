# Zero Front

A minimal, playable FPS team deathmatch prototype. **Blue vs red, 4 vs 4 by default, first team to 25 eliminations.** Play alongside AI teammates against an AI team, with automatic respawns. Team size supports 1–8 actors per side. This is one human playing locally with bots, not network multiplayer.

Maps use five different procedural structures: irregular room networks, rings around central obstructions, warehouses, linked courtyards, and offset corridors. Seeds change the footprint, room count, connections, open space, and cover within each structure. Reflection or 180-degree symmetry gives both teams equivalent starting areas. **New match** selects a different structure from the previous match; Resume keeps the current map. Agents can reproduce a layout and match by using the same seed and action sequence.

## Run

The repository includes the trained **Pixel self-play** model. On first launch, the standalone game selects it for both teams' bots. Python and PyTorch are only needed for training, not playing.

From this directory, install Node.js 22.12+ and run:

```sh
npm install
npm start
```

This launches a dedicated Electron game window. After dependency installation, Windows users can also double-click **Play Zero Front.cmd** in the parent folder. Choose team size and the model for each team's bots, then select Play. The mouse is captured while playing. Switching to another app releases the mouse and pauses; returning to the game resumes it. Escape deliberately pauses and releases the mouse until you select Resume. F11 toggles fullscreen; close the window or use Alt+F4 to quit.

The optional browser and HTTP agent server remains available with `npm run web`, normally at `http://127.0.0.1:4173`.

## Controls

- **WASD:** move
- **Mouse or arrow keys:** aim
- **Left mouse button or Space:** fire
- **R:** reload
- **Escape:** pause
- **F11:** toggle fullscreen in the standalone game

Mouse capture is automatic in the standalone game. In embedded browsers that do not support capture, right-drag or arrow keys remain available for aiming. Eliminated players respawn after 2 seconds with brief spawn protection. A match also ends after 3 minutes, with the higher score winning.

Health regenerates after 3 seconds without taking damage, at 50 HP per second up to 100. Any new damage restarts the delay. The same rules apply to humans and bots; healing uses simulation time, so pausing also pauses recovery. This scales the [Modern Warfare III Season 2 multiplayer baseline](https://www.callofduty.com/patchnotes/2024/02/call-of-duty-modern-warfare-iii-season-2-patch-notes) to the arena's 100 HP. Tune `HEALTH_REGEN_DELAY` and `HEALTH_REGEN_RATE` in `dist/simulation.js`.

## Round results and replay

At the end of a round, the results show the winner, final score, duration, map layout, and each player's kills, deaths, accuracy, and hits/shots. Select **Watch replay** to review the entire round from above. Every player has a facing arrow; shots, eliminations, respawns, health, ammo, and reload state are visible as you move through the timeline.

- **Play/Pause or Space:** control playback; choose a speed from ¼× to 4×.
- **Timeline or Left/Right:** seek through the round; arrows skip 1 second, or 5 seconds with Shift. Home/End jump to the start/end.
- **Drag:** pan. **Scroll or +/−:** zoom. **Fit map, F, or double-click:** show the whole arena.
- **Click a player or use the Player selector:** inspect them; **Follow** keeps the camera on them.
- **Results or Escape:** return to the scoreboard. **New match:** start another round.

Replays record every simulation tick at 60 Hz, including the initial and final states, and interpolate movement for playback. The latest round is available locally in memory; starting a new match replaces it. Replays are not saved to disk or uploaded. Pausing the match adds no replay time.

## Agent access

Round setup has separate **Blue bots** and **Red bots** choices. **Scripted** uses the original pathfinding/combat rules; locally trained models appear alongside it. The human stays under your control. **New match** opens setup again; **Resume** keeps the current round's models.

**Pixel self-play** is bundled at `models/pixel-selfplay.json`. It was initialized from scripted demonstrations and teacher corrections on student-played games, then refined through self-play. It remains experimental and does not consistently beat Scripted. See [the model notes](./models/README.md) and [training workflow](./training/README.md).

The standalone game scans `models/*.json`. Select **Refresh** after training or **Import model…** to copy a compatible model into that library. Models are validated data-only weights; incompatible or missing selected models block Play. Browser imports work for the current session only; to use the bundled model in browser play, import `models/pixel-selfplay.json` and select it for each team.

Bundled weights are a snapshot at the repository commit. The game does not download newer versions automatically. After `git pull`, open **New match** or press **Refresh** in round setup to load a newly published snapshot. Running and resumed rounds retain their existing weights. A local trainer can also publish updated weights; refreshing reads those local files.

### Train a pixel model

A fresh clone includes inference weights, but not the original optimizer checkpoints, datasets, or training run. Follow the [training guide](./training/README.md#continuous-self-play) to create an imitation model and start a new continuous self-play run. Its PPO stage trains only against that run's current and historical policies. Rewards encourage damage, eliminations, team wins, and briefly keeping a visible enemy near the crosshair; they do not provide scripted actions or hidden model inputs. Training can continue indefinitely, saving every two minutes and benchmarking every ten minutes.

The first training workflow is behavioral cloning: a small convolutional network learns scripted-bot demonstrations from **64×36 first-person semantic RGB pixels** plus its own status. Its inputs do not include the map, coordinates, or hidden enemies. Decisions run at 10 Hz. The scripted teacher does use privileged map/target information, so the student cannot reproduce all its navigation from one frame; treat the included model as an experimental baseline.

With Python, PyTorch and NumPy available:

```sh
npm run train -- --name "My pixel model" --output models/my-pixel-model.json
npm run evaluate -- --model models/my-pixel-model.json
```

Training uses CUDA when available and exports the best validation checkpoint as JSON. This command runs the original imitation trainer. The [training guide](./training/README.md) also covers stronger demonstration policies, corrections from learner-visited states, PPO refinement, continuous self-play, and paired full-team benchmarks against other bots.

The renderer exposes `window.FPSLab`. The optional local server (`npm run web`) also provides a separate headless match for Python or HTTP clients. Both use the same deterministic 60 Hz simulation and low-resolution software renderer. See [AGENT_API.md](./AGENT_API.md).

```sh
python examples/random_agent.py --team-size 4
npm test
```

Static hosting supports browser play and its browser API. Run the local server for Python/HTTP use. HTTP actions do not control the visible browser match.

This prototype uses seeded procedural arenas, simple materials, one weapon, and 2.5D rendering. Bots can use the scripted baseline or compatible trained pixel models. Source-level access is unrestricted, so this is not a secure competition environment.
