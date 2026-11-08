# Pixel policy training

The workflow trains a fresh pixel policy from scripted demonstrations, adds teacher corrections on states visited by the student (DAgger), then continues with self-play PPO. The repository includes a **Pixel self-play** inference snapshot at `models/pixel-selfplay.json`. Local datasets, optimizer checkpoints, logs, and run reports are excluded from Git.

The policy receives 64×36 first-person semantic RGB pixels and six own-status values: health, ammunition, reloading, spawn protection, pitch, and alive. No world coordinates, hidden enemies, map, or teacher state enter the network. The `cnn-v2` model has nine turn speeds, including fine adjustments, and acts every six simulation ticks. The game uses deterministic actions; PPO samples actions during training.

Run commands below from `prototype/`, after installing Node.js dependencies with `npm install`. Training also requires Python and the dependencies in `training/requirements.txt`:

```powershell
python -m pip install -r training/requirements.txt
```

The training examples use CUDA. Use `--device cpu` if a CUDA-capable GPU and compatible PyTorch installation are unavailable. Playing the game or evaluating exported weights in Node.js does not require Python or PyTorch.

## Teacher pretraining and correction

`train-expert.py` first teaches visible-pixel aiming and combat from a scripted expert with privileged map navigation. Subsequent `--reactive-nav` demonstrations and student corrections teach navigation from visible wall/floor cues. This avoids requiring the student to infer the teacher's hidden route choice from a single image. The teacher is used only for supervision; its code is never attached to the deployed learned controller. The camping/scan-only teacher is not used in this workflow.

Validation uses separate maps. Training keeps the weights with the lowest validation loss, reduces the learning rate on a plateau, and can stop when improvement stalls. DAgger stages let the learned policy control a team, label its encountered states with teacher actions, and mix original demonstration data into retraining. Dataset caches include engine, renderer, policy, teacher, and learner hashes and carry seed ancestry.

Run from `prototype/` with Python, PyTorch, NumPy, and Node.js installed:

```powershell
python training/train-expert.py --device cuda --train-seeds 10000:10120 --validation-seeds 20000:20020 --seconds 180 --max-samples 600000 --max-validation-samples 60000 --epochs 60 --patience 10 --lr-patience 3 --min-delta 0.0002 --batch-size 512 --learning-rate 0.0005 --collection-workers 4 --cache-dir training/runs/teacher-bootstrap/cache --name "Pixel teacher student" --output training/runs/teacher-bootstrap/bc.json --parity-output training/runs/teacher-bootstrap/bc-parity.json
```

For a correction stage, use `--resume <student.json> --learner <student.json> --replay-data <datasetCache>`, where `datasetCache` is recorded in the first model's training metadata. Repeat `--replay-data` to aggregate earlier correction datasets as well. Add `--reactive-nav` for the visible-navigation teacher. `--collection-workers 4` collects independent map groups concurrently while preserving serial sample order and quotas. Keep training maps separate from validation maps. Outputs under `training/runs/` remain local.

Imitation accuracy alone does not establish playing strength. Development evaluation checks full learned teams against Scripted and against themselves:

```powershell
node training/evaluate-bootstrap.mjs --model training/runs/teacher-bootstrap/bc.json --seeds 610000:610005 --workers 4 --output training/runs/teacher-bootstrap/bc-evaluation.json
```

Each seed produces two scripted-opponent matches (both sides) and one self match. All use the game's actual deterministic controller and standard 180-second/25-kill rules. Reports include kills, hits, zero-contact matches, time to first contact, and actual movement. These are development maps and must be recorded as selection ancestry when choosing a starting policy. They are distinct from the periodic benchmark maps.

## Continuous self-play

### Start a new run from an imitation model

After generating `training/runs/teacher-bootstrap/bc.json` above, start a new run with separate checkpoint and model-output paths:

```powershell
python training/selfplay_control.py start --run-dir training/runs/my-selfplay -- --initial training/runs/teacher-bootstrap/bc.json --output models/my-selfplay.json --name "My self-play" --learning-rate 0.00003 --clip 0.1 --entropy 0.001 --critic-warmup-updates 8 --benchmark-minutes 10
```

Use a corrected imitation model instead of `bc.json` if you have completed DAgger stages. Keep the initializer unchanged for the lifetime of the run; its hash is checked on resume. The bundled `models/pixel-selfplay.json` is already PPO-refined and is **not accepted** by this trainer's imitation-only `--initial` option. It is ready for play and evaluation, but cannot resume the original run without that run's full checkpoint.

The initializer supplies the actor and visual network; the critic and optimizer start fresh. Eight critic warmup updates keep actor weights fixed while the value head learns. Subsequent PPO updates use only games between the current policy and frozen policies from this run. No scripted opponents or teacher action labels enter this stage.

Manage your new run with:

```powershell
python training/selfplay_control.py status --run-dir training/runs/my-selfplay
python training/selfplay_control.py stop --run-dir training/runs/my-selfplay
python training/selfplay_control.py start --run-dir training/runs/my-selfplay
```

The control tool preserves the initial launch arguments on resume. The supervisor runs indefinitely, including after the game closes, while the computer remains awake. It does not configure Windows startup or power settings. `stop` saves at an update boundary. Restarting `start` after a shutdown resumes the latest checkpoint. Refresh round setup and choose **My self-play** to play with your own exported model.

### Original development run

The bundled model descends from 600,000 initial demonstrations, 360,000 examples from the visible-navigation teacher, and 240,000 student corrections (plus 240,000 replay examples in the last stage). The selected stages trained for 60, 33, and 27 epochs, exporting their best validation checkpoints. Before self-play refinement, the initializer won 9/40 paired matches against Scripted on 20 fresh maps. Its 20 self matches produced 736 kills, contact in 19/20 maps, and three draws. These describe the initializer, not the current bundled checkpoint; consistent superiority over Scripted has not been established.

The original machine keeps its immutable initializer at `training/runs/teacher-selfplay/initial-policy.json`, its evaluation at `training/runs/teacher-selfplay/final-evaluation.json`, and its stage record at `training/runs/teacher-bootstrap/summary.json`. These local files are not included in a fresh clone. On that machine, `python training/selfplay_control.py start`, `status`, and `stop` manage the existing run using the default `training/runs/teacher-selfplay/` directory.

### Rewards and monitoring

The active `shaped-v2` reward is:

| Event | Reward |
| --- | ---: |
| Actual enemy health removed | +0.01 per HP |
| Own health lost | −0.01 per HP |
| Own kill / death | +1 / −1 |
| Team score gain minus opponent score gain | 0.25 × score difference |
| Complete match win / loss / draw | +5 / −5 / −0.5 |
| Visible enemy near the crosshair | Up to +0.02 per second, capped at +0.2 per life |

Crosshair credit uses enemy-colored pixels from the policy observation. The central 2×2 pixels receive full quality, falling to zero at radius four pixels. Only the nearest enemy pixel contributes; own actor must remain alive and unprotected. Partial test cutoffs do not receive match-outcome rewards. There is no movement or hidden-position reward. These are engineered rewards; policy actions remain learned.

Defaults are eight 4v4 arenas, 10 Hz decisions, and training seeds 1000000–1009999. Half of matches sample a fixed historical opponent; the others use the current policy on both teams. Historical actors are excluded from optimization. The pool retains the initial teacher-trained policy and seven recent versions.

Every two minutes, the current weights are atomically published to the configured `--output` path (`models/my-selfplay.json` in the new-run example; `models/pixel-selfplay.json` in the original development run). Opening New Match reloads the selected model. If setup has remained open, press Refresh. Running and resumed rounds retain the weights they started with. Repository weights are a commit snapshot: `git pull` retrieves later published versions, while the game itself never downloads weights.

Every ten minutes, and once at startup, a separate worker benchmarks the published policy against Scripted and other compatible models in `models/`, excluding the candidate's own weights. Each opponent receives ten 4v4 matches on five maps, seeds 600000–600004, played from both sides. Benchmarks never supply rewards, actions, optimization data, pool members, or automatic checkpoint selection. Repeated monitoring maps are not a fresh final test.

## Checkpoints and provenance

The self-play run contains:

- The initializer at the chosen `--initial` path: immutable demonstration-trained weights, identified by SHA-256; it need not be inside the run directory.
- `checkpoint.pt` and `checkpoint.previous.pt`: actor, critic, optimizer, random states, opponent pool, counters, and configuration.
- `run.json` and `status.json`: lineage, configuration, heartbeat, progress, rewards, and benchmark status.
- `metrics.jsonl` and service logs: bounded training diagnostics.
- `benchmarks/`: bounded frozen checkpoint evaluations.

Teacher-initialized exports explicitly record their imitation ancestry and that the later self-play stage uses no demonstrations. Ordinary resume rejects changed initialization, configuration, or game/trainer source. Full resume restores optimizer and random states, but starts new matches instead of reconstructing unfinished games.

The trainer still supports random initialization when `--initial` is omitted, using a separate run directory and output. It never silently represents an imitation-initialized model as learning from scratch.

## Verification

Game exports pass a PyTorch-to-JavaScript inference parity check. Tests cover teacher collection, correction data and ancestry, model loading, rewards, evaluation role assignment, warm starts, critic-only warmup, and checkpoint recovery:

```powershell
npm test
python -m unittest training.test_training training.test_expert_training training.test_ppo training.test_selfplay
```
