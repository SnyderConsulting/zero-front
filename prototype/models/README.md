# Bundled pixel policy

`pixel-selfplay.json` contains the **Pixel self-play** inference weights included with this repository. The standalone game selects this model for both teams' bots on first launch; the human stays under your control. No Python or PyTorch installation is needed for inference.

The policy was initialized from scripted demonstrations and teacher corrections, then refined through self-play against its current and historical policies. Its `training` metadata records the checkpoint's update count, agent decisions, source hash, seed ancestry, and reward settings. This is an experimental agent; consistent superiority over Scripted has not been established.

## Observation and action contract

- **Input:** 64×36 first-person semantic RGB pixels plus normalized health, ammunition, reload state, spawn protection, pitch, and alive state.
- **Architecture:** `cnn-v2`, with two convolution layers and six categorical action heads.
- **Actions:** forward/backward movement, strafing, turning, pitch, firing, and reloading, selected every six simulation ticks (10 Hz).
- **Deployment:** deterministic argmax actions. No map, world coordinates, or hidden enemy positions are supplied.
- **Format:** version 1 of `arena-pixel-policy`; the legacy format identifier is retained for compatibility. Files contain JSON weights and metadata, never executable model code.

## Updates and training

These weights are a snapshot at the repository commit. Use `git pull` to obtain later published snapshots. The game reads local files and does not download weights automatically. Open **New match** or press **Refresh** in round setup after an update; a running round keeps its existing weights.

A local trainer publishes to its configured output path. Imported or locally trained policies can be placed alongside the bundled model and selected separately. Avoid overwriting the bundled file if you want to keep your own experiments independent of repository updates.

This JSON is sufficient for play and evaluation. It does not include the original critic, optimizer, random state, or opponent pool needed to resume that training run. The continuous self-play trainer currently accepts imitation-stage initializers, so the bundled PPO-refined export is not accepted by its `--initial` option. See the [training guide](../training/README.md) for the supported imitation-to-self-play workflow.
