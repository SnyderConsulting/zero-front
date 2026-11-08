"""Pixel-only shared-team PPO against unchanged scripted opponents (CUDA supported)."""
from __future__ import annotations
import argparse
import base64
from collections import deque
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
HEADS = [("forward", [-1, 0, 1]), ("strafe", [-1, 0, 1]),
         ("turn", [-3.6, -1.8, -.8, -.3, 0, .3, .8, 1.8, 3.6]),
         ("pitch", [-1.2, 0, 1.2]), ("fire", [False, True]), ("reload", [False, True])]
OLD_HEADS = [*HEADS[:2], ("turn", [-3.6, -1.2, 0, 1.2, 3.6]), *HEADS[3:]]
SCALARS = ["health", "ammo", "reloading", "spawnProtection", "pitch", "alive"]
TENSORS = {"conv1": ("conv1.weight", [8, 3, 5, 5]), "conv1Bias": ("conv1.bias", [8]),
           "conv2": ("conv2.weight", [16, 8, 3, 3]), "conv2Bias": ("conv2.bias", [16]),
           "fc": ("fc.weight", [64, 1574]), "fcBias": ("fc.bias", [64]),
           "policy": ("policy.weight", [22, 64]), "policyBias": ("policy.bias", [22])}

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.fc = nn.Linear(1574, 64)
        self.policy = nn.Linear(64, 22)
        self.value = nn.Linear(64, 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, pixels, scalars):
        image = pixels.permute(0, 3, 1, 2).contiguous().float() / 255
        visual = torch.relu(self.conv2(torch.relu(self.conv1(image)))).flatten(1)
        hidden = torch.relu(self.fc(torch.cat((visual, scalars), dim=1)))
        return self.policy(hidden), self.value(hidden).squeeze(-1)

    def act(self, pixels, scalars, action=None):
        logits, value = self(pixels, scalars)
        distributions = [Categorical(logits=head) for head in torch.split(logits, [len(bins) for _, bins in HEADS], dim=1)]
        if action is None:
            action = torch.stack([distribution.sample() for distribution in distributions], dim=1)
        logprob = torch.stack([distribution.log_prob(action[:, index]) for index, distribution in enumerate(distributions)], dim=1).sum(1)
        entropy = torch.stack([distribution.entropy() for distribution in distributions], dim=1).sum(1)
        return action, logprob, entropy, value


def load_actor(model, path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != "arena-pixel-policy" or raw.get("version") != 1 or raw.get("architecture") not in ("cnn-v1", "cnn-v2"):
        raise ValueError("Initial model must be Arena cnn-v1 or cnn-v2 JSON.")
    if raw.get("actionSpace") != ("discrete-v1" if raw["architecture"] == "cnn-v1" else "discrete-v2"):
        raise ValueError("Initial model action space differs from its architecture.")
    if raw.get("observation") != {"kind": "semantic-rgb", "width": 64, "height": 36, "scalars": SCALARS} or raw.get("actionRepeat") != 6:
        raise ValueError("Initial model observation/cadence differs.")
    state = model.state_dict()
    for name, (key, shape) in TENSORS.items():
        tensor = raw["weights"][name]
        expected = [18, 64] if name == "policy" and raw["architecture"] == "cnn-v1" else [18] if name == "policyBias" and raw["architecture"] == "cnn-v1" else shape
        data = np.asarray(tensor["data"], dtype=np.float32)
        if tensor["shape"] != expected or data.size != math.prod(expected) or not np.isfinite(data).all() or np.max(np.abs(data)) > 100:
            raise ValueError(f"Invalid {name} tensor.")
        data = torch.from_numpy(data.reshape(expected))
        if raw["architecture"] == "cnn-v1" and name in ("policy", "policyBias"):
            old_offset, rows = 0, []
            for (_, old_bins), (_, new_bins) in zip(OLD_HEADS, HEADS):
                for value in new_bins:
                    closest = min(range(len(old_bins)), key=lambda index: abs(float(old_bins[index]) - float(value)))
                    row = data[old_offset + closest]
                    # New fine-turn bins may copy zero-turn weights. Keep zero-turn as
                    # the deterministic warm-start winner until PPO actually learns otherwise.
                    if name == "policyBias":
                        row = row - abs(float(old_bins[closest]) - float(value)) * .001
                    rows.append(row)
                old_offset += len(old_bins)
            data = torch.stack(rows)
        state[key] = data
    model.load_state_dict(state)
    return raw


class NodeEnvironments:
    def __init__(self, args):
        self.process = subprocess.Popen([args.node, str(ROOT / "training/rl-env.mjs")], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.first = self.request({"command": "init", "environments": args.environments, "teamSize": args.team_size,
            "seedStart": args.seed_start, "seedCount": args.seed_count, "seconds": args.seconds,
            "heads": [{"name": name, "values": bins} for name, bins in HEADS],
            "navigationReward": args.navigation_reward, "explorationReward": args.exploration_reward})

    def request(self, command):
        self.process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Environment worker stopped: " + self.process.stderr.read())
        packet = json.loads(line)
        if packet.get("type") == "error":
            raise RuntimeError(packet["error"])
        count = packet["count"]
        packet["pixels"] = np.frombuffer(base64.b64decode(packet["pixels"]), dtype=np.uint8).reshape(count, 36, 64, 3).copy()
        for name in ("scalars", "alive", "rewards", "done"):
            packet[name] = np.asarray(packet[name], dtype=np.float32)
        packet["scalars"] = packet["scalars"].reshape(count, 6)
        return packet

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)


def advantages(rewards, values, dones, final_value, gamma=.995, lam=.95):
    """GAE uses the terminal mask of the transition, never the reset observation's value."""
    result = np.empty_like(rewards)
    carry = np.zeros_like(final_value)
    for step in reversed(range(len(rewards))):
        next_value = final_value if step == len(rewards) - 1 else values[step + 1]
        continuation = 1 - dones[step]
        delta = rewards[step] + gamma * next_value * continuation - values[step]
        carry = delta + gamma * lam * continuation * carry
        result[step] = carry
    return result, result + values


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_ranges(seeds, ranges=()):
    """Compact complete ancestry without exceeding the model's metadata array bound."""
    intervals = [[int(seed), int(seed) + 1] for seed in seeds] + [list(pair) for pair in ranges]
    for pair in intervals:
        if len(pair) != 2 or any(type(value) is not int for value in pair) or not 0 <= pair[0] < pair[1] <= 4294967296:
            raise ValueError("Invalid seed provenance range.")
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:merged[-1][1] = max(merged[-1][1], end)
        else:merged.append([start, end])
    if len(merged) > 128:raise ValueError("Too many disjoint seed ranges for model format v1.")
    return merged


def export(model, args, initial, total_steps, update, metrics, observation):
    state = model.state_dict()
    source = initial.get("training", {})
    prior_ranges = seed_ranges(source.get("trainSeeds", []) + source.get("priorTrainingSeeds", []),
        source.get("trainingSeedRanges", []) + source.get("priorTrainingSeedRanges", []))
    selection_ranges = seed_ranges(source.get("validationSeeds", []) + source.get("selectionSeeds", []),
        source.get("validationSeedRanges", []) + source.get("selectionSeedRanges", []))
    metadata = {"algorithm": "ppo", "status": "candidate", "seed": args.seed, "actorSteps": total_steps,
        "updates": update, "initialModel": initial.get("name", "random initialization"),
        "trainSeeds": [], "priorTrainingSeeds": [],
        "trainingSeedRanges": [[args.seed_start, args.seed_start + args.seed_count]], "priorTrainingSeedRanges": prior_ranges,
        "validationSeeds": list(range(4000, 4005)), "selectionSeedRanges": selection_ranges, "teamSize": args.team_size,
        "reward": {"hit": .15, "kill": 1, "death": -1.2, "teamScore": .2, "win": 6, "loss": -6,
                   "missedShot": -.002, "navigation": args.navigation_reward, "exploration": args.exploration_reward},
        "gamma": args.gamma, "gaeLambda": args.gae_lambda, "rolloutSteps": args.rollout_steps,
        "learningRate": args.learning_rate, "clip": args.clip, "entropyCoefficient": args.entropy,
        "trainingMetrics": metrics, "limitations": "Pixel-only policy. Rewards use simulation diagnostics; those are never model inputs. Training win rate uses sampled actions; deployed model uses argmax."}
    artifact = {"format": "arena-pixel-policy", "version": 1, "name": args.name,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "architecture": "cnn-v2",
        "observation": {"kind": "semantic-rgb", "width": 64, "height": 36, "scalars": SCALARS},
        "actionSpace": "discrete-v2", "actionRepeat": 6,
        "weights": {name: {"shape": shape, "data": state[key].detach().cpu().flatten().tolist()} for name, (key, shape) in TENSORS.items()},
        "training": metadata}
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    reference = ActorCritic().eval()
    reference.load_state_dict({key: value.detach().cpu() for key, value in state.items()})
    with torch.no_grad():
        logits = reference(torch.from_numpy(observation["pixels"][:1]), torch.from_numpy(observation["scalars"][:1]))[0][0].tolist()
    fixture = output.with_suffix(".parity.json")
    fixture.write_text(json.dumps({"pixels": base64.b64encode(observation["pixels"][0].tobytes()).decode(),
                                  "scalars": observation["scalars"][0].tolist(), "logits": logits}), encoding="utf-8")
    parity = subprocess.run([args.node, str(ROOT / "training/check-parity.mjs"), "--model", str(temporary), "--fixture", str(fixture)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if parity.returncode:
        raise RuntimeError("Export parity failed: " + parity.stderr)
    temporary.replace(output)
    output.with_suffix(".ppo-state.json").write_text(json.dumps({"actorSteps": total_steps, "updates": update,
        "valueWeight": state["value.weight"].detach().cpu().flatten().tolist(), "valueBias": state["value.bias"].detach().cpu().tolist(),
        "note": "Value weights only; optimizer and environment state restart on warm start."}), encoding="utf-8")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", default="models/pixel-baseline.json")
    parser.add_argument("--output", default="training/runs/ppo-candidate.json")
    parser.add_argument("--name", default="Pixel PPO candidate")
    parser.add_argument("--total-steps", type=int, default=1000000)
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--team-size", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=.995)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--clip", type=float, default=.15)
    parser.add_argument("--entropy", type=float, default=.003)
    parser.add_argument("--target-kl", type=float, default=.035)
    parser.add_argument("--navigation-reward", type=float, default=.025)
    parser.add_argument("--exploration-reward", type=float, default=.02)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--evaluate-every", type=int, default=25, help="Updates between full-team tuning evaluations; 0 disables.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--node", default="node")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.total_steps < 1 or not 1 <= args.environments <= 64 or not 1 <= args.team_size <= 8 or not 1 <= args.rollout_steps <= 1024:
        raise ValueError("Invalid environment or rollout sizes.")
    if not 1 <= args.minibatch <= 8192 or not 1 <= args.epochs <= 20 or args.save_every < 1 or args.evaluate_every < 0:
        raise ValueError("Invalid optimizer or checkpoint settings.")
    if not 0 < args.learning_rate <= .01 or not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1 or not 0 < args.clip < 1:
        raise ValueError("Invalid PPO parameters.")
    for start, end in [(4000, 4005), (50000, 50020)]:
        if args.seed_start < end and args.seed_start + args.seed_count > start:
            raise ValueError("Reserved tuning/final evaluation seeds cannot be used for training.")
    torch.manual_seed(args.seed);np.random.seed(args.seed & 0xffffffff)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = ActorCritic().to(device)
    initial = load_actor(model, resolve(args.initial))
    state_path = resolve(args.initial).with_suffix(".ppo-state.json")
    prior_steps = 0
    if state_path.exists():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        weight, bias = np.asarray(saved["valueWeight"], dtype=np.float32), np.asarray(saved["valueBias"], dtype=np.float32)
        if weight.shape != (64,) or bias.shape != (1,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError("Invalid value checkpoint.")
        with torch.no_grad():
            model.value.weight.copy_(torch.from_numpy(weight).reshape(1, 64));model.value.bias.copy_(torch.from_numpy(bias))
        prior_steps = int(saved.get("actorSteps", 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    env = NodeEnvironments(args)
    obs = env.first
    count = args.environments * args.team_size
    episodes = deque(maxlen=100)
    start_time, steps = time.perf_counter(), 0
    output = resolve(args.output);output.parent.mkdir(parents=True, exist_ok=True)
    csvfile = output.with_suffix(".ppo.csv").open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=["update", "actorSteps", "stepsPerSecond", "policyLoss", "valueLoss", "entropy", "kl", "completed", "winRate", "meanReturn"])
    writer.writeheader()
    metrics = {};update = 0
    best_tuning = (-1.0, -math.inf)
    try:
        while steps < args.total_steps:
            update += 1
            horizon = min(args.rollout_steps, math.ceil((args.total_steps - steps) / count))
            pixels = np.empty((horizon, count, 36, 64, 3), dtype=np.uint8)
            scalars = np.empty((horizon, count, 6), dtype=np.float32)
            actions = np.empty((horizon, count, 6), dtype=np.int64)
            oldlog = np.empty((horizon, count), dtype=np.float32)
            values = np.empty_like(oldlog);rewards = np.empty_like(oldlog);dones = np.empty_like(oldlog);alive = np.empty_like(oldlog)
            model.eval()
            for t in range(horizon):
                pixels[t], scalars[t], alive[t] = obs["pixels"], obs["scalars"], obs["alive"]
                with torch.no_grad():
                    action, logprob, _, value = model.act(torch.from_numpy(obs["pixels"]).to(device), torch.from_numpy(obs["scalars"]).to(device))
                actions[t], oldlog[t], values[t] = action.cpu().numpy(), logprob.cpu().numpy(), value.cpu().numpy()
                obs = env.request({"command": "step", "actions": actions[t].reshape(-1).tolist()})
                rewards[t], dones[t] = obs["rewards"], obs["done"]
                episodes.extend(row for row in obs["episodes"] if row["complete"])
                steps += count
            with torch.no_grad():
                final_value = model(torch.from_numpy(obs["pixels"]).to(device), torch.from_numpy(obs["scalars"]).to(device))[1].cpu().numpy()
            adv, returns = advantages(rewards, values, dones, final_value, args.gamma, args.gae_lambda)
            size = horizon * count
            tensors = [torch.from_numpy(array.reshape((size,) + array.shape[2:])).to(device) for array in (pixels, scalars, actions, oldlog, values, adv, returns, alive)]
            px, own, act, log_old, value_old, advantage, target, mask = tensors
            active_adv = advantage[mask > 0]
            advantage = (advantage - active_adv.mean()) / (active_adv.std(unbiased=False) + 1e-8) if active_adv.numel() else advantage * 0
            accumulated = [];stop = False;model.train()
            for epoch in range(args.epochs):
                for index in torch.randperm(size, device=device).split(args.minibatch):
                    _, logprob, entropy, value = model.act(px[index], own[index], act[index])
                    ratio = (logprob - log_old[index]).exp()
                    importance = mask[index];denominator = importance.sum().clamp_min(1)
                    actor_loss = (torch.maximum(-advantage[index] * ratio, -advantage[index] * ratio.clamp(1 - args.clip, 1 + args.clip)) * importance).sum() / denominator
                    clipped_value = value_old[index] + (value - value_old[index]).clamp(-.3, .3)
                    value_loss = .5 * torch.maximum((value - target[index]).square(), (clipped_value - target[index]).square()).mean()
                    entropy_mean = (entropy * importance).sum() / denominator
                    loss = actor_loss + .5 * value_loss - args.entropy * entropy_mean
                    if not torch.isfinite(loss):raise RuntimeError("PPO loss became non-finite.")
                    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(), .5);optimizer.step()
                    with torch.no_grad():
                        kl = (((ratio - 1) - (logprob - log_old[index])) * importance).sum() / denominator
                    accumulated.append([actor_loss.item(), value_loss.item(), entropy_mean.item(), kl.item()])
                    if kl > args.target_kl:stop = True;break
                if stop:break
            mean = np.asarray(accumulated).mean(0)
            metrics = {"update": update, "actorSteps": prior_steps + steps, "stepsPerSecond": steps / (time.perf_counter() - start_time),
                "policyLoss": float(mean[0]), "valueLoss": float(mean[1]), "entropy": float(mean[2]), "kl": float(mean[3]),
                "completed": len(episodes), "winRate": sum(row["win"] for row in episodes) / max(1, len(episodes)),
                "meanReturn": sum(row["returns"] for row in episodes) / max(1, len(episodes))}
            writer.writerow(metrics);csvfile.flush();print(json.dumps(metrics), flush=True)
            checkpoint = update % args.save_every == 0 or steps >= args.total_steps
            tuning = args.evaluate_every and update % args.evaluate_every == 0
            if checkpoint or tuning:
                export(model, args, initial, prior_steps + steps, update, metrics, obs)
                if tuning:
                    report = output.with_suffix(".tuning.json")
                    result = subprocess.run([args.node, str(ROOT / "training/evaluate-teams.mjs"), "--model", str(output), "--seeds", "4000:4005", "--purpose", "tuning", "--team-size", "4", "--output", str(report)],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if result.returncode:raise RuntimeError("Tuning evaluation failed: " + result.stderr)
                    measured = json.loads(report.read_text(encoding="utf-8"))
                    overall = measured["overall"]
                    ranking = (overall["winRate"], overall["modelScore"] - overall["scriptedScore"])
                    if ranking > best_tuning:
                        best_tuning = ranking
                        shutil.copyfile(output, output.with_suffix(".best.json"))
                        shutil.copyfile(report, output.with_suffix(".best.tuning.json"))
                        shutil.copyfile(output.with_suffix(".ppo-state.json"), output.with_suffix(".best.ppo-state.json"))
                    print(json.dumps({"tuning": str(report), "wins": overall["wins"], "matches": overall["matches"],
                                      "score": [overall["modelScore"], overall["scriptedScore"]], "best": list(best_tuning)}), flush=True)
    except KeyboardInterrupt:
        if metrics:
            export(model, args, initial, prior_steps + steps, update, metrics, obs)
        print("Stopped; last completed update exported.", flush=True)
    finally:
        env.close();csvfile.close()
    print(json.dumps({"model": str(output), "actorSteps": prior_steps + steps, "elapsedSeconds": time.perf_counter() - start_time}), flush=True)

if __name__ == "__main__":
    try:main()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"PPO failed: {error}", file=sys.stderr);sys.exit(1)
