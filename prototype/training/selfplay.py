"""Continuous pixel PPO from random or imitation weights, using this run's own policies."""
from __future__ import annotations
import argparse
import base64
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn

try:
    from .ppo import ActorCritic, HEADS, SCALARS, TENSORS, advantages, load_actor, seed_ranges
except ImportError:
    from ppo import ActorCritic, HEADS, SCALARS, TENSORS, advantages, load_actor, seed_ranges

ROOT = Path(__file__).resolve().parents[1]
STATE_FORMAT = "arena-strict-selfplay-state-v1"
TEACHER_STATE_FORMAT = "arena-teacher-selfplay-state-v1"
CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_hash():
    digest = hashlib.sha256()
    for name in ("training/selfplay.py", "training/selfplay-env.mjs", "training/ppo.py",
                 "dist/policy.js", "dist/simulation.js", "dist/renderer.js", "dist/mapgen.js"):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def log_event(directory, event):
    path = directory / "metrics.jsonl"
    if path.exists() and path.stat().st_size > 20 * 1024 * 1024:
        os.replace(path, directory / "metrics.previous.jsonl")
    line = json.dumps({"at": utc(), **event}, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


class Environments:
    def __init__(self, args, directory):
        self.errors = (directory / "environment-errors.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen([args.node, str(ROOT / "training/selfplay-env.mjs")], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
            text=True, encoding="utf-8", creationflags=CREATION_FLAGS)
        self.first = self.request({"command": "init", "environments": args.environments,
            "teamSize": args.team_size, "seedStart": args.seed_start, "seedCount": args.seed_count,
            "rewardProfile": args.reward_profile,
            "seconds": 180, "heads": [{"name": name, "values": bins} for name, bins in HEADS]})

    def request(self, command):
        self.process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Self-play environment stopped; see environment-errors.log")
        packet = json.loads(line)
        if packet.get("type") == "error":
            raise RuntimeError(packet["error"])
        count = packet["count"]
        packet["pixels"] = np.frombuffer(base64.b64decode(packet["pixels"]), dtype=np.uint8).reshape(count, 36, 64, 3).copy()
        for name in ("scalars", "alive", "rewards", "done"):
            packet[name] = np.asarray(packet[name], dtype=np.float32)
        packet["scalars"] = packet["scalars"].reshape(count, 6)
        for name in ("teams", "environments", "episodeIds"):
            packet[name] = np.asarray(packet[name], dtype=np.int64)
        return packet

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.errors.close()


class Opponents:
    """Frozen policies originate only in this run, and stay attached for a whole match."""
    def __init__(self, device, rng, capacity=8, historical_probability=.5):
        self.device, self.rng = device, rng
        self.capacity, self.probability = capacity, historical_probability
        self.pool = []
        self.assignments = {}

    def add(self, state, update):
        with torch.random.fork_rng(devices=[]):
            policy = ActorCritic().to(self.device).eval()
        policy.load_state_dict(state)
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        self.pool.append((int(update), policy))
        if len(self.pool) > self.capacity:
            # Retain the initial policy plus the latest seven versions. Active games own
            # references to evicted versions until their round ends.
            self.pool.pop(1)

    def route(self, observation):
        learner = np.ones(observation["count"], dtype=np.float32)
        frozen = []
        for environment in np.unique(observation["environments"]):
            indices = np.flatnonzero(observation["environments"] == environment)
            episode = int(observation["episodeIds"][indices[0]])
            assignment = self.assignments.get(int(environment))
            if assignment is None or assignment[0] != episode:
                if self.pool and self.rng.random() < self.probability:
                    assignment = (episode, self.rng.randrange(2), self.rng.choice(self.pool))
                else:
                    assignment = (episode, None, None)
                self.assignments[int(environment)] = assignment
            _, team, entry = assignment
            if entry is not None:
                chosen = indices[observation["teams"][indices] == team]
                learner[chosen] = 0
                frozen.append((chosen, entry[1]))
        return learner, frozen

    def states(self):
        return [{"update": update, "weights": cpu_state(policy)} for update, policy in self.pool]


def training_config(args):
    keys = ("seed", "seed_start", "seed_count", "environments", "team_size", "rollout_steps",
            "minibatch", "epochs", "learning_rate", "gamma", "gae_lambda", "clip", "entropy",
            "target_kl", "historical_probability", "pool_size", "reward_profile")
    config = {key: getattr(args, key) for key in keys}
    if args.critic_warmup_updates:
        config["critic_warmup_updates"] = args.critic_warmup_updates
    if args.initial:
        initial = Path(args.initial)
        if not initial.is_absolute():
            initial = ROOT / initial
        initial = initial.resolve()
        config["initial"] = {"path": str(initial), "sha256": hashlib.sha256(initial.read_bytes()).hexdigest()}
    return config


def initialize_actor(model, args, config):
    """Load only the imitation actor; no teacher code participates in PPO rollouts."""
    if not config.get("initial"):
        return {"origin": "random", "initialModel": None, "initialModelSha256": None,
                "initialTraining": None}
    initial = config["initial"]
    path = Path(initial["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != initial["sha256"]:
        raise ValueError("Initial model changed while starting the run.")
    raw = load_actor(model, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != initial["sha256"]:
        raise ValueError("Initial model changed while loading its weights.")
    source = raw.get("training", {})
    if not isinstance(source, dict) or source.get("algorithm") not in ("behavioral-cloning", "behavioral-cloning-expert", "dagger"):
        raise ValueError("Teacher initialization requires an imitation-trained model with recorded provenance.")
    training = seed_ranges(source.get("trainSeeds", []) + source.get("priorTrainingSeeds", []),
                           source.get("trainingSeedRanges", []) + source.get("priorTrainingSeedRanges", []))
    validation = seed_ranges(source.get("validationSeeds", []) + source.get("priorValidationSeeds", []),
                             source.get("validationSeedRanges", []) + source.get("priorValidationSeedRanges", []))
    selection = seed_ranges(source.get("selectionSeeds", []), source.get("selectionSeedRanges", []))
    if any(start < 600005 and end > 600000 for start, end in training + validation + selection):
        raise ValueError("Initial model provenance overlaps periodic benchmark seeds.")
    if any(start < args.seed_start + args.seed_count and end > args.seed_start for start, end in validation + selection):
        raise ValueError("Self-play training seeds overlap initial model validation/selection seeds.")
    # Keep compact ancestry in exports; the immutable initial-policy file retains
    # the complete demonstration/training record and is identified by its hash.
    provenance = {"algorithm": source["algorithm"], "teacher": source.get("teacher"),
                  "architecture": raw["architecture"], "samples": source.get("samples"),
                  "trainingSeedRanges": training, "validationSeedRanges": validation,
                  "selectionSeedRanges": selection}
    return {"origin": "teacher-initialized", "initialModel": raw.get("name", path.name),
            "initialModelSha256": initial["sha256"], "initialTraining": provenance}


def set_training_phase(model, args, updates):
    """A new value head can learn returns without disturbing the pretrained actor."""
    warmup = updates < args.critic_warmup_updates
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not warmup or name.startswith("value."))
    return warmup


def save_state(directory, model, optimizer, opponents, rng, config, identity, counters):
    origin = identity.get("origin", "random")
    state = {"format": TEACHER_STATE_FORMAT if origin == "teacher-initialized" else STATE_FORMAT,
        "origin": origin, "initialModel": identity.get("initialModel"),
        "initialModelSha256": identity.get("initialModelSha256"), "initialTraining": identity.get("initialTraining"),
        "runId": identity["runId"],
        "createdAt": identity["createdAt"], "sourceSha256": identity["sourceSha256"],
        "rewardSpec": identity.get("rewardSpec"),
        "rewardHistory": identity.get("rewardHistory", []),
        "config": config, "counters": counters, "model": cpu_state(model),
        "optimizer": optimizer.state_dict(), "opponents": opponents.states(),
        "routingRandomState": rng.getstate(), "torchRandomState": torch.get_rng_state(),
        "cudaRandomStates": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "resumeNote": "Actor, critic, optimizer, random states and pool resume; unfinished matches restart."}
    temporary = directory / "checkpoint.partial"
    torch.save(state, temporary)
    current, previous = directory / "checkpoint.pt", directory / "checkpoint.previous.pt"
    if current.exists():
        os.replace(current, previous)
    os.replace(temporary, current)


def read_state(directory, config, digest):
    errors = []
    for path in (directory / "checkpoint.pt", directory / "checkpoint.previous.pt"):
        if not path.exists():
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            errors.append(f"{path.name}: {error}")
            continue
        expected_origin = "teacher-initialized" if config.get("initial") else "random"
        expected_format = TEACHER_STATE_FORMAT if config.get("initial") else STATE_FORMAT
        if not isinstance(state, dict) or state.get("format") != expected_format or state.get("origin") != expected_origin:
            raise ValueError("Resume rejected: checkpoint initialization mode differs from this run.")
        if state.get("config") != config:
            raise ValueError("Resume rejected: training settings changed. Use a new run directory for a different experiment.")
        if state.get("sourceSha256") != digest:
            raise ValueError("Resume rejected: training/game source changed. Preserve this run or start a separate run.")
        if config.get("initial") and (state.get("initialModelSha256") != config["initial"]["sha256"] or
                                      not isinstance(state.get("initialTraining"), dict) or not state.get("initialModel")):
            raise ValueError("Resume rejected: initial-model provenance differs or is missing.")
        try:
            validate_state_payload(state, config)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"{path.name}: {error}")
            continue
        return state
    if errors:
        raise ValueError("No readable checkpoint: " + "; ".join(errors))
    if (directory / "run.json").exists():
        raise ValueError("Existing run has no checkpoint. Refusing to silently restart its training.")
    return None


def validate_state_payload(state, config):
    """A readable but incomplete/corrupt checkpoint can fall back to the prior save."""
    if not isinstance(state["runId"], str) or not state["runId"] or not isinstance(state["createdAt"], str):
        raise ValueError("Invalid run identity.")
    counters = state["counters"]
    for key in ("updates", "actorSteps", "learnerSteps", "completedMatches"):
        if type(counters[key]) is not int or counters[key] < 0:
            raise ValueError("Invalid training counters.")
    if not isinstance(counters["wallSeconds"], (int, float)) or not math.isfinite(counters["wallSeconds"]) or counters["wallSeconds"] < 0:
        raise ValueError("Invalid elapsed training time.")
    shapes = {key: shape for key, shape in TENSORS.values()}
    shapes.update({"value.weight": [1, 64], "value.bias": [1]})

    def weights(value):
        if not isinstance(value, dict) or set(value) != set(shapes):
            raise ValueError("Incomplete policy/value tensors.")
        for key, shape in shapes.items():
            tensor = value[key]
            if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid saved {key} tensor.")

    weights(state["model"])
    pool = state["opponents"]
    if not isinstance(pool, list) or not 1 <= len(pool) <= config["pool_size"] or pool[0]["update"] != 0:
        raise ValueError("Invalid self-play snapshot pool.")
    for item in pool:
        if type(item["update"]) is not int or not 0 <= item["update"] <= counters["updates"]:
            raise ValueError("Invalid snapshot update.")
        weights(item["weights"])
    optimizer = state["optimizer"]
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError("Missing optimizer state.")
    if not isinstance(state["torchRandomState"], torch.Tensor) or state["torchRandomState"].dtype != torch.uint8:
        raise ValueError("Missing PyTorch random state.")
    if not isinstance(state["cudaRandomStates"], list) or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 for value in state["cudaRandomStates"]):
        raise ValueError("Invalid CUDA random states.")
    random.Random().setstate(state["routingRandomState"])


def publish(model, args, identity, counters, observation, directory):
    state = cpu_state(model)
    teacher_initialized = identity.get("origin") == "teacher-initialized"
    artifact = {"format": "arena-pixel-policy", "version": 1, "name": args.name,
        "createdAt": utc(), "architecture": "cnn-v2",
        "observation": {"kind": "semantic-rgb", "width": 64, "height": 36, "scalars": SCALARS},
        "actionSpace": "discrete-v2", "actionRepeat": 6,
        "weights": {name: {"shape": shape, "data": state[key].flatten().tolist()} for name, (key, shape) in TENSORS.items()},
        "training": {"algorithm": "ppo-self-play-after-imitation" if teacher_initialized else "ppo-strict-self-play",
            "origin": identity.get("origin", "random"), "runId": identity["runId"],
            "initialModel": identity.get("initialModel"), "initialModelSha256": identity.get("initialModelSha256"),
            "initialTraining": identity.get("initialTraining"), "demonstrations": teacher_initialized,
            "selfPlayDemonstrations": False, "scriptedTrainingOpponents": False,
            "sourceSha256": identity["sourceSha256"], "actorSteps": counters["actorSteps"],
            "learnerSteps": counters["learnerSteps"], "updates": counters["updates"],
            "trainingSeedRanges": [[args.seed_start, args.seed_start + args.seed_count]],
            "teamSize": args.team_size, "rewardProfile": args.reward_profile,
            "reward": identity.get("rewardSpec", {}),
            "rewardHistory": identity.get("rewardHistory", []),
            "opponents": "current policy and this run's frozen historical policies only",
            "benchmarkUse": "read-only monitoring; never used for optimization, selection, rewards or opponent pool",
            "deployment": "deterministic argmax; training samples categorical actions",
            "status": "continuously training; strength not guaranteed"}}
    if teacher_initialized:
        for source_key, export_key in (("trainingSeedRanges", "priorTrainingSeedRanges"),
                                       ("validationSeedRanges", "validationSeedRanges"),
                                       ("selectionSeedRanges", "selectionSeedRanges")):
            artifact["training"][export_key] = identity["initialTraining"][source_key]
    temporary = directory / "published-candidate.json"
    atomic_json(temporary, artifact)
    with torch.random.fork_rng(devices=[]):
        reference = ActorCritic().eval()
    reference.load_state_dict(state)
    with torch.no_grad():
        logits = reference(torch.from_numpy(observation["pixels"][:1]), torch.from_numpy(observation["scalars"][:1]))[0][0].tolist()
    fixture = directory / "parity.json"
    atomic_json(fixture, {"pixels": base64.b64encode(observation["pixels"][0].tobytes()).decode(),
                          "scalars": observation["scalars"][0].tolist(), "logits": logits})
    result = subprocess.run([args.node, str(ROOT / "training/check-parity.mjs"), "--model", str(temporary), "--fixture", str(fixture)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60, creationflags=CREATION_FLAGS)
    if result.returncode:
        raise RuntimeError("Self-play export parity failed: " + result.stderr)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".partial")
    staging.write_bytes(temporary.read_bytes())
    os.replace(staging, output)
    return output, json.loads(result.stdout)


class Benchmarks:
    """Separate CPU worker. Results are recorded but never passed to the learner."""
    def __init__(self, args, directory):
        self.args, self.directory = args, directory
        self.process, self.log = None, None
        self.last_started = 0.0
        self.latest = None
        self.job = None
        self.root = directory / "benchmarks"
        self.root.mkdir(exist_ok=True)

    def poll(self):
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            return
        self.log.close()
        self.latest = {**self.job, "finishedAt": utc(), "exitCode": code,
                       "state": "complete" if code == 0 else "failed"}
        atomic_json(self.directory / "benchmark-status.json", self.latest)
        log_event(self.directory, {"type": "benchmark_finished", **self.latest})
        self.process, self.log = None, None
        # Bound retention to 48 reports and associated frozen models/logs.
        # Include failed/interrupted jobs too, which may never produce a report.
        groups = sorted(self.root.glob("benchmark-*.model.json"))
        for frozen in groups[:-48]:
            path = frozen.with_name(frozen.name.removesuffix(".model.json") + ".json")
            for artifact in (path, frozen, path.with_suffix(".log")):
                artifact.unlink(missing_ok=True)

    def due(self):
        self.poll()
        return self.process is None and time.time() - self.last_started >= self.args.benchmark_minutes * 60

    def start(self, output, counters):
        name = f"benchmark-{int(time.time())}-{counters['updates']:09d}"
        report = self.root / (name + ".json")
        frozen = self.root / (name + ".model.json")
        frozen.write_bytes(output.read_bytes())
        self.log = report.with_suffix(".log").open("w", encoding="utf-8")
        self.process = subprocess.Popen([self.args.node, str(ROOT / "training/benchmark-models.mjs"),
            "--model", str(frozen), "--models-dir", str(ROOT / "models"), "--exclude-model", str(output),
            "--seeds", "600000:600005", "--team-size", str(self.args.team_size), "--output", str(report)],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=CREATION_FLAGS | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
        self.last_started = time.time()
        self.job = {"startedAt": utc(), "pid": self.process.pid, "update": counters["updates"],
                    "actorSteps": counters["actorSteps"], "report": str(report), "state": "running"}
        atomic_json(self.directory / "benchmark-status.json", self.job)
        log_event(self.directory, {"type": "benchmark_started", **self.job})

    def close(self):
        self.poll()
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.log.close()
            atomic_json(self.directory / "benchmark-status.json", {**self.job, "state": "interrupted", "finishedAt": utc()})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="training/runs/teacher-selfplay")
    parser.add_argument("--output", default="models/pixel-selfplay.json")
    parser.add_argument("--name", default="Pixel self-play")
    parser.add_argument("--initial", help="Imitation policy JSON used only to initialize a new self-play run.")
    parser.add_argument("--critic-warmup-updates", type=int, default=0,
                        help="Initial updates fitting only the value head; actor and visual weights stay fixed.")
    parser.add_argument("--reward-profile", choices=["sparse-v1", "shaped-v1", "shaped-v2"], default="shaped-v2")
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--team-size", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=.999)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--clip", type=float, default=.15)
    parser.add_argument("--entropy", type=float, default=.01)
    parser.add_argument("--target-kl", type=float, default=.03)
    parser.add_argument("--historical-probability", type=float, default=.5)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=1000000)
    parser.add_argument("--seed-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=829471)
    parser.add_argument("--save-minutes", type=float, default=2)
    parser.add_argument("--snapshot-minutes", type=float, default=10)
    parser.add_argument("--benchmark-minutes", type=float, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--node", default="node")
    parser.add_argument("--max-updates", type=int, default=0, help="Smoke testing only; 0 runs indefinitely.")
    parser.add_argument("--no-benchmarks", action="store_true", help="Smoke testing only.")
    args = parser.parse_args(argv)
    finite = (args.learning_rate, args.gamma, args.gae_lambda, args.clip, args.entropy,
              args.target_kl, args.historical_probability, args.save_minutes,
              args.snapshot_minutes, args.benchmark_minutes)
    if not all(math.isfinite(value) for value in finite):
        raise ValueError("PPO parameters and intervals must be finite.")
    if not 0 <= args.seed <= 4294967295:
        raise ValueError("Random seed must be an unsigned 32-bit integer.")
    if not args.name.strip() or len(args.name) > 80 or any(ord(character) < 32 for character in args.name):
        raise ValueError("Model name must contain 1–80 printable characters.")
    if not 1 <= args.environments <= 64 or not 1 <= args.team_size <= 8 or not 2 <= args.rollout_steps <= 1024:
        raise ValueError("Invalid environment sizes.")
    if not 1 <= args.minibatch <= 8192 or not 1 <= args.epochs <= 20 or not 2 <= args.pool_size <= 32:
        raise ValueError("Invalid optimizer/pool sizes.")
    if not 0 <= args.historical_probability <= 1 or args.max_updates < 0 or args.critic_warmup_updates < 0:
        raise ValueError("Invalid historical probability or update limit.")
    if not 0 <= args.seed_start < args.seed_start + args.seed_count <= 4294967296 or not 1 <= args.seed_count <= 10000:
        raise ValueError("Invalid training seed range.")
    if args.seed_start < 600005 and args.seed_start + args.seed_count > 600000:
        raise ValueError("Benchmark seeds cannot be used for training.")
    if min(args.save_minutes, args.snapshot_minutes, args.benchmark_minutes) <= 0:
        raise ValueError("Intervals must be positive.")
    if not 0 < args.learning_rate <= .01 or not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1 or not 0 < args.clip < 1 or args.entropy < 0 or args.target_kl <= 0:
        raise ValueError("Invalid PPO parameters.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.initial:
        initial = Path(args.initial)
        initial = (ROOT / initial if not initial.is_absolute() else initial).resolve()
        output = Path(args.output)
        output = (ROOT / output if not output.is_absolute() else output).resolve()
        if initial == output:
            raise ValueError("Initial policy and live output must be different files.")
    directory = Path(args.run_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    # A second direct trainer must not share checkpoints with the first.
    try:
        from .selfplay_control import process_lock
    except ImportError:
        from selfplay_control import process_lock
    with process_lock(directory / "trainer.lock"):
        train(args, directory)


def train(args, directory):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    config, digest = training_config(args), source_hash()
    saved = read_state(directory, config, digest)
    identity = {"runId": saved["runId"] if saved else str(uuid.uuid4()),
                "createdAt": saved["createdAt"] if saved else utc(), "sourceSha256": digest}
    if saved and saved.get("rewardSpec"):
        identity["rewardSpec"] = saved["rewardSpec"]
    identity["rewardHistory"] = saved.get("rewardHistory", []) if saved else []
    if saved:
        identity.update({key: saved.get(key) for key in ("origin", "initialModel", "initialModelSha256", "initialTraining")})
    else:
        identity.update(initialize_actor(model, args, config))
    counters = saved["counters"] if saved else {"updates": 0, "actorSteps": 0, "learnerSteps": 0, "completedMatches": 0, "wallSeconds": 0.0}
    counters.setdefault("rewardComponentTotals", {})
    opponents = Opponents(device, rng, args.pool_size, args.historical_probability)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for item in saved["opponents"]:
            opponents.add(item["weights"], item["update"])
        rng.setstate(saved["routingRandomState"])
        torch.set_rng_state(saved["torchRandomState"])
        if saved["cudaRandomStates"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cudaRandomStates"])
    else:
        opponents.add(cpu_state(model), 0)
    atomic_json(directory / "run.json", {**identity, "config": config,
        "benchmarkMinutes": args.benchmark_minutes, "saveMinutes": args.save_minutes,
        "snapshotMinutes": args.snapshot_minutes, "device": str(device), "output": args.output,
        "trainingOpponents": "only current and historical policies from this run", "externalBenchmarksAffectTraining": False})
    environment, benchmarks = None, Benchmarks(args, directory)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    recent = deque(maxlen=100)
    session_start, session_steps = time.perf_counter(), 0
    last_saved, last_snapshot = time.monotonic(), time.monotonic()
    initial_updates, initial_wall = counters["updates"], counters["wallSeconds"]
    metrics, status = {}, {}
    try:
        # Establish a recoverable origin before launching workers or checking export.
        save_state(directory, model, optimizer, opponents, rng, config, identity, counters)
        environment = Environments(args, directory)
        obs = environment.first
        if obs.get("rewardProfile") != args.reward_profile or not isinstance(obs.get("rewardSpec"), dict):
            raise ValueError("Environment did not confirm the requested reward profile.")
        if identity.get("rewardSpec") and identity["rewardSpec"] != obs["rewardSpec"]:
            raise ValueError("Resume rejected: reward specification changed.")
        identity["rewardSpec"] = obs["rewardSpec"]
        # Save the worker-confirmed reward contract in run provenance and checkpoints.
        run_metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        atomic_json(directory / "run.json", {**run_metadata, "rewardSpec": identity["rewardSpec"]})
        save_state(directory, model, optimizer, opponents, rng, config, identity, counters)
        count, horizon = obs["count"], args.rollout_steps
        output, parity = publish(model, args, identity, counters, obs, directory)
        log_event(directory, {"type": "resumed" if saved else "initialized", **identity, **counters, "parity": parity})
        if not args.no_benchmarks:
            benchmarks.start(output, counters)
        while not stop[0] and not (directory / "stop.request").exists():
            critic_warmup = set_training_phase(model, args, counters["updates"])
            pixels = np.empty((horizon, count, 36, 64, 3), dtype=np.uint8)
            scalars = np.empty((horizon, count, 6), dtype=np.float32)
            actions = np.empty((horizon, count, 6), dtype=np.int64)
            oldlog = np.empty((horizon, count), dtype=np.float32)
            values, rewards, dones, alive, learners = [np.empty_like(oldlog) for _ in range(5)]
            component_sums = {}
            model.eval()
            for step in range(horizon):
                pixels[step], scalars[step], alive[step] = obs["pixels"], obs["scalars"], obs["alive"]
                learner, frozen = opponents.route(obs)
                learners[step] = learner
                px = torch.from_numpy(obs["pixels"]).to(device)
                own = torch.from_numpy(obs["scalars"]).to(device)
                with torch.no_grad():
                    action, logprob, _, value = model.act(px, own)
                    for indices, policy in frozen:
                        index = torch.as_tensor(indices, device=device)
                        action[index] = policy.act(px[index], own[index])[0]
                actions[step], oldlog[step], values[step] = action.cpu().numpy(), logprob.cpu().numpy(), value.cpu().numpy()
                obs = environment.request({"command": "step", "actions": actions[step].reshape(-1).tolist()})
                rewards[step], dones[step] = obs["rewards"], obs["done"]
                for name, components in obs.get("rewardComponents", {}).items():
                    amount = float(np.asarray(components, dtype=np.float64).dot(learner))
                    component_sums[name] = component_sums.get(name, 0.0) + amount
                recent.extend(row for row in obs["episodes"] if row["complete"])
                counters["completedMatches"] += sum(bool(row["complete"]) for row in obs["episodes"])
                counters["actorSteps"] += count
                counters["learnerSteps"] += int((learner * alive[step]).sum())
                session_steps += count
            with torch.no_grad():
                final = model(torch.from_numpy(obs["pixels"]).to(device), torch.from_numpy(obs["scalars"]).to(device))[1].cpu().numpy()
            advantage, returns = advantages(rewards, values, dones, final, args.gamma, args.gae_lambda)
            size = horizon * count
            arrays = (pixels, scalars, actions, oldlog, values, advantage, returns, alive * learners, learners)
            tensors = [torch.from_numpy(array.reshape((size,) + array.shape[2:])).to(device) for array in arrays]
            px, own, act, log_old, value_old, advantage, target, actor_mask, value_mask = tensors
            active_adv = advantage[actor_mask > 0]
            advantage = (advantage - active_adv.mean()) / (active_adv.std(unbiased=False) + 1e-8) if active_adv.numel() else advantage * 0
            accumulated, early_stop = [], False
            model.train()
            for _ in range(args.epochs):
                for index in torch.randperm(size, device=device).split(args.minibatch):
                    _, logprob, entropy, value = model.act(px[index], own[index], act[index])
                    ratio = (logprob - log_old[index]).exp()
                    importance, value_importance = actor_mask[index], value_mask[index]
                    denominator = importance.sum().clamp_min(1)
                    actor_loss = (torch.maximum(-advantage[index] * ratio, -advantage[index] * ratio.clamp(1 - args.clip, 1 + args.clip)) * importance).sum() / denominator
                    clipped_value = value_old[index] + (value - value_old[index]).clamp(-.3, .3)
                    value_loss = .5 * (torch.maximum((value - target[index]).square(), (clipped_value - target[index]).square()) * value_importance).sum() / value_importance.sum().clamp_min(1)
                    entropy_mean = (entropy * importance).sum() / denominator
                    loss = .5 * value_loss if critic_warmup else actor_loss + .5 * value_loss - args.entropy * entropy_mean
                    if not torch.isfinite(loss):
                        raise RuntimeError("PPO loss became non-finite; last saved checkpoint retained.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), .5)
                    optimizer.step()
                    with torch.no_grad():
                        kl = (((ratio - 1) - (logprob - log_old[index])) * importance).sum() / denominator
                    accumulated.append([actor_loss.item(), value_loss.item(), entropy_mean.item(), kl.item()])
                    if not critic_warmup and kl > args.target_kl:
                        early_stop = True
                        break
                if early_stop:
                    break
            mean = np.asarray(accumulated).mean(0)
            counters["updates"] += 1
            counters["wallSeconds"] = initial_wall + time.perf_counter() - session_start
            for name, amount in component_sums.items():
                counters["rewardComponentTotals"][name] = counters["rewardComponentTotals"].get(name, 0.0) + amount
            metrics = {"type": "update", **counters, "actorStepsPerSecond": session_steps / (time.perf_counter() - session_start),
                "phase": "critic-warmup" if critic_warmup else "self-play-ppo",
                "policyLoss": float(mean[0]), "valueLoss": float(mean[1]), "entropy": float(mean[2]), "kl": float(mean[3]),
                "nonzeroRewardFraction": float(np.count_nonzero(rewards) / rewards.size),
                "learnerNonzeroRewardFraction": float(np.count_nonzero(rewards * learners) / max(1, learners.sum())),
                "rewardComponentSums": component_sums,
                "recentMatches": len(recent), "recentDraws": sum(row["winner"] == "draw" for row in recent),
                "recentKills": sum(sum(row["scores"]) for row in recent),
                "recentHits": sum(team["hits"] for row in recent for team in row["teams"]),
                "recentShots": sum(team["shots"] for row in recent for team in row["teams"]),
                "poolUpdates": [item[0] for item in opponents.pool]}
            log_event(directory, metrics)
            benchmarks.poll()
            now = time.monotonic()
            if now - last_snapshot >= args.snapshot_minutes * 60:
                opponents.add(cpu_state(model), counters["updates"])
                last_snapshot = now
            benchmark_due = not args.no_benchmarks and benchmarks.due()
            if now - last_saved >= args.save_minutes * 60 or benchmark_due:
                save_state(directory, model, optimizer, opponents, rng, config, identity, counters)
                output, parity = publish(model, args, identity, counters, obs, directory)
                last_saved = now
                log_event(directory, {"type": "checkpoint", **counters, "parity": parity})
            if benchmark_due:
                benchmarks.start(output, counters)
            status = {"state": "running", "pid": os.getpid(), "heartbeat": utc(), **identity, **metrics,
                      "checkpointAgeSeconds": now - last_saved, "model": str(output),
                      "benchmarkMinutes": args.benchmark_minutes, "benchmark": benchmarks.job if benchmarks.process else benchmarks.latest}
            atomic_json(directory / "status.json", status)
            if args.max_updates and counters["updates"] - initial_updates >= args.max_updates:
                break
        save_state(directory, model, optimizer, opponents, rng, config, identity, counters)
        output, parity = publish(model, args, identity, counters, obs, directory)
        atomic_json(directory / "status.json", {**status, "state": "stopped", "heartbeat": utc(), **counters, "parity": parity})
        log_event(directory, {"type": "stopped", **counters})
    except Exception as error:
        atomic_json(directory / "status.json", {**status, "state": "failed", "heartbeat": utc(), "error": str(error), **counters})
        raise
    finally:
        benchmarks.close()
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
