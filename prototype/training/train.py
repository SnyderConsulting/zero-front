"""Train a real first-person pixel imitation policy and export data-only game weights."""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
HEADS = [("forward", 3), ("strafe", 3), ("turn", 5), ("pitch", 3), ("fire", 2), ("reload", 2)]
SCALARS = ["health", "ammo", "reloading", "spawnProtection", "pitch", "alive"]
TENSORS = {
    "conv1": ("conv1.weight", [8, 3, 5, 5]), "conv1Bias": ("conv1.bias", [8]),
    "conv2": ("conv2.weight", [16, 8, 3, 3]), "conv2Bias": ("conv2.bias", [16]),
    "fc": ("fc.weight", [64, 1574]), "fcBias": ("fc.bias", [64]),
    "policy": ("policy.weight", [18, 64]), "policyBias": ("policy.bias", [18]),
}


class PixelPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.fc = nn.Linear(1574, 64)
        self.policy = nn.Linear(64, 18)

    def forward(self, pixels, scalars):
        image = pixels.permute(0, 3, 1, 2).contiguous().float() / 255.0
        visual = torch.relu(self.conv2(torch.relu(self.conv1(image)))).flatten(1)
        return self.policy(torch.relu(self.fc(torch.cat((visual, scalars), dim=1))))


def seed_list(value):
    seeds = []
    try:
        for item in value.split(","):
            if ":" in item:
                start, end = map(int, item.split(":"))
                if end <= start or end - start > 128:
                    raise ValueError()
                seeds.extend(range(start, end))
            else:
                seeds.append(int(item))
        seeds = list(dict.fromkeys(seeds))
        if not seeds or len(seeds) > 128 or any(seed < 0 or seed > 4294967295 for seed in seeds):
            raise ValueError()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use up to 128 seeds, e.g. 1000:1020 or 42,51,68; range end is exclusive.") from error
    return seeds


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def collect(seeds, args, split):
    command = {"command": "collect", "seeds": seeds, "teamSize": args.team_size,
               "seconds": args.seconds, "actorsPerFrame": args.actors_per_frame,
               "maxSamples": args.max_samples if split == "train" else args.max_validation_samples}
    process = subprocess.Popen([args.node, str(ROOT / "training/bridge.mjs")], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    pixel_parts, scalar_parts, label_parts = [], [], []
    try:
        process.stdin.write(json.dumps(command) + "\n")
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Collection bridge stopped: " + process.stderr.read())
            packet = json.loads(line)
            if packet["type"] == "error":
                raise RuntimeError(packet["error"])
            if packet["type"] == "samples":
                count = packet["count"]
                pixels = np.frombuffer(base64.b64decode(packet["pixels"], validate=True), dtype=np.uint8)
                if pixels.size != count * 36 * 64 * 3:
                    raise RuntimeError("Collection bridge returned invalid pixels.")
                pixel_parts.append(pixels.reshape(count, 36, 64, 3))
                scalar_parts.append(np.asarray(packet["scalars"], dtype=np.float32).reshape(count, 6))
                label_parts.append(np.asarray(packet["labels"], dtype=np.int64).reshape(count, 6))
            if packet["type"] == "progress":
                print(f"Collect {split}: seed {packet['seed']}, {packet['samples']} samples ({packet['total']} total)", flush=True)
            if packet["type"] == "complete":
                break
        process.stdin.write('{"command":"close"}\n')
        process.stdin.flush()
        process.stdin.close()
        process.wait(timeout=10)
        if process.returncode:
            raise RuntimeError("Collection bridge failed: " + process.stderr.read())
        if not pixel_parts:
            raise RuntimeError("No demonstration samples collected.")
        return (np.concatenate(pixel_parts), np.concatenate(scalar_parts), np.concatenate(label_parts))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def validate_dataset(data):
    pixels, scalars, labels = data
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (36, 64, 3) or not len(pixels):
        raise ValueError("Dataset pixel shape is invalid or empty.")
    if scalars.shape != (len(pixels), 6) or labels.shape != (len(pixels), 6) or not np.isfinite(scalars).all():
        raise ValueError("Dataset status or action shape is invalid.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Dataset action labels must be integers.")
    for index, (_, size) in enumerate(HEADS):
        if np.any(labels[:, index] < 0) or np.any(labels[:, index] >= size):
            raise ValueError("Dataset action labels are invalid.")


def load_dataset(args):
    source = hashlib.sha256()
    for filename in ("dist/simulation.js", "dist/renderer.js", "dist/mapgen.js", "dist/policy.js", "training/bridge.mjs"):
        source.update((ROOT / filename).read_bytes())
    config = {"trainSeeds": args.train_seeds, "validationSeeds": args.validation_seeds,
              "teamSize": args.team_size, "seconds": args.seconds, "actorsPerFrame": args.actors_per_frame,
              "maxSamples": args.max_samples, "maxValidationSamples": args.max_validation_samples,
              "sourceDigest": source.hexdigest()}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    cache = ROOT / "training/runs" / f"dataset-{digest}.npz"
    if cache.exists() and not args.no_cache:
        print(f"Loading cached demonstrations: {cache}", flush=True)
        with np.load(cache, allow_pickle=False) as saved:
            train = tuple(saved[f"train_{key}"] for key in ("pixels", "scalars", "labels"))
            valid = tuple(saved[f"valid_{key}"] for key in ("pixels", "scalars", "labels"))
    else:
        train = collect(args.train_seeds, args, "train")
        valid = collect(args.validation_seeds, args, "validation")
        if not args.no_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **{f"{split}_{key}": item for split, data in (("train", train), ("valid", valid))
                                         for key, item in zip(("pixels", "scalars", "labels"), data)})
    for data in (train, valid):
        validate_dataset(data)
    return train, valid, config


def known_training_seeds(training):
    if not isinstance(training, dict):
        raise ValueError("Training metadata must be an object.")
    result = set()
    for key in ("trainSeeds", "priorTrainingSeeds"):
        seeds = training.get(key, [])
        if not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 or seed > 4294967295 for seed in seeds):
            raise ValueError(f"Invalid {key} provenance.")
        result.update(seeds)
    return result


def import_weights(model, path, validation_seeds):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != "arena-pixel-policy" or raw.get("version") != 1 or raw.get("architecture") != "cnn-v1":
        raise ValueError("Resume requires an Arena pixel policy v1 JSON model.")
    if raw.get("observation") != {"kind": "semantic-rgb", "width": 64, "height": 36, "scalars": SCALARS}:
        raise ValueError("Resume observation contract differs.")
    if raw.get("actionSpace") != "discrete-v1" or raw.get("actionRepeat") != 6:
        raise ValueError("Resume action contract differs.")
    prior_seeds = known_training_seeds(raw.get("training", {}))
    if prior_seeds.intersection(validation_seeds):
        raise ValueError("Validation seeds overlap seeds used to train the resumed model.")
    state = {}
    for name, (key, shape) in TENSORS.items():
        tensor = raw.get("weights", {}).get(name, {})
        data = np.asarray(tensor.get("data", []), dtype=np.float32)
        if tensor.get("shape") != shape or data.size != math.prod(shape) or not np.isfinite(data).all() or np.abs(data).max() > 100:
            raise ValueError(f"Invalid resume tensor: {name}")
        state[key] = torch.from_numpy(data.reshape(shape))
    model.load_state_dict(state, strict=True)
    return raw


def export_model(model, args, metadata):
    state = model.state_dict()
    weights = {name: {"shape": shape, "data": state[key].detach().cpu().float().flatten().tolist()}
               for name, (key, shape) in TENSORS.items()}
    return {"format": "arena-pixel-policy", "version": 1, "name": args.name,
            "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "architecture": "cnn-v1",
            "observation": {"kind": "semantic-rgb", "width": 64, "height": 36, "scalars": SCALARS},
            "actionSpace": "discrete-v1", "actionRepeat": 6, "weights": weights, "training": metadata}


def make_losses(labels, device):
    losses, counts = [], {}
    for index, (name, size) in enumerate(HEADS):
        count = np.bincount(labels[:, index], minlength=size)
        # Mild reweighting prevents the common "don't fire" class dominating learning.
        weight = np.sqrt(len(labels) / np.maximum(count, 1))
        weight = np.minimum(weight / weight.mean(), 4)
        losses.append(nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=device)))
        counts[name] = count.tolist()
    return losses, counts


def batch_metrics(logits, targets, losses):
    offset, total, correct = 0, logits.sum() * 0, []
    for index, (_, size) in enumerate(HEADS):
        head = logits[:, offset:offset + size]
        total = total + losses[index](head, targets[:, index]) / len(HEADS)
        correct.append((head.argmax(1) == targets[:, index]).sum().item())
        offset += size
    return total, correct


@torch.no_grad()
def evaluate(model, data, losses, batch_size, device):
    model.eval()
    total, correct = 0.0, np.zeros(6, dtype=np.float64)
    pixels, scalars, labels = data
    for start in range(0, len(pixels), batch_size):
        selection = slice(start, start + batch_size)
        x = torch.from_numpy(pixels[selection]).to(device)
        own = torch.from_numpy(scalars[selection]).to(device)
        target = torch.from_numpy(labels[selection]).to(device)
        loss, hits = batch_metrics(model(x, own), target, losses)
        total += loss.item() * len(x)
        correct += hits
    accuracy = {name: float(correct[index] / len(pixels)) for index, (name, _) in enumerate(HEADS)}
    return {"loss": total / len(pixels), "meanActionAccuracy": float(correct.mean() / len(pixels)), "headAccuracy": accuracy}


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--name", default="Pixel baseline")
    command.add_argument("--output", default="models/pixel-baseline.json")
    command.add_argument("--resume", help="Warm-start from an exported JSON model; starts a fresh optimizer.")
    command.add_argument("--epochs", type=int, default=12)
    command.add_argument("--patience", type=int, default=0, help="Stop after this many epochs without material validation improvement; 0 disables.")
    command.add_argument("--min-delta", type=float, default=.0001, help="Minimum absolute validation-loss improvement for patience and LR scheduling.")
    command.add_argument("--lr-patience", type=int, default=0, help="Halve learning rate after a validation plateau; 0 disables.")
    command.add_argument("--batch-size", type=int, default=256)
    command.add_argument("--learning-rate", type=float, default=.001)
    command.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    command.add_argument("--seed", type=int, default=2026, help="Network initialization and minibatch seed.")
    command.add_argument("--train-seeds", type=seed_list, default=seed_list("1000:1020"))
    command.add_argument("--validation-seeds", type=seed_list, default=seed_list("2000:2005"))
    command.add_argument("--seconds", type=int, default=45, help="Simulation seconds per demonstration map.")
    command.add_argument("--team-size", type=int, default=4)
    command.add_argument("--actors-per-frame", type=int, default=4)
    command.add_argument("--max-samples", type=int, default=36000)
    command.add_argument("--max-validation-samples", type=int, default=9000)
    command.add_argument("--no-cache", action="store_true")
    command.add_argument("--node", default="node")
    command.add_argument("--parity-output", default="training/runs/pixel-parity.json")
    return command


class ValidationPlateau:
    """Track material improvement independently from selection of the absolute best model."""
    def __init__(self, initial, patience, min_delta):
        self.best, self.patience, self.min_delta, self.stale = initial, patience, min_delta, 0

    def update(self, loss):
        if not math.isfinite(loss):
            raise ValueError("Validation loss became non-finite.")
        if loss < self.best - self.min_delta:
            self.best, self.stale = loss, 0
        else:
            self.stale += 1
        return self.patience > 0 and self.stale >= self.patience


def main():
    args = parser().parse_args()
    if set(args.train_seeds).intersection(args.validation_seeds):
        raise ValueError("Training and validation seeds must be disjoint.")
    if not 1 <= args.epochs <= 1000 or not 1 <= args.batch_size <= 4096 or not 0 < args.learning_rate <= .1:
        raise ValueError("Invalid epoch count, batch size or learning rate.")
    if not 0 <= args.patience <= 1000 or not 0 <= args.lr_patience <= 1000 or not math.isfinite(args.min_delta) or args.min_delta < 0:
        raise ValueError("Invalid validation plateau settings.")
    if not 2 <= args.team_size <= 8 or not 1 <= args.actors_per_frame <= args.team_size * 2 - 1 or not 1 <= args.seconds <= 180:
        raise ValueError("Invalid collection settings.")
    if not len(args.train_seeds) <= args.max_samples <= 1000000 or not len(args.validation_seeds) <= args.max_validation_samples <= 1000000:
        raise ValueError("Sample limits must cover every requested seed and stay below one million.")
    if not args.name.strip() or len(args.name) > 80 or any(ord(char) < 32 for char in args.name):
        raise ValueError("Model name must contain 1–80 printable characters.")
    random.seed(args.seed)
    np.random.seed(args.seed & 0xffffffff)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    print(f"Training device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    start = time.perf_counter()
    model = PixelPolicy().to(device)
    resumed = import_weights(model, resolve(args.resume), args.validation_seeds) if args.resume else None
    previous_seeds = known_training_seeds(resumed.get("training", {})) if resumed else set()
    if len(previous_seeds.union(args.train_seeds)) > 128:
        raise ValueError("A model lineage may contain at most 128 distinct training seeds in format v1.")
    train, valid, dataset_config = load_dataset(args)
    previous_seeds.update(dataset_config.get("priorTrainingSeeds", []))
    if previous_seeds.intersection(args.validation_seeds) or len(previous_seeds.union(args.train_seeds)) > 128:
        raise ValueError("Dataset ancestry overlaps validation or exceeds 128 training seeds.")
    print(f"Demonstrations: {len(train[0])} training / {len(valid[0])} held-out validation", flush=True)
    losses, counts = make_losses(train[2], device)
    initial = evaluate(model, valid, losses, args.batch_size, device)
    print("Initial held-out metrics: " + json.dumps(initial), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=args.lr_patience,
        threshold=args.min_delta, threshold_mode="abs", min_lr=args.learning_rate * .05) if args.lr_patience else None
    plateau = ValidationPlateau(initial["loss"], args.patience, args.min_delta)
    best, best_epoch, best_state = initial, 0, copy.deepcopy(model.state_dict())
    completed_epochs, stopped_early = 0, False
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".training.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["epoch", "train_loss", "validation_loss", "mean_action_accuracy", *[name + "_accuracy" for name, _ in HEADS], "learning_rate"])
        generator = torch.Generator().manual_seed(args.seed)
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train[0]), generator=generator).numpy()
            training_loss = 0.0
            for start_index in range(0, len(order), args.batch_size):
                index = order[start_index:start_index + args.batch_size]
                pixels = torch.from_numpy(train[0][index]).to(device)
                scalars = torch.from_numpy(train[1][index]).to(device)
                targets = torch.from_numpy(train[2][index]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = batch_metrics(model(pixels, scalars), targets, losses)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss became non-finite; model was not exported.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                training_loss += loss.item() * len(index)
            metrics = evaluate(model, valid, losses, args.batch_size, device)
            training_loss /= len(order)
            completed_epochs = epoch
            writer.writerow([epoch, training_loss, metrics["loss"], metrics["meanActionAccuracy"], *metrics["headAccuracy"].values(), optimizer.param_groups[0]["lr"]])
            logfile.flush()
            print(f"Epoch {epoch:3d}/{args.epochs}: train loss {training_loss:.4f}, held-out loss {metrics['loss']:.4f}, action accuracy {metrics['meanActionAccuracy']:.1%}", flush=True)
            if metrics["loss"] < best["loss"]:
                best, best_epoch, best_state = metrics, epoch, copy.deepcopy(model.state_dict())
            if scheduler:
                scheduler.step(metrics["loss"])
            if plateau.update(metrics["loss"]):
                stopped_early = True
                print(f"Validation plateau: stopping after {epoch} epochs; exporting best epoch {best_epoch}.", flush=True)
                break
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    metadata = {"algorithm": "behavioral-cloning", "teacher": "scripted-bot-with-privileged-navigation",
                "status": "experimental-baseline", "epochs": completed_epochs, "requestedEpochs": args.epochs, "selectedEpoch": best_epoch,
                "earlyStopped": stopped_early, "patience": args.patience, "minDelta": args.min_delta,
                "lrPatience": args.lr_patience, "finalLearningRate": optimizer.param_groups[0]["lr"],
                "selection": "lowest held-out weighted cross-entropy", "seed": args.seed,
                "trainSeeds": args.train_seeds, "validationSeeds": args.validation_seeds,
                "samples": len(train[0]), "validationSamples": len(valid[0]), "teamSize": args.team_size,
                "secondsPerSeed": args.seconds, "actorsPerFrame": args.actors_per_frame,
                "learningRate": args.learning_rate, "batchSize": args.batch_size,
                "sourceDigest": dataset_config["sourceDigest"], "torchVersion": str(torch.__version__),
                "device": str(device), "elapsedSeconds": elapsed, "initialValidation": initial,
                "bestValidation": best, "actionClassCounts": counts,
                "limitations": "Pixel imitation baseline; no hidden map or enemy positions enter the model. Validation imitation accuracy does not establish gameplay competence."}
    if previous_seeds:
        metadata["priorTrainingSeeds"] = sorted(previous_seeds)
    if resumed:
        metadata["warmStartedFrom"] = resumed["name"]
        metadata["optimizerRestarted"] = True
    exported = export_model(model, args, metadata)
    # Write atomically so the desktop never sees a half-written model.
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(exported, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    # CPU float32 gives the closest reference to the browser runtime, independent of GPU kernels.
    model = model.to("cpu").eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(valid[0][:1]), torch.from_numpy(valid[1][:1]))[0].tolist()
    parity = {"pixels": base64.b64encode(valid[0][0].tobytes()).decode("ascii"),
              "scalars": valid[1][0].tolist(), "logits": logits}
    parity_path = resolve(args.parity_output)
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.write_text(json.dumps(parity, allow_nan=False), encoding="utf-8")
    parity_result = subprocess.run([args.node, str(ROOT / "training/check-parity.mjs"), "--model", str(temporary), "--fixture", str(parity_path)],
                                  cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if parity_result.returncode:
        raise RuntimeError("JavaScript parity verification failed: " + parity_result.stderr.strip())
    print(parity_result.stdout.strip(), flush=True)
    temporary.replace(output)
    print(json.dumps({"model": str(output), "metricsCsv": str(csv_path), "selectedEpoch": best_epoch,
                      "initialValidation": initial, "bestValidation": best, "elapsedSeconds": elapsed}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        sys.exit(1)
