"""Start, inspect, or gracefully stop the detached self-play training service."""
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def process_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                stream.write(b"0")
                stream.flush()
                stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError("This run already has an active process.") from error
    try:
        yield
    finally:
        stream.close()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def active(directory):
    try:
        with process_lock(directory / "service.lock"):
            return False
    except RuntimeError:
        return True


def supervise(directory, arguments):
    with process_lock(directory / "service.lock"):
        failures = 0
        while not (directory / "stop.request").exists():
            path = directory / "service.log"
            if path.exists() and path.stat().st_size > 20 * 1024 * 1024:
                os.replace(path, directory / "service.previous.log")
            with path.open("a", encoding="utf-8") as initial_log:
                started = time.time()
                process = subprocess.Popen([sys.executable, "-u", str(ROOT / "training/selfplay.py"),
                    "--run-dir", str(directory), *arguments], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                (directory / "service.json").write_text(json.dumps({"supervisorPid": os.getpid(),
                    "trainerPid": process.pid, "startedAt": datetime.now(timezone.utc).isoformat(), "arguments": arguments}), encoding="utf-8")
            for line in process.stdout:
                if path.exists() and path.stat().st_size > 20 * 1024 * 1024:
                    os.replace(path, directory / "service.previous.log")
                with path.open("a", encoding="utf-8") as log:
                    log.write(line)
            code = process.wait()
            with path.open("a", encoding="utf-8") as log:
                log.write(f"\nTrainer exited {code} at {datetime.now(timezone.utc).isoformat()}\n")
            if code == 0 or (directory / "stop.request").exists():
                return
            failures = failures + 1 if time.time() - started < 120 else 1
            deadline = time.time() + min(300, 15 * 2 ** min(failures - 1, 5))
            while time.time() < deadline and not (directory / "stop.request").exists():
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "stop", "status", "supervise"])
    parser.add_argument("--run-dir", default="training/runs/teacher-selfplay")
    args, extra = parser.parse_known_args()
    directory = Path(args.run_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.command == "supervise":
        supervise(directory, extra)
    elif args.command == "status":
        print(json.dumps({"serviceActive": active(directory), "stopRequested": (directory / "stop.request").exists(),
            "service": read_json(directory / "service.json"), "training": read_json(directory / "status.json"),
            "benchmark": read_json(directory / "benchmark-status.json")}, indent=2))
    elif args.command == "stop":
        (directory / "stop.request").write_text("Stop after the current PPO update and save.\n", encoding="utf-8")
        print("Stop requested. The trainer saves at the next update boundary; check status for stopped.")
    else:
        if active(directory):
            print("Self-play is already running.")
            return
        # Preserve exact previous options on resume, unless explicitly supplied.
        previous = read_json(directory / "service.json")
        if not extra and previous:
            extra = previous["arguments"]
        (directory / "stop.request").unlink(missing_ok=True)
        log = (directory / "supervisor.log").open("a", encoding="utf-8")
        try:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "supervise",
                "--run-dir", str(directory), "--", *extra], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                start_new_session=os.name != "nt")
        finally:
            log.close()
        print(json.dumps({"supervisorPid": process.pid, "runDirectory": str(directory), "state": "starting"}))


if __name__ == "__main__":
    main()
