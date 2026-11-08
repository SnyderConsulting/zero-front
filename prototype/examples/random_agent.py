"""Exercise FIELDLAB's local headless API using only Python's standard library.

Start the server with `node server.mjs`, then run this file from the project root.
This is an integration example, not a trained or competent policy.
"""

import argparse
import base64
import json
import random
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:4173")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--team-size", type=int, default=4, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--score-limit", type=int, default=25, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--snapshot", type=Path, help="Optional final image destination, e.g. frame.ppm")
    args = parser.parse_args()
    policy_rng = random.Random(args.seed)

    def request(endpoint, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(args.url.rstrip("/") + endpoint, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.load(response)

    state = request("/api/reset", {
        "seed": args.seed, "teamSize": args.team_size, "scoreLimit": args.score_limit,
        "resolution": [320, 180],
    })
    total_reward = 0.0
    for index in range(args.steps):
        state = request("/api/step", {
            "action": {
                "forward": policy_rng.choice([0, 0.5, 1]),
                "strafe": policy_rng.choice([-0.5, 0, 0.5]),
                "turn": policy_rng.uniform(-1.2, 1.2),
                "fire": True,
                "reload": state["observation"]["ammo"] == 0,
            },
            "ticks": 6,
        })
        total_reward += state["reward"]
        if index % 20 == 0 or state["terminated"] or state["truncated"]:
            obs = state["observation"]
            print(f"step={index + 1:3d} time={obs['time']:.1f}s health={obs['health']} "
                  f"ammo={obs['ammo']} score={obs['scores'][0]}-{obs['scores'][1]} "
                  f"kills={obs['kills']} deaths={obs['deaths']} reward={total_reward:.2f}")
        if state["terminated"] or state["truncated"]:
            break

    if args.snapshot:
        pixels = request("/api/frame")
        rgb = base64.b64decode(pixels["data"], validate=True)
        assert len(rgb) == pixels["width"] * pixels["height"] * 3
        header = f"P6\n{pixels['width']} {pixels['height']}\n255\n".encode("ascii")
        args.snapshot.write_bytes(header + rgb)
        print(f"Saved RGB image to {args.snapshot.resolve()}")
    print("Episode result:", json.dumps(state["observation"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as error:
        raise SystemExit(f"Cannot reach the local agent API: {error}. Start `node server.mjs` first.")
