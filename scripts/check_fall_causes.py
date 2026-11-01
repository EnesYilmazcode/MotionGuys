"""At each fall in recorded rounds: was the body really over a hole, and what was it doing just before?

Run:  python scripts/check_fall_causes.py runs/real_base
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    for folder in sys.argv[1:]:
        kinds, over, nearest_all, since_leap, since_up = Counter(), Counter(), [], [], []
        for path in sorted(Path(folder).glob("seed_*.json")):
            meta = json.loads(path.read_text())
            runners = len(meta["runners"])
            hips = np.fromfile(path.with_suffix(".bin"), dtype=np.float32, count=meta["frames"] * runners * 3).reshape(meta["frames"], runners, 3)
            radius = meta["arena"]["hexRadius"]
            width = math.sqrt(3) * radius
            center = lambda q, r: (width * (q + r / 2), 1.5 * radius * r)
            floors = [{tuple(t) for t in floor} for floor in meta["arena"]["tiles"]]
            gone = [set() for _ in floors]
            last_land, last_up = {}, {}
            for event in meta["events"]:
                who = event.get("runner")
                if event["type"] == "tile_gone":
                    gone[event["layer"]].add((event["q"], event["r"]))
                elif event["type"] == "leap_landed":
                    last_land[who] = event["t"]
                elif event["type"] == "recover":
                    last_up[who] = event["t"]
                elif event["type"] == "fall":
                    kinds[event["cause"]] += 1
                    frame = min(int(event["t"] * meta["fps"]), meta["frames"] - 1)
                    x, y = hips[frame, who, :2]
                    layer = event["layer"]
                    solid = floors[layer] - gone[layer]
                    distance = min((math.hypot(cx - x, cy - y) for cx, cy in (center(*t) for t in solid)), default=99.0)
                    nearest_all.append(distance)
                    # Inside a tile's inner circle means the hips were plainly over floor.
                    over["hips over floor" if distance < radius * 0.8 else "edge" if distance < radius * 1.2 else "clear hole"] += 1
                    since_leap.append(event["t"] - last_land.get(who, -99))
                    since_up.append(event["t"] - last_up.get(who, -99))
        total = sum(kinds.values())
        print(folder, total, "falls", dict(kinds))
        print("  hips at the moment of the fall:", {k: f"{100 * v / total:.0f}%" for k, v in over.items()})
        print("  distance to nearest solid tile centre p10/50/90:", np.percentile(nearest_all, [10, 50, 90]).round(2))
        print(f"  within 1 s of a leap landing: {100 * np.mean(np.array(since_leap) < 1):.0f}%, within 1 s of getting up: {100 * np.mean(np.array(since_up) < 1):.0f}%, within 2 s of getting up: {100 * np.mean(np.array(since_up) < 2):.0f}%")


if __name__ == "__main__":
    main()
