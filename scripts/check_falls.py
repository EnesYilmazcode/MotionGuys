"""At each fall in the recorded rounds, was an untouched tile close enough to have been worth going for?

Run:  python scripts/check_falls.py
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("public/hex")
GAITS = ("slow", "walk", "jog", "sprint")
# One tile over, then roughly a leap's reach.
REACHES = (2.1, 3.0)


def main():
    falls = 0
    gaits = Counter()
    ahead = Counter()
    for path in sorted(ROOT.glob("seed_*.json")):
        meta = json.loads(path.read_text())
        count = meta["frames"] * len(meta["runners"]) * 3
        hips = np.fromfile(path.with_suffix(".bin"), dtype=np.float32, count=count).reshape(meta["frames"], len(meta["runners"]), 3)
        radius = meta["arena"]["hexRadius"]
        centers = [{(q, r): (math.sqrt(3) * radius * (q + r / 2), 1.5 * radius * r) for q, r in floor} for floor in meta["arena"]["tiles"]]
        touched = set()
        gait = {}
        for event in meta["events"]:
            if event["type"] == "tile_touch":
                touched.add((event["layer"], event["q"], event["r"]))
            elif event["type"] == "gait":
                gait[event["runner"]] = event["gait"]
            elif event["type"] == "fall" and event["cause"] in ("hole", "edge"):
                falls += 1
                gaits[GAITS[gait.get(event["runner"], 1)]] += 1
                frame = min(int(event["t"] * meta["fps"]), meta["frames"] - 1)
                here = hips[frame, event["runner"], :2]
                travel = here - hips[max(frame - 6, 0), event["runner"], :2]
                heading = travel / max(np.linalg.norm(travel), 1e-6)
                nearest = math.inf
                for (q, r), center in centers[event["layer"]].items():
                    if (event["layer"], q, r) in touched:
                        continue
                    offset = np.array(center) - here
                    distance = float(np.linalg.norm(offset))
                    # Within 60 degrees of the way it was already going.
                    if distance > 0.01 and float(offset @ heading) / distance > 0.5:
                        nearest = min(nearest, distance)
                for reach in REACHES:
                    ahead[reach] += nearest <= reach
    print(f"{falls} falls from a hole or an edge")
    for name in GAITS:
        print(f"  while told to {name}: {100 * gaits[name] / falls:.0f}%")
    for reach in REACHES:
        print(f"  untouched tile ahead within {reach} m: {100 * ahead[reach] / falls:.0f}%")


if __name__ == "__main__":
    main()
