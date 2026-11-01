"""Are leaps helping? When they happen, where, what the floor ahead really looked like, and what follows them.

Run:  python scripts/check_leap_value.py public/hex
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

LOOKS = (1.0, 1.5, 2.0, 2.5)


def hex_at(x, y, radius):
    fq = (x * math.sqrt(3) / 3 - y / 3) / radius
    fr = (y * 2 / 3) / radius
    fs = -fq - fr
    q, r, s = round(fq), round(fr), round(fs)
    dq, dr, ds = abs(q - fq), abs(r - fr), abs(s - fs)
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    return q, r


def main():
    for folder in sys.argv[1:]:
        early, total, centre, own_only, truly, landed, fell_soon, fell_2, rounds = 0, 0, 0, 0, 0, 0, 0, 0, 0
        times, radii = [], []
        for path in sorted(Path(folder).glob("seed_*.json")):
            rounds += 1
            meta = json.loads(path.read_text())
            count = len(meta["runners"])
            hips = np.fromfile(path.with_suffix(".bin"), dtype=np.float32, count=meta["frames"] * count * 3).reshape(meta["frames"], count, 3)
            radius = meta["arena"]["hexRadius"]
            tiles = [{tuple(t) for t in floor} for floor in meta["arena"]["tiles"]]
            gone = [set() for _ in tiles]
            touched = [set() for _ in tiles]
            events = meta["events"]
            start = next(e["t"] for e in events if e["type"] == "go")
            landings = []
            for event in events:
                who = event.get("runner")
                if event["type"] == "tile_touch":
                    touched[event["layer"]].add((event["q"], event["r"]))
                elif event["type"] == "tile_gone":
                    gone[event["layer"]].add((event["q"], event["r"]))
                elif event["type"] == "leap_landed":
                    landed += 1
                    landings.append((event["t"], who))
                elif event["type"] == "fall":
                    for t, runner in landings:
                        if runner == who and event["t"] - t < 1.0:
                            fell_soon += 1
                        if runner == who and event["t"] - t < 2.0:
                            fell_2 += 1
                elif event["type"] == "leap":
                    total += 1
                    t = event["t"] - start
                    times.append(t)
                    early += t < 8
                    frame = min(int(event["t"] * meta["fps"]), meta["frames"] - 1)
                    x, y = hips[frame, who, :2]
                    radii.append(math.hypot(x, y))
                    centre += math.hypot(x, y) < 5
                    travel = hips[min(frame + 3, meta["frames"] - 1), who, :2] - hips[max(frame - 6, 0), who, :2]
                    angle = math.atan2(travel[1], travel[0])
                    layer = event["layer"]
                    own = hex_at(x, y, radius)
                    blocked_with_own, blocked_without = True, True
                    for offset in (0.0, math.pi / 6, -math.pi / 6):
                        cells = [hex_at(x + math.cos(angle + offset) * d, y + math.sin(angle + offset) * d, radius) for d in LOOKS]
                        clear = lambda c: c in tiles[layer] and c not in gone[layer] and c not in touched[layer]
                        if all(clear(c) for c in cells):
                            blocked_with_own = False
                        if all(clear(c) for c in cells if c != own):
                            blocked_without = False
                    own_only += blocked_with_own and not blocked_without
                    truly += blocked_without
        print(folder, rounds, "rounds,", round(total / rounds, 1), "leaps per round")
        print(f"  in the first 8 s of play: {100 * early / total:.0f}% of leaps, {early / rounds:.1f} per round")
        print(f"  within 5 m of the centre: {100 * centre / total:.0f}%")
        print("  seconds into play p10/50/90:", np.percentile(times, [10, 50, 90]).round(1))
        print(f"  way blocked only because of the tile it stands on: {100 * own_only / total:.0f}%")
        print(f"  way blocked by other tiles too: {100 * truly / total:.0f}%")
        print(f"  landings followed by a fall within 1 s: {100 * fell_soon / max(landed, 1):.0f}%, within 2 s: {100 * fell_2 / max(landed, 1):.0f}%")


if __name__ == "__main__":
    main()
