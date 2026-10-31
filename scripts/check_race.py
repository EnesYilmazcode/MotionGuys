"""Summarize runs/races/seed_N: progress over time, failures by cause, and stuck runners.

Run:  python scripts/check_race.py 1 2 3
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RACES = Path(__file__).resolve().parent.parent / "runs" / "races"


def main():
    for seed in sys.argv[1:] or ["1"]:
        meta = json.loads((RACES / f"seed_{seed}.json").read_text())
        count = len(meta["runners"])
        data = (RACES / f"seed_{seed}.bin").read_bytes()
        hips = np.frombuffer(data, np.float32, meta["frames"] * count * 3).reshape(meta["frames"], count, 3)
        fps = meta["fps"]
        print(f"\nseed {seed}: {meta['frames'] / fps:.1f} s, windows {meta['windows']}")
        print("real doors:", [r["real"] for r in meta["course"]["doorRows"]])
        print("events:", dict(Counter(e["type"] for e in meta["events"])))
        for i, runner in enumerate(meta["runners"]):
            track = hips[:, i]
            alive = ~np.isnan(track[:, 0])
            samples = " ".join(
                f"{track[f, 0]:.0f}" if alive[f] else "-" for f in range(0, meta["frames"], 10 * fps)
            )
            valid = track[alive]
            still = np.linalg.norm(np.diff(valid[:, :2], axis=0), axis=1) * fps
            print(
                f"{runner['name']:5s} finish {runner['finishedAt'] and round(runner['finishedAt'], 1)} "
                f"{runner['counts']} last y {valid[-1, 1]:.1f} median speed {np.median(still):.2f} | x every 10 s: {samples}"
            )


if __name__ == "__main__":
    main()
