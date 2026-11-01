"""Compare policies by how they play real recorded rounds: how long, how often they fall, and how the leaps go.

Run:  python scripts/check_leaps.py runs/smart_base runs/smart_bigsave
"""

import json
import sys
from pathlib import Path

import numpy as np


def summarize(folder):
    rounds = []
    for path in sorted(Path(folder).glob("seed_*.json")):
        meta = json.loads(path.read_text())
        events = meta["events"]
        winner = next((e for e in events if e["type"] == "winner"), None)
        last, on_foot, leap_from = {}, [], {}
        for event in events:
            runner = event.get("runner")
            if event["type"] == "go":
                last = {i: event["t"] for i in range(len(meta["runners"]))}
            elif event["type"] == "recover":
                last[runner] = event["t"]
            elif event["type"] in ("fall", "knock") and last.get(runner) is not None:
                on_foot.append(event["t"] - last[runner])
                last[runner] = None
        leaps = [e for e in events if e["type"] == "leap"]
        touches = [e for e in events if e["type"] == "leap_touch"]
        tiles = sum(sum(1 for q in floor) for floor in meta["arena"]["tiles"])
        rounds.append({
            "seconds": winner["t"] if winner else None,
            "falls": sum(1 for e in events if e["type"] == "fall"),
            "leapFalls": sum(1 for e in events if e["type"] == "fall" and e.get("cause") == "leap"),
            "knocks": sum(1 for e in events if e["type"] == "knock"),
            "tiles": sum(r["tilesTouched"] for r in meta["runners"]) / tiles,
            "leaps": len(leaps),
            "gapLeaps": sum(1 for e in leaps if e.get("blocked")),
            "landed": sum(1 for e in touches if e["floor"]),
            "touched": len(touches),
            "onFoot": on_foot,
            "wins": winner["runner"] if winner else None,
        })
    return rounds


def main():
    print(f"{'folder':24} {'n':>3} {'sec p10/50/90':>16} {'falls':>6} {'onfoot':>6} {'tiles':>6} {'leaps':>6} {'save%':>5} {'land%':>6} {'failed':>6}")
    for folder in sys.argv[1:]:
        rounds = summarize(folder)
        if not rounds:
            print(folder, "no rounds")
            continue
        seconds = [r["seconds"] for r in rounds if r["seconds"] is not None]
        mean = lambda key: sum(r[key] for r in rounds) / len(rounds)
        foot = [s for r in rounds for s in r["onFoot"]]
        leaps, gaps = sum(r["leaps"] for r in rounds), sum(r["gapLeaps"] for r in rounds)
        landed, touched = sum(r["landed"] for r in rounds), sum(r["touched"] for r in rounds)
        band = "/".join(f"{v:.0f}" for v in np.percentile(seconds, [10, 50, 90])) if seconds else "none"
        print(
            f"{Path(folder).name:24} {len(rounds):3} {band:>16} {mean('falls'):6.1f} {np.median(foot):6.1f} {100 * mean('tiles'):5.0f}% {mean('leaps'):6.1f} "
            f"{100 * gaps / max(leaps, 1):4.0f}% {100 * landed / max(touched, 1):5.0f}% {mean('leapFalls'):6.1f}"
        )


if __name__ == "__main__":
    main()
