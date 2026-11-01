"""Summarize recorded hexagon rounds. Run:  python scripts/check_hex.py [runs/hex]"""

import glob
import json
import sys
from statistics import median

NAMES = ("Optimus", "Figure", "G1", "H1", "Apollo", "Atlas", "T1", "Petman")
folder = sys.argv[1] if len(sys.argv) > 1 else "runs/hex"
wins, rounds = {}, []
for path in sorted(glob.glob(f"{folder}/seed_*.json")):
    with open(path) as file:
        meta = json.load(file)
    events = meta["events"]
    winner = next(e["runner"] for e in events if e["type"] == "winner")
    wins[NAMES[winner]] = wins.get(NAMES[winner], 0) + 1
    runners = meta["runners"]
    out = sorted(r["eliminatedAt"] for r in runners if r["eliminatedAt"])
    win_at = next(e["t"] for e in events if e["type"] == "winner")
    causes = [e["cause"] for e in events if e["type"] == "fall"]
    rounds.append({
        "seconds": win_at, "tiles": sum(e["type"] == "tile_touch" for e in events), "falls": len(causes),
        "hole": causes.count("hole"), "edge": causes.count("edge"), "leap": causes.count("leap"),
        "leaps": sum(r["leaps"] for r in runners), "landed": sum(r["leapsLanded"] for r in runners),
        "getups": sum(len(r["getupSeconds"]) for r in runners), "getup_list": [s for r in runners for s in r["getupSeconds"]],
        "unrecovered": sum(r["unrecovered"] for r in runners),
        "speed": median(r["medianSpeed"] for r in runners), "slide": meta["footSlide"]["p95MetersPerSecond"], "slide50": meta["footSlide"]["medianMetersPerSecond"],
    })
    print(f"seed {meta['seed']}: winner {NAMES[winner]} at {win_at:.1f} s, last gap {out[-1] - out[-2]:.1f} s, {rounds[-1]['falls']} falls, {rounds[-1]['leaps']} leaps")
print("wins", wins)
total = lambda key: sum(r[key] for r in rounds)
getups = [s for r in rounds for s in r["getup_list"]]
print(f"rounds {len(rounds)}, seconds {min(r['seconds'] for r in rounds):.1f} to {max(r['seconds'] for r in rounds):.1f}")
print(f"tiles {total('tiles')}, falls {total('falls')} (hole {total('hole')}, edge {total('edge')}, leap {total('leap')})")
print(f"leaps {total('leaps')}, landed {total('landed')}")
print(f"get-ups {total('getups')}, median {median(getups):.2f} s, slowest {max(getups):.2f} s, unrecovered {total('unrecovered')}")
print(f"median runner speed {median(r['speed'] for r in rounds):.2f} m/s, foot slide median {median(r['slide50'] for r in rounds):.2f}, p95 {median(r['slide'] for r in rounds):.2f} m/s")
