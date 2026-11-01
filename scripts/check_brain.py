"""Did the trained brain beat the hand-written policy in real recorded rounds?

Run after: modal run scripts/modal_hex.py --seeds 101,102,... --brains half
       or: python scripts/check_brain.py --first 101 --last 124
Each round gives four random robots the network and four the old policy, so a tie is 50% of wins.
"""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=101)
    parser.add_argument("--last", type=int, default=124)
    parser.add_argument("--dir", default="runs/hex")
    args = parser.parse_args()
    groups = {True: {"wins": 0, "falls": 0, "tiles": 0, "seconds": [], "knocks": 0, "leaps": 0, "landed": 0, "speed": []}, False: {"wins": 0, "falls": 0, "tiles": 0, "seconds": [], "knocks": 0, "leaps": 0, "landed": 0, "speed": []}}
    rounds, lengths = 0, []
    for seed in range(args.first, args.last + 1):
        path = Path(args.dir) / f"seed_{seed}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        meta = json.loads(path.read_text())
        rounds += 1
        win = next(e for e in meta["events"] if e["type"] == "winner")
        lengths.append(win["t"])
        for index, runner in enumerate(meta["runners"]):
            group = groups[runner["brain"]]
            group["wins"] += index == win["runner"]
            group["falls"] += runner["falls"]
            group["tiles"] += runner["tilesTouched"]
            group["seconds"].append(runner["eliminatedAt"] if runner["eliminatedAt"] is not None else win["t"])
            group["knocks"] += runner["knocksDealt"]
            group["leaps"] += runner["leaps"]
            group["landed"] += runner["leapsLanded"]
            group["speed"].append(runner["medianSpeed"])
    print(f"{rounds} rounds, length {min(lengths):.1f} to {max(lengths):.1f} s, median {sorted(lengths)[len(lengths) // 2]:.1f} s")
    for brain, group in groups.items():
        name = "network" if brain else "script "
        seconds = sum(group["seconds"]) / len(group["seconds"])
        print(f"{name}: {group['wins']}/{rounds} wins, lasted {seconds:.1f} s on average, falls {group['falls']}, tiles {group['tiles']}, tiles per second alive {group['tiles'] / sum(group['seconds']):.2f}, knockdowns dealt {group['knocks']}, leaps {group['landed']}/{group['leaps']}, speed {sum(group['speed']) / len(group['speed']):.2f} m/s")
    wins = groups[True]["wins"]
    # Chance of doing this well if both policies were equal.
    tail = sum(math.comb(rounds, k) for k in range(wins, rounds + 1)) / 2**rounds
    print(f"network wins {wins}/{rounds}, a one-sided p of {tail:.2g} against a fair coin")


if __name__ == "__main__":
    main()
