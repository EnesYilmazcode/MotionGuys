"""The mathematical ceiling: how long can one robot possibly last on N tiles?

Run:  python scripts/ceiling.py

A tile lasts TOUCH_SECONDS from the first foot on it and each tile can only be used once, so survival is bounded by how long a
robot can stay on each tile plus any time spent in the air between them. A path visiting every tile once has to exist too, so this
searches for one. Numbers come from scripts/modal_stop.py and real hops. These are upper bounds, not targets anyone can quite reach.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hex_brain import GAIT_SPEED, HEX_RADIUS, HEX_WIDTH, TOUCH_SECONDS, small_floor

HOP_AIR_SECONDS = 0.47
TILE_WIDTH = HEX_WIDTH
ADJACENT = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


def longest_path(cells):
    """Longest walk through neighbouring cells, each used once. A full tour is usually found at once."""
    cells = set(cells)
    best = 0

    def go(cell, seen):
        nonlocal best
        best = max(best, len(seen))
        if best == len(cells):
            return True
        for dq, dr in ADJACENT:
            next_cell = (cell[0] + dq, cell[1] + dr)
            if next_cell in cells and next_cell not in seen:
                seen.add(next_cell)
                if go(next_cell, seen):
                    return True
                seen.remove(next_cell)
        return False

    for start in sorted(cells):
        if go(start, {start}):
            break
    return best


def main():
    crossing = TILE_WIDTH / GAIT_SPEED[0]
    print(f"tile timer {TOUCH_SECONDS} s, tile {TILE_WIDTH:.2f} m wide, slow walk {GAIT_SPEED[0]} m/s crosses one in {crossing:.2f} s, hop {HOP_AIR_SECONDS} s in the air")
    print("A: never stops, slow walk straight across each tile")
    print("B: stays the full timer on each tile, stepping across")
    print("C: stays the full timer on each tile, then hops, the true ceiling")
    print(f"{'tiles':>5} {'tour':>5} {'A':>7} {'B':>7} {'C':>7}")
    for count in (5, 7, 10, 13, 16, 19, 25, 31, 37):
        reach = longest_path(small_floor(count))
        a = reach * min(crossing, TOUCH_SECONDS)
        b = reach * TOUCH_SECONDS
        c = reach * TOUCH_SECONDS + (reach - 1) * HOP_AIR_SECONDS
        print(f"{count:5} {reach:5} {a:6.1f}s {b:6.1f}s {c:6.1f}s")


if __name__ == "__main__":
    main()
