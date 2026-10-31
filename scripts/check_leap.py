"""Draw jump settings from runs/leaps.npz side by side, to check the numbers by eye.

Run:  python scripts/check_leap.py place2.2_stretch2.4 place1.6_stretch1.8
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs"
BONES = (
    ("Hips", "Chest"), ("Chest", "Neck1"), ("Neck1", "Head"), ("Head", "HeadEnd"),
    ("Chest", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Chest", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftLeg"), ("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot"), ("LeftFoot", "LeftToeEnd"),
    ("Hips", "RightLeg"), ("RightLeg", "RightShin"), ("RightShin", "RightFoot"), ("RightFoot", "RightToeEnd"),
)
FPS = 30
PANELS = 12
STEP_FRAMES = 3


def main():
    data = np.load(RUNS / "leaps.npz")
    names = [str(n) for n in data["joints"]]
    index = {name: i for i, name in enumerate(names)}
    keys = sys.argv[1:]
    fig, axes = plt.subplots(len(keys), PANELS, figsize=(PANELS * 1.5, len(keys) * 2.4), squeeze=False)
    for row, key in enumerate(keys):
        track = data[key]
        for column, ax in enumerate(axes[row]):
            frame = column * STEP_FRAMES
            pose = track[frame]
            for a, b in BONES:
                ax.plot(pose[[index[a], index[b]], 0], pose[[index[a], index[b]], 2], "k-", lw=1)
            ax.axhline(0, color="tab:orange", lw=1.5)
            center = pose[index["Hips"], 0]
            ax.set_xlim(center - 1.1, center + 1.1)
            ax.set_ylim(-0.1, 2.6)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{frame / FPS:.1f}s x={center:.1f}", fontsize=7)
        axes[row][0].set_ylabel(key, fontsize=7)
    fig.tight_layout()
    fig.savefig(RUNS / "leaps.png", dpi=95)
    for key in keys:
        hips = data[key][:, index["Hips"], 0]
        steps = np.diff(hips) * FPS
        print(key, "hips speed median", round(float(np.median(steps)), 2), "max", round(float(steps.max()), 2), "biggest single frame jump m", round(float(np.diff(hips).max()), 3))


if __name__ == "__main__":
    main()
