"""Render runs/gym_clips.npz as a side-view stick figure video, runs/gym_clips.mp4.

Run:  python scripts/render_clip.py
"""

import json
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from check_clip import RUNS, cube_bounds, world_positions

SIZE = (1280, 720)
LANE_COLORS = {"SmallJumpSP": "#d1495b", "MidJumpSP": "#2e86ab", "AvoidanceSP": "#3b8f5a"}
LEAD_IN_M = 3.0
LEAD_OUT_M = 2.5


def main():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is not on PATH")
    data = np.load(RUNS / "gym_clips.npz")
    meta = json.loads(str(data["meta"]))
    parents = data["parents"]
    fps = int(data["fps"])
    bones = [(p, j) for j, p in enumerate(parents) if p >= 0]

    out = RUNS / "gym_clips.mp4"
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
        stdin=subprocess.PIPE,
    )
    fig, ax = plt.subplots(figsize=(SIZE[0] / 100, SIZE[1] / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    count = 0
    for lane, color in LANE_COLORS.items():
        world = world_positions(data[f"poses_{lane}"], parents)
        y0 = world[0, 0, 1]
        cubes = [c for c in meta["cubes"] if cube_bounds(c)[2] - 0.5 < y0 < cube_bounds(c)[3] + 0.5]
        x0, x1, *_ = cube_bounds(cubes[0])
        first = int(np.argmax(world[:, 0, 0] > x0 - LEAD_IN_M))
        last = int(np.argmax(world[:, 0, 0] > x1 + LEAD_OUT_M))
        for frame in range(first, last + 1):
            ax.clear()
            ax.set_facecolor("#f4f1ea")
            ax.axhline(0, color="#555555", lw=2)
            for cube in cubes:
                bx0, bx1, _, _, bz0, bz1 = cube_bounds(cube)
                ax.add_patch(plt.Rectangle((bx0, bz0), bx1 - bx0, bz1 - bz0, color="#e8a33d"))
            for parent, joint in bones:
                ax.plot(world[frame, [parent, joint], 0], world[frame, [parent, joint], 2],
                        color=color, lw=3, solid_capstyle="round")
            center = (x0 + x1) / 2
            ax.set_xlim(center - 4.0, center + 4.0)
            ax.set_ylim(-0.3, 4.2)
            ax.set_axis_off()
            fig.canvas.draw()
            encoder.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
            count += 1
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out}, {count} frames, {count / fps:.1f} s")


if __name__ == "__main__":
    main()
