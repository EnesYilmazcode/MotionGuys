"""Check runs/gym_clips.npz: print motion stats and draw one contact sheet per lane.

Run:  python scripts/check_clip.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs"
PANELS = 10


def quat_rotate(q, v):
    u, w = q[..., :3], q[..., 3:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_mul(a, b):
    au, aw = a[..., :3], a[..., 3:]
    bu, bw = b[..., :3], b[..., 3:]
    return np.concatenate(
        [aw * bu + bw * au + np.cross(au, bu), aw * bw - np.sum(au * bu, -1, keepdims=True)], -1
    )


def world_positions(poses, parents):
    """Forward kinematics for [T, J, 7] local transforms, XYZ then quaternion XYZW."""
    pos = np.zeros(poses.shape[:2] + (3,))
    rot = np.zeros(poses.shape[:2] + (4,))
    for j, parent in enumerate(parents):
        if parent < 0:
            pos[:, j], rot[:, j] = poses[:, j, :3], poses[:, j, 3:]
        else:
            pos[:, j] = pos[:, parent] + quat_rotate(rot[:, parent], poses[:, j, :3])
            rot[:, j] = quat_mul(rot[:, parent], poses[:, j, 3:])
    return pos


def cube_bounds(cube):
    # The gym's cubes are axis aligned up to a 90 degree yaw, and width is along world X.
    x, y, z = cube["position"]
    width, depth, height = cube["dimensions"]
    quarter_turn = abs(abs(cube["rotation"][2]) - 0.7071) < 0.01
    dx, dy = (depth, width) if quarter_turn else (width, depth)
    return x - dx / 2, x + dx / 2, y - dy / 2, y + dy / 2, z - height / 2, z + height / 2


def main():
    data = np.load(RUNS / "gym_clips.npz")
    meta = json.loads(str(data["meta"]))
    print(json.dumps(meta["report"], indent=2))
    print("cubes:", json.dumps(meta["cubes"]))
    parents = data["parents"]
    names = [str(n) for n in data["joint_names"]]
    fps = int(data["fps"])
    print(f"{len(names)} joints, first: {names[:6]}")

    for key in [k for k in data.files if k.startswith("poses_")]:
        lane = key.removeprefix("poses_")
        poses = data[key]
        world = world_positions(poses, parents)
        y0 = world[0, 0, 1]
        ground_speed = np.linalg.norm(np.diff(world[:, 0, :2], axis=0), axis=1) * fps
        lane_cubes = [
            c for c in meta["cubes"] if cube_bounds(c)[2] - 0.5 < y0 < cube_bounds(c)[3] + 0.5
        ]
        print(
            f"\n{lane}: {len(poses)} frames, root x {world[0, 0, 0]:.2f} to {world[-1, 0, 0]:.2f}, "
            f"drift y {world[:, 0, 1].min() - y0:+.2f}/{world[:, 0, 1].max() - y0:+.2f}, "
            f"median speed {np.median(ground_speed):.2f} m/s"
        )
        print(
            f"  root height {world[:, 0, 2].min():.2f} to {world[:, 0, 2].max():.2f}, "
            f"head top {world[:, :, 2].max(axis=1).min():.2f} to {world.max(axis=(1))[:, 2].max():.2f}, "
            f"lowest joint {world[:, :, 2].min():.2f}"
        )
        for cube in lane_cubes:
            x0, x1, y_min, y_max, z0, z1 = cube_bounds(cube)
            inside = (
                (world[..., 0] > x0) & (world[..., 0] < x1)
                & (world[..., 1] > y_min) & (world[..., 1] < y_max)
                & (world[..., 2] > z0) & (world[..., 2] < z1)
            )
            hit = inside.any(axis=1)
            over = (world[:, 0, 0] > x0) & (world[:, 0, 0] < x1)
            print(
                f"  {cube['name']} x[{x0:.2f},{x1:.2f}] z[{z0:.2f},{z1:.2f}]: "
                f"{hit.sum()} frames with a joint inside, root crossed: {bool(over.any())}"
            )

        # Show the crossing, from 1.5 m before the obstacle to 1.5 m after it.
        x0, x1, *_ = cube_bounds(lane_cubes[0])
        first = int(np.argmax(world[:, 0, 0] > x0 - 1.5))
        last = int(np.argmax(world[:, 0, 0] > x1 + 1.5))
        frames = np.linspace(first, last, PANELS).astype(int)
        fig, axes = plt.subplots(1, PANELS, figsize=(PANELS * 2.2, 3.2), sharey=True)
        for ax, frame in zip(axes, frames):
            for j, parent in enumerate(parents):
                if parent >= 0:
                    ax.plot(world[frame, [parent, j], 0], world[frame, [parent, j], 2], "k-", lw=1)
            for cube in lane_cubes:
                x0, x1, _, _, z0, z1 = cube_bounds(cube)
                ax.add_patch(plt.Rectangle((x0, z0), x1 - x0, z1 - z0, color="tab:orange", alpha=0.5))
            cx = world[frame, 0, 0]
            ax.set_xlim(cx - 1.2, cx + 1.2)
            ax.set_ylim(-0.05, 2.6)
            ax.set_aspect("equal")
            ax.set_title(f"{frame / fps:.1f}s", fontsize=9)
        fig.suptitle(lane)
        fig.tight_layout()
        fig.savefig(RUNS / f"{lane}.png", dpi=90)
        plt.close(fig)
        print(f"  sheet: {RUNS / (lane + '.png')}")


if __name__ == "__main__":
    main()
