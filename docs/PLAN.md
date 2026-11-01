# MotionGuys

Repository name: `MotionGuys`

GitHub description: `MotionBricks characters race through an obstacle course`

Eight generated runners on courses that can actually beat them. MotionBricks supplies body motion. This project supplies the world, the rules, and the failures.

## Done

- Headless MotionBricks on Modal, with clearance and timing measurements in `MOTIONBRICKS.md`.
- An eight-runner race with fake doors, two sweepers, and fake tiles.
- Failures decided by generated joints against course geometry, with a scripted tumble and a generated get-up.
- A browser replay of recorded races on an original mannequin.

## Done since

- Hex-A-Bots: an arena rebuilt from footage, with real robot meshes. See `HEXAGONS.md`.

## Next: a slower, tenser final

Rounds now last about 70 s. Next:

1. Slow the runners down, and test whether the leap still works from a slower approach.
2. Reduce foot sliding on turns, which a slow close-up will make obvious.
3. Steer leap landings, because touchdown distance varies widely and causes misses.
4. Render a close final between the last two runners, with sound, as the first real post.
5. As a second post, compare Kimodo and MotionBricks on one simple task. Both use the SOMA skeleton and the same mocap data.

## Later

Saved camera cuts for a real video, runner collisions, and physics for falls in place of the scripted tumble.

## Scope

Original geometry, no ripped game assets. Robot models only under licenses that allow redistribution, credited in `CREDITS.md`. Online networking, accounts, and a level editor can wait.
