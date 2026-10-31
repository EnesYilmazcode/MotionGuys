# MotionGuys

Repository name: `MotionGuys`

GitHub description: `MotionBricks characters race through an obstacle course`

Eight generated runners on courses that can actually beat them. MotionBricks supplies body motion. This project supplies the world, the rules, and the failures.

## Done

- Headless MotionBricks on Modal, with clearance and timing measurements in `MOTIONBRICKS.md`.
- An eight-runner race with fake doors, two sweepers, and fake tiles.
- Failures decided by generated joints against course geometry, with a scripted tumble and a generated get-up.
- A browser replay of recorded races on an original mannequin.

## Next: better Spleef

Jumps and the spectator camera work. Next:

1. Steer leap landings, because touchdown distance varies widely and causes misses.
2. Reduce foot sliding on sharp turns, which shows more up close.
3. Try real SOMA body shapes in place of drawn builds, and check for foot sliding.
4. Add sound and a tighter cut for the X video.

## Later

Saved camera cuts for a real video, runner collisions, and physics for falls in place of the scripted tumble.

## Scope

Original geometry and characters only, no ripped game assets. Online networking, accounts, and a level editor can wait.
