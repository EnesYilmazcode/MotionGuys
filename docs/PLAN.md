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

The first round works and is recorded. Next:

1. Move the camera closer and cut to the falls, because runners look small from this distance.
2. End a recording when the winner is decided, so the winner never falls afterward.
3. Look for foot sliding on sharp turns, and tune turning or tile size if it shows.
4. Try a jump over a tile. It needs a longer flight and a much shorter run-up than the upstream primitives have.
5. Try real SOMA body shapes in place of drawn builds, and check for foot sliding.

## Later

Saved camera cuts for a real video, runner collisions, and physics for falls in place of the scripted tumble.

## Scope

Original geometry and characters only, no ripped game assets. Online networking, accounts, and a level editor can wait.
