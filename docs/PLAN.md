# MotionGuys

Repository name: `MotionGuys`

GitHub description: `MotionBricks characters race through an obstacle course`

Eight generated runners on courses that can actually beat them. MotionBricks supplies body motion. This project supplies the world, the rules, and the failures.

## Done

- Headless MotionBricks on Modal, with clearance and timing measurements in `MOTIONBRICKS.md`.
- An eight-runner race with fake doors, two sweepers, and fake tiles.
- Failures decided by generated joints against course geometry, with a scripted tumble and a generated get-up.
- A browser replay of recorded races on an original mannequin.

## Next: Spleef

Eight runners on a floor that falls away where they step, last one standing wins. It suits MotionBricks because failure needs no explanation and most of the skill is footwork.

1. Size tiles from the measured jump. `jump_0_75m` clears about 0.6 m, so runners cannot jump a normal missing tile. They route around holes, unless a longer authored jump proves itself.
2. Test sharp turns at running speed, which set the smallest usable tile.
3. Stack floors so a fall lands on the layer below, and the bottom layer eliminates.
4. Try different SOMA body shapes and check for foot sliding before promising them.

## Later

Saved camera cuts for a real video, runner collisions, and physics for falls in place of the scripted tumble.

## Scope

Original geometry and characters only, no ripped game assets. Online networking, accounts, and a level editor can wait.
