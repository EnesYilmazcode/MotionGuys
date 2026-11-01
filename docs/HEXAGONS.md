# Rebuilding the hexagon round from footage

The first Spleef round was built from memory of how this kind of round works. Most of what memory said was wrong. This is what real gameplay footage shows, frame by frame, and what the replay does about it.

Footage studied, not stored in this repo: two no-commentary rounds on YouTube, `z4QXKVR6kms` at 720p30 and `A3z75iHnlSM` at 1080p30. The written rules come from the community wiki.

## What was wrong

| Topic | First version | Footage | Now |
|---|---|---|---|
| Tile shape | Squares in a grid | Hexagons in a honeycomb, hairline gaps | Same |
| Touched tile | Shakes, then falls out of the floor | Stays still, whitens, disappears in place | Same |
| Tile timer | 1.5 s | 0.43 s from first foot ring to gone | 0.9 s, see below |
| Floors | 3, all the same size | Many, each wider than the one above | 6, 11 m to 18.5 m radius |
| Tile color | Checkerboard | One color per floor, about 1 tile in 6 darker | Same |
| Start | On the floor, already running | Start hexagons floating above the top floor | Same |
| Bottom | A void | Pink slime with grass islands | Same |
| Surroundings | Dark blue background | Padded octagon rails per floor, candy hills, floating signs | Same, in original geometry |
| End of a round | Abrupt stop | The floor freezes, winner banner | Same, plus a crown |

## Measured

- **Tile timer.** At 30 fps a foot ring appears on frame 5, the tile starts to pale on frame 11, is full white on frame 16, and is gone on frame 18. That is 0.43 s, and color holds for the first 47% of it. The tile is cut from white to nothing in one frame.
- **Tile size.** A tile spans about 1.8 times a character's height corner to corner.
- **Colors.** Sampled from frames: yellow `#f8b823` and `#cc8f25`, blue `#147ac5`, purple `#94429c`, slime `#fd95bf`, rails `#ec6416`.
- **Layer order.** Blue is directly under the yellow top and the bottom floor is blue, which the wiki also says.

## Inferred, not measured

- **Six floors.** No source states the count. Six tiers are visible in the footage, and the wiki's 1,634 tiles for a full arena fit six floors of about 272 tiles.
- **The middle of the color order.** Yellow, blue, purple, yellow, purple, blue fits what shows through the holes, but the footage never shows all six at once.

## Changed on purpose

- **The timer is 0.9 s, not 0.43 s.** In the footage a character crosses a tile in about 0.33 s, so a tile lasts 1.3 crossings. MotionBricks runners top out near 3.3 m/s, so crossing takes about 0.7 s here. At a literal 0.43 s, runners fell through their own tiles and all eight were out in 25 s. Keeping the ratio gave rounds of 39 s to 55 s with the hand-written policy. With the trained network rounds run 61 s to 89 s.
- **Tiles are 2.4 m corner to corner, not 3.2 m.** These bodies are tall and thin, not short and round, and slower, so a smaller tile keeps the same rhythm.
- **Eight players, not twenty,** with floors to match, like the game's smaller variant.
- **Falls look different.** The game's characters land on their feet. Here a fall is a MuJoCo ragdoll and a generated get-up, because recovery is the point of using MotionBricks.

## Other fixes from the audit

- **Support comes from feet.** Spleef dropped a runner as soon as the hips were over a hole. Now a planted foot on a tile holds the runner up. Foot sliding at the 95th percentile went from about 6 m/s to 4.6 m/s, measured over 12 rounds against the 6 Spleef rounds, though the two arenas differ. Slower gaits and shoves between robots later pushed it back up to about 5.4 m/s.
- **No idle clips in recordings.** Zero input drops MotionBricks into upstream idle clips, which are under an evaluation-only license. A waiting robot holds its last generated pose instead, and the browser shifts its weight slowly during the countdown and after a win. The run fails if a clip ever plays.

## Robots

The recording stores joint positions, not bone rotations, and robots are rigid links anyway. `scripts/build_robots.py` cuts each model into 13 parts, one per bone, and the replay stretches each part between its two recorded joints. Nothing is rigged or skinned, and any model with a rough human layout works after someone writes down its joint positions.

The real G1, H1, and Apollo come from official MuJoCo Menagerie meshes, cut along their real links. Sketchfab models are cut by nearest bone, per triangle.

There is no usable Figure model. No downloadable one turned up on Sketchfab, so "Figure" is a repainted CC BY Tesla Bot mesh, close in shape but not a real scan. Petman is a CC BY Sketchfab model of the Boston Dynamics robot, and T1 is the official Booster Robotics model in MuJoCo Menagerie. The crown is a CC0 museum prop.

## Physics

ACP is still not usable, see below, so the physics is built here. `scripts/hex_physics.py` puts all eight robots in one MuJoCo scene, built from the MotionBricks skeleton, so each ball joint maps straight onto a MotionBricks local rotation. Knees and elbows are hinges, because every recorded pose bends them one way around local z. Tiles are hexagon prisms that lose their collision when they vanish.

A running robot is copied in from MotionBricks each frame and acts as a solid obstacle. A limp robot is simulated, 16 steps per 30 Hz frame. Each joint is pulled toward the pose the robot was in when it fell, with a small authored sway while airborne, and softens on landing. The body lands wherever contact puts it.

An early version sent every fallen robot into a T-pose. MuJoCo reads a joint `range` in degrees by default, and the limits were written in radians, so 150 degrees became 2.6. Every joint was pinned near the skeleton's rest pose, which is a T. A test that compares each joint to its held pose in free fall now reads 0.0 radians of drift.

This is a handoff, not a physics controller tracking the motion. A running robot has no balance to lose. The closer option, driving a simulated body to follow MotionBricks, would need its own trained tracker, with [ProtoMotions](https://github.com/NVlabs/ProtoMotions) as the likely starting point.

## Why ACP is not in here

[ACP](https://leggedrobotics.github.io/ACP/) is a real physics policy for the Unitree G1, so unlike MotionBricks it can really fall and recover. It would be a good fit. No code, weights, or license turned up as of 2026-10-31, so it cannot be redistributed. Worth checking again if the code comes out.
