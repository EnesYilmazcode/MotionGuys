# MotionGuys

Repository name: `MotionGuys`

GitHub description: `MotionBricks characters race through an obstacle course`

Eight autonomous runners in a colorful obstacle race. The camera watches them compete. Start with a single course and make the body movement worth watching.

## Skeleton, implemented

- Three.js browser preview with eight colored runners.
- Seeded fixed-step race, standings, pause, restart, and orbit camera.
- A course with explicit run, vault, duck, and jump sections.
- Motion provider types and an explicit unconnected MotionBricks adapter.
- README in the Fly2048 and JevRubiksCube format.

## First real motion milestone

1. Done 2026-10-30: run the upstream runtime headlessly on Modal, using the character code behind Scene Preview. Local throughput on the RTX 3050 Ti is unmeasured.
2. Done 2026-10-30: record one character running, leaping the wall, and ducking under the bar. Results are in `MOTIONBRICKS.md`.
3. Next: establish unit, axis, joint, and timing conversion. Play the output on one real skeleton in the browser.
4. Replace one procedural runner with generated motion. Do not also advance its root independently, which would apply movement twice.
5. Check clearance and planted feet against course geometry. Let actual motion completion determine obstacle exit time.
6. Done 2026-10-30: measure one runner, then two, then eight. Use recorded motion playback.

The course's gap has no matching primitive. Upstream's jumps cross a box and a wall, so make the jump section a solid obstacle or validate a gap crossing before treating it as supported.

## Make it a race

Move beyond seeded speed differences: obstacle timing, alternate routes, stumbles and checkpoint recovery. Add collision handling and clear rules for falls. Measure which behaviors come from the controller, generated motion, and physics.

## Make the video

Save the seed, controller decisions, model/version information, generated poses, and finish events. Replay those records with spectator camera cuts. Only then add a real hero GIF and measured results to the README.

## Scope

One course, eight runners, a spectator camera. Online networking, accounts, asset generation, and a level editor can wait.
