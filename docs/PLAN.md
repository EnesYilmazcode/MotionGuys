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

1. Run the official SOMA Scene Preview independently. Confirm hardware requirements and local throughput before choosing local inference or Modal.
2. Record one character running, vaulting the wall, and ducking under the ceiling. Reuse the upstream primitives and rig conventions where possible.
3. Establish unit, axis, joint, and timing conversion. Play the output on one real skeleton in the browser.
4. Replace one procedural runner with generated motion. Do not also advance its root independently, which would apply movement twice.
5. Check clearance and planted feet against course geometry. Let actual motion completion determine obstacle exit time.
6. Measure one runner, then two, then eight. Decide between live generation and recorded motion playback from those measurements.

The jump section is a visual placeholder. Validate an appropriate primitive or replace it with another demonstrated interaction before treating it as a supported MotionBricks action.

## Make it a race

Move beyond seeded speed differences: obstacle timing, alternate routes, stumbles and checkpoint recovery. Add collision handling and clear rules for falls. Measure which behaviors come from the controller, generated motion, and physics.

## Make the video

Save the seed, controller decisions, model/version information, generated poses, and finish events. Replay those records with spectator camera cuts. Only then add a real hero GIF and measured results to the README.

## Scope

One course, eight runners, a spectator camera. Online networking, accounts, asset generation, and a level editor can wait.
