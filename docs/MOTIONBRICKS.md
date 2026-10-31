# Connecting MotionBricks

Official source: https://github.com/NVlabs/motionbricks

Checked on 2026-10-30. The official repository provides SOMA human and Unitree G1 models, a Python runtime, scene and primitive editors, and a Scene Preview. Its documented runtime uses TensorRT and an NVIDIA GPU.

MotionBricks fills motion between recent poses and target poses. It does not supply the race rules, navigation policy, obstacle collision system, or multiplayer networking.

## Current boundary

`scripts/modal_race.py` runs the upstream character code headlessly and owns everything else. The browser never calls the model. It replays recorded joint positions.

- Race world: meters, Z up, forward along positive X. The browser maps this to Three.js as `(y, z, x)`.
- A recording is hips as float32 meters, then 20 joints as int16 millimeters from the hips, for each frame and runner.

Lessons from driving the runtime directly:

- Create a new Smart Primitive per action. `align_sequence_pivot` clamps to limits around a primitive's first placement, so reusing one drags the runner back to that spot.
- The model only sees poses. Writing a tumble into `current_pose` and recording it is enough for the next generation to start from the floor.
- Discard frames before the runner is moving and fail the run if a clip plays. Idle clips are sample motions under an evaluation-only license.

## First headless run

Measured on 2026-10-30 at upstream commit `54fc563` with SOMA FP32 on a Modal L4. `scripts/modal_clip.py` drives `SOMACharacter` without the viewer through three obstacles from the official gym scene. `scripts/check_clip.py` runs forward kinematics and draws the crossings.

| Obstacle | Primitive | Interaction | Joints inside the obstacle |
|---|---|---|---|
| 0.5 m box | `jump_0_5m` | 2.2 s to 4.0 s | 0 of 270 frames |
| 0.85 m wall | `jump_0_75m` | 2.2 s to 4.6 s | 2 of 270 frames, the lead foot clips it |
| Bar, underside at 1.45 m | `avoidance` | 2.2 s to 4.0 s | 0 of 270 frames |

The checks cover joint positions only, not the body mesh. The wall crossing is a leap with no hand plant. No primitive here crosses a gap, so the course's gap is still untested.

| Runners sharing one engine | Mean per frame | p95 | Max |
|---|---|---|---|
| 1 | 1.2 ms | 7.6 ms | 9.5 ms |
| 2 | 2.3 ms | 8.2 ms | 15.5 ms |
| 8 | 9.2 ms | 26.1 ms | 62.1 ms |

A 30 Hz frame allows 33.3 ms. The times cover pose work on the Modal GPU and exclude network and browser time. Recorded playback is the chosen path because it adds no network latency and replays the same race every time.

Poses are XYZ plus quaternion XYZW local transforms for 77 joints, Z up, root in world space, at 30 Hz. Runners cover about 3.3 m/s at max speed. Characters start in an idle clip from upstream's sample motions, which use an evaluation-only license, so check those frames before committing recorded poses.

## Implementation order

Use the official Scene Preview as the reference integration. Get one SOMA clip through a working runtime first, rather than inventing a server API around assumed model inputs. Keep one motion context per runner. Buffer generated chunks for playback and move each runner from its generated root trajectory. Keep race simulation and pose timestamps on an explicit shared clock.

The official preview documents walking, running, strafing, crouched strafing, jumping over objects, vaulting, ducking, and sitting. Do not assume gap jumps, falling, recovery, or physical contact are solved by the model. Those need a demonstrated primitive or a separate implementation.

Model files and generated recordings belong in ignored `models/` and `runs/` directories. Record model revisions and licenses when adding real assets. No weights or credentials are needed for the current skeleton.
