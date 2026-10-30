# Connecting MotionBricks

Official source: https://github.com/NVlabs/motionbricks

Checked on 2026-10-30. The official repository provides SOMA human and Unitree G1 models, a Python runtime, scene and primitive editors, and a Scene Preview. Its documented runtime uses TensorRT and an NVIDIA GPU.

MotionBricks fills motion between recent poses and target poses. It does not supply the race rules, navigation policy, obstacle collision system, or multiplayer networking.

## Current boundary

`src/motion/provider.ts` defines this project's contract. It is not an existing NVIDIA HTTP endpoint or upstream wire format. `MotionBricksProvider.generate()` deliberately throws until integration is implemented. The browser preview does not call it.

- Input: runner identity, desired action, world position, facing, speed, and recent pose frames.
- Output: timed skeletal frames for that runner.
- Project world: meters, Y up, forward along positive Z. Joint rotations use quaternion XYZW.
- Joint naming, local versus world joint coordinates, timestamps, upstream axes, and rig retargeting still need to be specified from real output.

## Implementation order

Use the official Scene Preview as the reference integration. Get one SOMA clip through a working runtime first, rather than inventing a server API around assumed model inputs. Keep one motion context per runner. Buffer generated chunks for playback and move each runner from its generated root trajectory. Keep race simulation and pose timestamps on an explicit shared clock.

The official preview documents walking, running, strafing, crouched strafing, vaulting, ducking, and sitting. Do not assume jumping, falling, recovery, or physical contact are solved by the model. Those need a demonstrated primitive or a separate implementation.

Model files and generated recordings belong in ignored `models/` and `runs/` directories. Record model revisions and licenses when adding real assets. No weights or credentials are needed for the current skeleton.
