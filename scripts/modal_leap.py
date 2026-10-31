"""Measure a shorter, longer jump built from the upstream no-hands jump.

Run:  modal run scripts/modal_leap.py

The upstream jump spends about 2 s running to its entry pose. This skips that
target, places the airborne pose close ahead, and pushes the landing farther out,
then measures what MotionBricks actually generates. Writes runs/leaps.npz.
"""

import io
import json
from pathlib import Path

import modal
from modal_race import CACHE, DT, FEET, FPS, RECORDED, Skeleton, image, volume

SPEED_STRENGTH = 0.75
PLACE_AHEAD = (1.6, 2.2, 2.8)
STRETCH = (0.0, 0.6, 1.2, 1.8, 2.4)
REPEATS = 2
AIRBORNE_HEIGHT = 0.15

app = modal.App("motionguys-leap", image=image.add_local_python_source("modal_race"))


def make_leap_type():
    import numpy as np
    from app.runtime.common.smart_primitive_behaviors import SmartPrimitiveBehavior
    from motionbricks.runtime.target_pose import TargetPoseInferenceParams

    class Leap(SmartPrimitiveBehavior):
        """Go straight to the airborne pose, then to a landing pushed farther out."""

        place_ahead = 2.2
        stretch = 1.2
        direction = np.array([1.0, 0.0])

        def get_targets(self, character_properties, output_target_poses, stream=None):
            index = self.sequence_index("NoHandsJump")
            self.align_sequence_pivot(index, character_properties, stream)
            params = self.get_world_target_poses_for_sequence(index, output_target_poses, stream)
            poses = output_target_poses.numpy().copy()
            action, landing = poses[4:8].copy(), poses[8:12].copy()
            ground = np.asarray(character_properties.position)[:2]
            shift = ground + self.direction * self.place_ahead - action[0, 0, :2]
            action[:, 0, :2] += shift
            landing[:, 0, :2] += shift + self.direction * self.stretch
            poses[0:4], poses[4:8] = action, landing
            output_target_poses.assign(poses)
            return (
                TargetPoseInferenceParams(transition_early_by_frames=2, num_poses_to_use=4, allowed_token_counts=(6, 7, 8)),
                TargetPoseInferenceParams(
                    transition_early_by_frames=params[2].transition_early_by_frames,
                    num_poses_to_use=4,
                    allowed_token_counts=(6, 7, 8, 9, 10),
                ),
            )

    return Leap


@app.function(gpu="A10", volumes={CACHE: volume}, timeout=1500)
def measure() -> tuple[list[dict], bytes]:
    import math

    import numpy as np
    import warp as wp
    from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
    from app.runtime.common.motion import LocomotionCommand
    from app.runtime.common.rig import ASSETS_ROOT, load_rig_assets, model_to_runtime
    from app.runtime.soma.character import SOMACharacter
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.model.contract import ModelRig
    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine

    wp.config.kernel_cache_dir = f"{CACHE}/warp"
    wp.init()
    assets = load_rig_assets(ModelRig.SOMA)
    asset_pool = SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)
    reference = ASSETS_ROOT / "smart_primitives/soma/jump_0_75m.smart_primitive.json"
    asset_pool.load(reference)
    engine = MotionBricksSOMAEngine(
        SOMA_N1_FP32_ONNX_PATH,
        model_to_runtime=model_to_runtime(ModelRig.SOMA),
        build_options=TensorRTBuildOptions(enable_fp16=False),
        cache_dir=f"{CACHE}/trt",
    )
    volume.commit()
    skeleton = Skeleton(assets.clip_registry.skeleton)
    Leap = make_leap_type()
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    move = forward * SPEED_STRENGTH
    results, tracks = [], {}
    with wp.ScopedStream(engine.stream):
        for place in PLACE_AHEAD:
            for stretch in STRETCH:
                for repeat in range(REPEATS):
                    character = SOMACharacter(engine, asset_pool=asset_pool, initial_transform=wp.transform_identity())
                    command = LocomotionCommand(forward, move, use_max_speed=True)
                    for _ in range(int(2.5 * FPS)):
                        character.step(DT, command)
                    speed_before = float(np.linalg.norm(character.actual_velocity[:2]))
                    leap = asset_pool.instantiate(reference, Leap)
                    leap.place_ahead, leap.stretch = place, stretch
                    position = character.character_ground_position
                    leap.set_node_world_transform(
                        0,
                        wp.transform(
                            wp.vec3(float(position[0]) + place, float(position[1]), 0.0),
                            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 2),
                        ),
                        stream=engine.stream,
                    )
                    character.state_machine.start_interaction(character, leap)
                    feet, frames, finished = [], [], None
                    for frame in range(int(3.5 * FPS)):
                        character.step(DT, command)
                        joints = skeleton.world(character.current_pose.numpy())
                        feet.append([[joints[n][0], joints[n][2]] for n in FEET])
                        frames.append([joints[n] for n in RECORDED])
                        if finished is None and not character.state_machine.is_interacting:
                            finished = (frame + 1) * DT
                    feet, frames = np.array(feet), np.array(frames)
                    airborne = feet[:, :, 1].min(axis=1) > AIRBORNE_HEIGHT
                    runs, begin = [], None
                    for f, up in enumerate([*airborne, False]):
                        if up and begin is None:
                            begin = f
                        elif not up and begin is not None:
                            runs.append((begin, f - 1))
                            begin = None
                    row = {
                        "place": place, "stretch": stretch, "repeat": repeat,
                        "speedBefore": speed_before, "finished": finished,
                        "peakJointStep": float(np.linalg.norm(np.diff(frames - frames[:, :1], axis=0), axis=2).max() * FPS),
                        "lowestHead": float(frames[:, RECORDED.index("HeadEnd"), 2].min()),
                        "speedAfter": float(np.linalg.norm(np.diff(frames[-20:, 0, :2], axis=0), axis=1).mean() * FPS),
                    }
                    if runs:
                        a, b = max(runs, key=lambda r: r[1] - r[0])
                        row.update(
                            lead=a * DT, air=(b - a + 1) * DT,
                            gap=float(feet[min(b + 1, len(feet) - 1), :, 0].min() - feet[max(a - 1, 0), :, 0].max()),
                            peakFeet=float(feet[a:b + 1, :, 1].min(axis=1).max()),
                        )
                    else:
                        row.update(lead=None, air=0.0, gap=0.0, peakFeet=0.0)
                    results.append(row)
                    if repeat == 0:
                        tracks[f"place{place}_stretch{stretch}"] = frames.astype(np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, joints=np.array(RECORDED), **tracks)
    return results, buffer.getvalue()


@app.local_entrypoint()
def main():
    results, data = measure.remote()
    runs = Path(__file__).resolve().parent.parent / "runs"
    (runs / "leaps.npz").write_bytes(data)
    (runs / "leaps.json").write_text(json.dumps(results))
    print("place stretch  gap   air  lead  feetTop  done  headLow  speedAfter  peakJointStep")
    for r in results:
        print(
            f"{r['place']:5} {r['stretch']:7} {r['gap']:5.2f} {r['air']:5.2f} {r['lead'] if r['lead'] is None else round(r['lead'], 2)!s:>5} "
            f"{r['peakFeet']:7.2f} {r['finished'] if r['finished'] is None else round(r['finished'], 2)!s:>5} "
            f"{r['lowestHead']:7.2f} {r['speedAfter']:9.2f} {r['peakJointStep']:9.1f}"
        )
