"""Can a robot slow down or stop? Measure what MotionBricks does at low command strengths, without playing a sample clip.

Run:  modal run scripts/modal_stop.py
Writes runs/stops.json: the speed each strength settles at, how long a stop takes, whether a clip played, and how fast it gets going again.
"""

import json
from pathlib import Path

import modal
from modal_race import CACHE, DT, FEET, FPS, Skeleton, image, volume

STARTS = {"walk": 0.6, "jog": 0.7}
STRENGTHS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
REPEATS = 2
SETTLE_SECONDS = 3.0
HOLD_SECONDS = 4.0
RESUME_SECONDS = 2.0

app = modal.App("motionguys-stop", image=image.add_local_python_source("modal_race"))


@app.function(gpu="A10", volumes={CACHE: volume}, timeout=1500)
def measure() -> list[dict]:
    import numpy as np
    import warp as wp
    from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
    from app.runtime.common.motion import LocomotionCommand, PlaybackSource
    from app.runtime.common.rig import load_rig_assets, model_to_runtime
    from app.runtime.soma.character import SOMACharacter
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.model.contract import ModelRig
    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine

    wp.config.kernel_cache_dir = f"{CACHE}/warp"
    wp.init()
    assets = load_rig_assets(ModelRig.SOMA)
    asset_pool = SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)
    engine = MotionBricksSOMAEngine(
        SOMA_N1_FP32_ONNX_PATH,
        model_to_runtime=model_to_runtime(ModelRig.SOMA),
        build_options=TensorRTBuildOptions(enable_fp16=False),
        cache_dir=f"{CACHE}/trt",
    )
    skeleton = Skeleton(assets.clip_registry.skeleton)
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rows = []
    with wp.ScopedStream(engine.stream):
        for start, start_strength in STARTS.items():
            for strength in STRENGTHS:
                for repeat in range(REPEATS):
                    character = SOMACharacter(engine, asset_pool=asset_pool, initial_transform=wp.transform_identity())
                    command = LocomotionCommand(forward, forward * start_strength, use_max_speed=True)
                    for _ in range(int(SETTLE_SECONDS * FPS)):
                        character.step(DT, command)
                    # Direction stays set and only the strength falls, so the input is never exactly zero unless asked.
                    command = LocomotionCommand(forward, forward * strength, use_max_speed=strength > 0)
                    speeds, clip_frames, hips, feet_moved = [], 0, [], []
                    previous = None
                    for _ in range(int(HOLD_SECONDS * FPS)):
                        character.step(DT, command)
                        clip_frames += character._playback_source is PlaybackSource.CLIP
                        joints = skeleton.world(character.current_pose.numpy())
                        hips.append(joints["Hips"][:2].astype(float))
                        feet = {name: joints[name] for name in FEET}
                        if previous is not None:
                            feet_moved.append(max(float(np.linalg.norm((feet[n] - previous[n])[:2])) for n in FEET) * FPS)
                        previous = feet
                    hips = np.array(hips)
                    velocity = np.linalg.norm(np.diff(hips, axis=0), axis=1) * FPS
                    smooth = np.convolve(velocity, np.ones(9) / 9, mode="valid")
                    below = np.nonzero(smooth < 0.3)[0]
                    command = LocomotionCommand(forward, forward * start_strength, use_max_speed=True)
                    resumed = []
                    for _ in range(int(RESUME_SECONDS * FPS)):
                        character.step(DT, command)
                        clip_frames += character._playback_source is PlaybackSource.CLIP
                        resumed.append(skeleton.world(character.current_pose.numpy())["Hips"][:2].astype(float))
                    resumed = np.array(resumed)
                    again = np.convolve(np.linalg.norm(np.diff(resumed, axis=0), axis=1) * FPS, np.ones(9) / 9, mode="valid")
                    going = np.nonzero(again > 2.0)[0]
                    rows.append({
                        "start": start, "strength": strength, "repeat": repeat,
                        "settledSpeed": float(smooth[-30:].mean()), "minSpeed": float(smooth.min()),
                        "secondsBelow03": None if not len(below) else (int(below[0]) + 4) * DT,
                        "metersToSettle": float(np.linalg.norm(hips[min(45, len(hips) - 1)] - hips[0])),
                        "footSpeedP95Settled": float(np.percentile(feet_moved[-60:], 95)),
                        "clipFrames": int(clip_frames),
                        "secondsToResume": None if not len(going) else (int(going[0]) + 4) * DT,
                    })
    volume.commit()
    return rows


@app.local_entrypoint()
def main():
    rows = measure.remote()
    (Path(__file__).resolve().parent.parent / "runs" / "stops.json").write_text(json.dumps(rows))
    print("start strength settled minspeed secBelow0.3 m_in_1.5s footP95 clipFrames secResume")
    for r in rows:
        print(r["start"], r["strength"], round(r["settledSpeed"], 2), round(r["minSpeed"], 2), r["secondsBelow03"], round(r["metersToSettle"], 2), round(r["footSpeedP95Settled"], 2), r["clipFrames"], r["secondsToResume"])
