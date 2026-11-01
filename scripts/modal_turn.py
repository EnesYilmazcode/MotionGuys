"""Measure how MotionBricks answers a change of direction, so the training sim can copy it.

Run:  modal run scripts/modal_turn.py
Writes runs/turns.json: for each gait and turn angle, how long the velocity takes to come round, the speed lost, and how far the
body keeps travelling the old way. Also the footfall rhythm and how far feet land from the hips.
"""

import json
import math
from pathlib import Path

import modal
from modal_race import CACHE, DT, FEET, FPS, Skeleton, image, volume

GAITS = {"walk": 0.6, "jog": 0.7, "sprint": 1.0}
ANGLES = (30, 60, 90, 120, 150, 180)
REPEATS = 3
SETTLE_SECONDS = 3.0
WATCH_SECONDS = 3.0

app = modal.App("motionguys-turn", image=image.add_local_python_source("modal_race"))


@app.function(gpu="A10", volumes={CACHE: volume}, timeout=1500)
def measure() -> dict:
    import numpy as np
    import warp as wp
    from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
    from app.runtime.common.motion import LocomotionCommand
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
        build_options=__import__("motionbricks.inference.builder", fromlist=["x"]).TensorRTBuildOptions(enable_fp16=False),
        cache_dir=f"{CACHE}/trt",
    )
    skeleton = Skeleton(assets.clip_registry.skeleton)
    rows, rhythm = [], []
    with wp.ScopedStream(engine.stream):
        for gait, strength in GAITS.items():
            for angle in ANGLES:
                for repeat in range(REPEATS):
                    character = SOMACharacter(engine, asset_pool=asset_pool, initial_transform=wp.transform_identity())
                    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    command = LocomotionCommand(forward, forward * strength, use_max_speed=True)
                    feet_down, spreads = {name: False for name in FEET}, []
                    last_down = None
                    gaps = []
                    for _ in range(int(SETTLE_SECONDS * FPS)):
                        character.step(DT, command)
                        joints = skeleton.world(character.current_pose.numpy())
                        for name in FEET:
                            down = joints[name][2] < 0.08
                            if down and not feet_down[name]:
                                spreads.append(float(np.linalg.norm((joints[name] - joints["Hips"])[:2])))
                                now = character.character_ground_position[0]
                                if last_down is not None:
                                    gaps.append(1)
                                last_down = now
                            feet_down[name] = down
                    start = np.array(character.character_ground_position[:2], dtype=float)
                    speed_before = float(np.linalg.norm(character.actual_velocity[:2]))
                    radians = math.radians(angle)
                    new = np.array([math.cos(radians), math.sin(radians), 0.0], dtype=np.float32)
                    command = LocomotionCommand(new, new * strength, use_max_speed=True)
                    track = []
                    for frame in range(int(WATCH_SECONDS * FPS)):
                        character.step(DT, command)
                        joints = skeleton.world(character.current_pose.numpy())
                        track.append(joints["Hips"][:2].astype(float))
                    track = np.array(track)
                    velocity = np.diff(track, axis=0) * FPS
                    # Smooth over a stride so the sway of the hips does not read as turning.
                    width = 9
                    kernel = np.ones(width) / width
                    smooth = np.stack([np.convolve(velocity[:, k], kernel, mode="valid") for k in range(2)], axis=1)
                    along = smooth @ new[:2]
                    speed = np.linalg.norm(smooth, axis=1)
                    aligned = along / np.maximum(speed, 1e-6)
                    ninety = int(np.argmax(aligned > math.cos(math.radians(15)))) if (aligned > math.cos(math.radians(15))).any() else None
                    old_way = max(float((p - start) @ np.array([1.0, 0.0])) for p in track)
                    rows.append({
                        "gait": gait, "angle": angle, "repeat": repeat, "speedBefore": speed_before,
                        "secondsToHeading": None if ninety is None else (ninety + width // 2) * DT,
                        "minSpeed": float(speed.min()), "endSpeed": float(speed[-15:].mean()),
                        "overshoot": old_way, "sideways": float(np.max(np.abs((track - start) @ np.array([-math.sin(radians), math.cos(radians)])))) ,
                    })
                    rhythm.append({"gait": gait, "footSpreadP50": float(np.median(spreads)) if spreads else None, "footSpreadP95": float(np.percentile(spreads, 95)) if spreads else None, "footfalls": len(spreads), "seconds": SETTLE_SECONDS})
    volume.commit()
    return {"turns": rows, "rhythm": rhythm}


@app.local_entrypoint()
def main():
    result = measure.remote()
    out = Path(__file__).resolve().parent.parent / "runs" / "turns.json"
    out.write_text(json.dumps(result))
    print("gait   angle  sec_to_heading  min_speed  end_speed  overshoot")
    groups = {}
    for row in result["turns"]:
        groups.setdefault((row["gait"], row["angle"]), []).append(row)
    for (gait, angle), rows in groups.items():
        mean = lambda key: sum((r[key] or 9.0) for r in rows) / len(rows)
        print(f"{gait:6} {angle:5} {mean('secondsToHeading'):14.2f} {mean('minSpeed'):10.2f} {mean('endSpeed'):10.2f} {mean('overshoot'):10.2f}")
    for row in result["rhythm"][::len(result["rhythm"]) // 3]:
        print(row)
