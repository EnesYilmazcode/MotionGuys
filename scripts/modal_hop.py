"""Hops from a standstill, and from a slow walk: how far do they really go, at what angle to the body, and how long do they take?

Run:  modal run scripts/modal_hop.py
Writes runs/hops.json. The same steps as the real round: walk, freeze in the last generated pose, then start the leap.
"""

import json
import math
from pathlib import Path

import modal
from modal_leap import make_leap_type
from modal_race import CACHE, DT, FEET, FPS, Skeleton, image, volume

PLACES = (1.0, 1.6, 2.2)
STRETCHES = (0.6, 1.2, 1.8, 2.4)
ANGLES = (0, 45, 90, 135)
REPEATS = 2
WALK_SECONDS = 2.5
FREEZE_SECONDS = 0.6
WATCH_SECONDS = 2.4
AIRBORNE_HEIGHT = 0.15

app = modal.App("motionguys-hop", image=image.add_local_python_source("modal_race", "modal_leap"))


@app.function(gpu="A10", volumes={CACHE: volume}, timeout=2400)
def measure() -> list[dict]:
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
    skeleton = Skeleton(assets.clip_registry.skeleton)
    Leap = make_leap_type()
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rows = []
    with wp.ScopedStream(engine.stream):
        for mode in ("standing", "walking"):
            for place in PLACES:
                for stretch in STRETCHES:
                    for angle in ANGLES if mode == "standing" else (0, 45):
                        for repeat in range(REPEATS):
                            character = SOMACharacter(engine, asset_pool=asset_pool, initial_transform=wp.transform_identity())
                            walk = LocomotionCommand(forward, forward * 0.2, use_max_speed=True)
                            for _ in range(int(WALK_SECONDS * FPS)):
                                character.step(DT, walk)
                            if mode == "standing":
                                for _ in range(int(FREEZE_SECONDS * FPS)):
                                    character.actual_velocity[:] = 0.0
                                    character.player.record_pose(DT, character.current_pose)
                                character.state_machine.active_locomotion_state._force_regeneration = True
                            radians = math.radians(angle)
                            direction = np.array([math.cos(radians), math.sin(radians)])
                            start = np.array(skeleton.world(character.current_pose.numpy())["Hips"][:2], dtype=float)
                            position = character.character_ground_position
                            leap = asset_pool.instantiate(reference, Leap)
                            leap.place_ahead, leap.stretch, leap.direction = place, stretch, direction
                            yaw = radians - math.pi / 2
                            leap.set_node_world_transform(
                                0,
                                wp.transform(
                                    wp.vec3(float(position[0] + direction[0] * place), float(position[1] + direction[1] * place), 0.0),
                                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
                                ),
                                stream=engine.stream,
                            )
                            character.state_machine.start_interaction(character, leap)
                            command = LocomotionCommand(np.array([direction[0], direction[1], 0.0], dtype=np.float32), forward * 0.0 + np.array([direction[0], direction[1], 0.0], dtype=np.float32) * 0.2, use_max_speed=True)
                            hips, lowest, peak = [], [], 0.0
                            first_air, touch = None, None
                            airborne = False
                            for frame in range(int(WATCH_SECONDS * FPS)):
                                character.step(DT, command)
                                joints = skeleton.world(character.current_pose.numpy())
                                low = min(float(joints[name][2]) for name in FEET)
                                lowest.append(low)
                                hips.append(joints["Hips"][:2].astype(float))
                                if low >= AIRBORNE_HEIGHT:
                                    if first_air is None:
                                        first_air = frame * DT
                                    airborne = True
                                elif airborne and touch is None:
                                    touch = frame
                            hips = np.array(hips)
                            at = touch if touch is not None else len(hips) - 1
                            travel = hips[at] - start
                            rows.append({
                                "mode": mode, "place": place, "stretch": stretch, "angle": angle, "repeat": repeat,
                                "distance": float(np.linalg.norm(travel)),
                                "offAngle": float(abs(math.degrees(math.atan2(travel[1], travel[0])) - angle)),
                                "takeoff": first_air, "touch": None if touch is None else touch * DT,
                                "air": None if touch is None or first_air is None else touch * DT - first_air,
                                "peakFeet": float(max(lowest)),
                                "speedAfter": float(np.linalg.norm(hips[min(at + 9, len(hips) - 1)] - hips[at]) / (min(9, len(hips) - 1 - at) * DT + 1e-6)),
                            })
    volume.commit()
    return rows


@app.local_entrypoint()
def main():
    rows = measure.remote()
    (Path(__file__).resolve().parent.parent / "runs" / "hops.json").write_text(json.dumps(rows))
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], r["place"], r["stretch"], r["angle"]), []).append(r)
    print("mode place stretch angle  dist  off  takeoff touch  air  feet  after")
    mean = lambda v, k: sum((x[k] if x[k] is not None else float("nan")) for x in v) / len(v)
    for key, v in sorted(groups.items()):
        print(*key, *(round(mean(v, k), 2) for k in ("distance", "offAngle", "takeoff", "touch", "air", "peakFeet", "speedAfter")))
