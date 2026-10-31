"""Generate SOMA run, jump, and duck clips headlessly with MotionBricks on Modal.

Run:  modal run scripts/modal_clip.py

Drives the upstream character runtime without its viewer, one runner per
obstacle in the official gym scene, then times 1, 2, and 8 runners sharing an
engine. Writes runs/gym_clips.npz.
"""

import io
import json
import time
from pathlib import Path

import modal

COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"
REPO = "/opt/motionbricks"
CACHE = "/cache"
FPS = 30
LANE_SECONDS = 9.0
BENCH_FRAMES = 300
START_X = -4.0
LANES = ("SmallJumpSP", "MidJumpSP", "AvoidanceSP")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "libgl1", "libegl1", "libglu1-mesa", "libx11-6")
    .run_commands(
        f"git init {REPO}",
        f"git -C {REPO} remote add origin https://github.com/NVlabs/motionbricks",
        f"GIT_LFS_SKIP_SMUDGE=1 git -C {REPO} fetch --depth 1 origin {COMMIT}",
        f"GIT_LFS_SKIP_SMUDGE=1 git -C {REPO} checkout FETCH_HEAD",
        f"git -C {REPO} lfs pull --include 'motionbricks/assets/human/**,"
        "motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx,assets/**'",
        f"pip install -e {REPO}",
    )
    .env({"PYTHONPATH": REPO})
)

app = modal.App("motionguys", image=image)
volume = modal.Volume.from_name("motionguys-cache", create_if_missing=True)


def make_lane_runner(engine, asset_pool, scene, name):
    import numpy as np
    import warp as wp
    from app.runtime.common.smart_primitive_behaviors import behavior_types
    from app.runtime.soma.character import SOMACharacter

    definition = next(d for d in scene.smart_primitives if d.name == name)
    y = float(definition.transform.p[1])
    character = SOMACharacter(
        engine,
        asset_pool=asset_pool,
        initial_transform=wp.transform(wp.vec3(START_X, y, 0.0), wp.quat_identity()),
    )
    runtime = asset_pool.instantiate(definition.reference, behavior_types()[definition.behavior])
    runtime.set_node_world_transform(0, definition.transform, stream=engine.stream)
    character.state_machine.interaction_state.prepare(character, runtime)
    count = runtime.interaction_box_count
    from motionbricks.runtime.smart_primitive import InteractionBox

    return {
        "name": name,
        "character": character,
        "runtime": runtime,
        "boxes": wp.empty(count, dtype=InteractionBox, device=engine.device),
        "overlaps": wp.empty(count, dtype=wp.int32, device=engine.device),
        "started_frame": -1,
        "finished_frame": -1,
        "forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def step_runner(runner, frame, engine):
    from app.runtime.common.motion import LocomotionCommand
    from motionbricks.runtime.smart_primitive import compute_interaction_box_overlaps

    character = runner["character"]
    machine = character.state_machine
    character.step(
        1.0 / FPS,
        LocomotionCommand(
            facing_direction=runner["forward"],
            movement=runner["forward"],
            use_max_speed=True,
        ),
    )
    if runner["started_frame"] < 0:
        runner["runtime"].get_world_interaction_boxes(runner["boxes"], stream=engine.stream)
        compute_interaction_box_overlaps(
            runner["boxes"], character.probe_box, runner["overlaps"], engine.stream
        )
        if runner["overlaps"].numpy().any():
            machine.start_interaction(character, runner["runtime"])
            runner["started_frame"] = frame
    elif runner["finished_frame"] < 0 and not machine.is_interacting:
        runner["finished_frame"] = frame


@app.function(gpu="L4", volumes={CACHE: volume}, timeout=1800)
def generate() -> bytes:
    import numpy as np
    import warp as wp
    from app.runtime.common.io.scene_document import load_scene_document
    from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
    from app.runtime.common.rig import load_rig_assets, model_to_runtime
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine

    wp.init()
    scene, _, _ = load_scene_document(f"{REPO}/assets/scenes/soma_gym.scene.json")
    scene = scene.flatten()
    assets = load_rig_assets(scene.rig)
    asset_pool = SmartPrimitiveAssetPool(
        assets.clip_registry, assets.reference_pose, assets.model_to_runtime
    )
    for definition in scene.smart_primitives:
        asset_pool.load(definition.reference)

    t = time.perf_counter()
    engine = MotionBricksSOMAEngine(
        SOMA_N1_FP32_ONNX_PATH,
        model_to_runtime=model_to_runtime(scene.rig),
        build_options=TensorRTBuildOptions(enable_fp16=False),
        cache_dir=f"{CACHE}/trt",
    )
    volume.commit()
    load_seconds = time.perf_counter() - t

    skeleton = assets.clip_registry.skeleton
    out = {
        "joint_names": np.array(skeleton.joint_names),
        "parents": np.array([skeleton.joint_parent(i) for i in range(skeleton.num_joints)]),
        "fps": np.array(FPS),
    }
    report = {"gpu": wp.get_device().name, "commit": COMMIT, "load_seconds": load_seconds, "lanes": {}}
    with wp.ScopedStream(engine.stream):
        for name in LANES:
            runner = make_lane_runner(engine, asset_pool, scene, name)
            frames = []
            for frame in range(int(LANE_SECONDS * FPS)):
                step_runner(runner, frame, engine)
                frames.append(runner["character"].current_pose.numpy().copy())
            out[f"poses_{name}"] = np.stack(frames)
            if runner["started_frame"] < 0:
                raise RuntimeError(f"{name}: the runner never triggered its interaction")
            if runner["finished_frame"] < 0:
                raise RuntimeError(f"{name}: the interaction did not finish in {LANE_SECONDS} s")
            report["lanes"][name] = {
                "started_s": runner["started_frame"] / FPS,
                "finished_s": runner["finished_frame"] / FPS,
            }

        report["bench"] = {}
        for count in (1, 2, 8):
            runners = [
                make_lane_runner(engine, asset_pool, scene, LANES[i % len(LANES)])
                for i in range(count)
            ]
            times = []
            for frame in range(BENCH_FRAMES):
                t = time.perf_counter()
                for runner in runners:
                    step_runner(runner, frame, engine)
                engine.synchronize()
                times.append(time.perf_counter() - t)
            times_ms = np.array(times[30:]) * 1000.0
            report["bench"][count] = {
                "mean_ms": float(times_ms.mean()),
                "p95_ms": float(np.percentile(times_ms, 95)),
                "max_ms": float(times_ms.max()),
            }

    geometry = [
        {
            "name": cube.name,
            "position": list(map(float, cube.transform.p)),
            "rotation": list(map(float, cube.transform.q)),
            "dimensions": [float(v) for v in (cube.width, cube.depth, cube.height)],
        }
        for cube in scene.solid_cubes
    ]
    primitives = [
        {"name": d.name, "position": list(map(float, d.transform.p))} for d in scene.smart_primitives
    ]
    out["meta"] = np.array(json.dumps({"report": report, "cubes": geometry, "primitives": primitives}))
    volume.commit()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **out)
    return buffer.getvalue()


@app.local_entrypoint()
def main():
    data = generate.remote()
    path = Path(__file__).resolve().parent.parent / "runs" / "gym_clips.npz"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    print(f"wrote {path} ({len(data) / 1e6:.1f} MB)")
