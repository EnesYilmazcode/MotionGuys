"""Eight MotionBricks runners on stacked floors that drop away under their feet.

Run:  modal run scripts/modal_spleef.py --seeds 1,2,3
Try:  modal run scripts/modal_spleef.py --seconds 15     a quick look, no winner needed

Last one standing wins. Falling lands a runner on the floor below, and falling
past the bottom floor eliminates them. Runners always move, so they never stop
to think. Writes runs/spleef/seed_N.json and seed_N.bin.
"""

import json
import math
import time
from pathlib import Path

import modal
from modal_leap import make_leap_type
from modal_race import (
    BODY, CACHE, COLORS, COMMIT, DT, FEET, FPS, NAMES, RECORDED, RUNNERS, Skeleton, axis_angle, image, quat_mul, volume,
)

LAYERS = 3
LAYER_HEIGHT = 6.0
TILE = 1.25
TILES = 16
HALF = TILE * TILES / 2
DROP_SECONDS = 1.0
STEP_HEIGHT = 0.08
MAX_SECONDS = 150.0
PREROLL_SECONDS = 2.0
SPAWN_RADIUS = 8.5
THINK_FRAMES = 18
LIE_SECONDS = 0.7
NO_TRIGGER_SECONDS = 2.0
WIN_RUNOFF_SECONDS = 1.0
LEAP_PLACE = 2.2
LEAP_STRETCH = 2.4
LEAP_SECONDS = 1.8
AIRBORNE_HEIGHT = 0.15
# Measured in rounds from a commit at running speed: the rear foot leaves near 1.6 m, and hips touch down at 3.2 m to 4.7 m.
TAKEOFF_CHECKS = (0.0, 0.4, 0.8, 1.2, 1.6)
HOLE_CHECKS = (2.0, 2.4, 2.8)
LANDING_CHECKS = (3.4, 3.8, 4.2, 4.6)
LANDING_WIDTH = 0.25
LEAP_LOOK_FRAMES = 4
LEAP_CONE = 0.7
LEAP_COOLDOWN_SECONDS = 0.8
GEAR = ("cap", "cone", "bun", "brim", "cap", "cone", "bun", "brim")

# A sibling module is not mounted on its own, and the container would crash on import.
app = modal.App("motionguys-spleef", image=image.add_local_python_source("modal_race", "modal_leap"))


class Floors:
    """Which tiles are shaking or gone, and everything that happened."""

    def __init__(self):
        self.time = 0.0
        self.gone = [set() for _ in range(LAYERS)]
        self.shaking = [{} for _ in range(LAYERS)]
        self.events = []

    def log(self, kind, runner=None, **detail):
        self.events.append({"t": round(self.time, 3), "type": kind, "runner": runner, **detail})

    @staticmethod
    def tile_at(x, y):
        column, row = math.floor((x + HALF) / TILE), math.floor((y + HALF) / TILE)
        return (column, row) if 0 <= column < TILES and 0 <= row < TILES else None

    def has_floor(self, layer, x, y):
        tile = self.tile_at(x, y)
        return tile is not None and tile not in self.gone[layer]

    def safe(self, layer, x, y):
        tile = self.tile_at(x, y)
        return tile is not None and tile not in self.gone[layer] and tile not in self.shaking[layer]

    def step_on(self, layer, x, y, runner):
        tile = self.tile_at(x, y)
        if tile is None or tile in self.gone[layer] or tile in self.shaking[layer]:
            return
        self.shaking[layer][tile] = self.time
        self.log("tile_shake", runner, layer=layer, column=tile[0], row=tile[1])

    def update(self):
        for layer, shaking in enumerate(self.shaking):
            for tile, started in list(shaking.items()):
                if self.time - started >= DROP_SECONDS:
                    del shaking[tile]
                    self.gone[layer].add(tile)
                    self.log("tile_drop", None, layer=layer, column=tile[0], row=tile[1])


class Runner:
    def __init__(self, index, character, sight, noise, strength, bravery):
        self.index = index
        self.character = character
        self.sight = sight
        self.noise = noise
        self.strength = strength
        self.bravery = bravery
        self.leaping_until = -1.0
        self.leap_ready_at = 0.0
        self.leap_from = None
        self.leap_airborne = False
        self.leaps = 0
        self.leaps_landed = 0
        self.leaps_failed = 0
        self.layer = 0
        self.direction = None
        self.last_turn = -100
        self.speeds = []
        self.fall = None
        self.no_trigger_until = 0.0
        self.pending_getup = None
        self.getup_seconds = []
        self.falls = 0
        self.eliminated_at = None
        self.tiles_broken = 0
        self.slides = []
        self.previous_feet = None


def choose_direction(floors, runner, x, y, heading, rng, others):
    """Point at the longest safe run, with this runner's limited sight and judgment."""
    import numpy as np

    best, best_score = heading, -1e9
    for k in range(16):
        angle = 2 * math.pi * k / 16
        direction = np.array([math.cos(angle), math.sin(angle)])
        run = 0.0
        while run < runner.sight and floors.safe(runner.layer, x + direction[0] * (run + 0.5), y + direction[1] * (run + 0.5)):
            run += 0.5
        ahead_x, ahead_y = x + direction[0] * runner.sight * 0.7, y + direction[1] * runner.sight * 0.7
        around = sum(
            floors.safe(runner.layer, ahead_x + dx, ahead_y + dy)
            for dx in (-TILE * 1.5, 0.0, TILE * 1.5)
            for dy in (-TILE * 1.5, 0.0, TILE * 1.5)
        )
        crowd = sum(1 for px, py in others if math.hypot(ahead_x - px, ahead_y - py) < 1.5)
        score = run + 0.25 * around + 1.5 * float(direction @ heading) - 0.8 * crowd + rng.normal(0.0, runner.noise)
        if score > best_score:
            best, best_score = direction, score
    return best


def choose_leap(floors, runner, x, y, rng):
    """A direction near the current heading with solid takeoff and landing, preferring one that crosses a hole."""
    import numpy as np

    crossing, plain = [], []
    for k in range(24):
        angle = 2 * math.pi * k / 24
        direction = np.array([math.cos(angle), math.sin(angle)])
        # A leap at a sharp angle to the run lands somewhere much less predictable.
        if float(direction @ runner.direction) < LEAP_CONE:
            continue

        def clear(d, side=0.0):
            return floors.safe(runner.layer, x + direction[0] * d - direction[1] * side, y + direction[1] * d + direction[0] * side)

        def present(d):
            # The tile underfoot is always shaking, because the runner is standing on it.
            return floors.has_floor(runner.layer, x + direction[0] * d, y + direction[1] * d)

        landing = all(clear(d, side) for d in LANDING_CHECKS for side in (-LANDING_WIDTH, 0.0, LANDING_WIDTH))
        if landing and all(present(d) for d in TAKEOFF_CHECKS):
            (plain if all(clear(d) for d in HOLE_CHECKS) else crossing).append(direction)
    if crossing:
        return crossing[int(rng.integers(len(crossing)))], True
    if plain and rng.random() < runner.bravery / 4:
        return plain[int(rng.integers(len(plain)))], False
    return None, False


def start_leap(floors, runner, direction, crosses, engine, asset_pool, leap_reference, Leap):
    import numpy as np
    import warp as wp

    character = runner.character
    position = character.character_ground_position
    leap = asset_pool.instantiate(leap_reference, Leap)
    leap.place_ahead, leap.stretch, leap.direction = LEAP_PLACE, LEAP_STRETCH, direction
    yaw = math.atan2(direction[1], direction[0]) - math.pi / 2
    leap.set_node_world_transform(
        0,
        wp.transform(
            wp.vec3(float(position[0] + direction[0] * LEAP_PLACE), float(position[1] + direction[1] * LEAP_PLACE), 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
        ),
        stream=engine.stream,
    )
    character.state_machine.start_interaction(character, leap)
    runner.direction = direction
    runner.leaping_until = floors.time + LEAP_SECONDS
    runner.leaps += 1
    speed = float(np.linalg.norm(character.actual_velocity[:2]))
    runner.leap_from = (float(position[0]), float(position[1]), floors.time)
    runner.leap_airborne = False
    floors.log("leap", runner.index, layer=runner.layer, speed=round(speed, 2), crosses=crosses)


def begin_fall(floors, runner, pose, cause):
    import numpy as np

    character = runner.character
    if character.state_machine.is_interacting:
        character.state_machine.transition_to_locomotion(character)
    velocity = np.array([character.actual_velocity[0], character.actual_velocity[1], 0.0], dtype=np.float64) * 0.5
    length = np.linalg.norm(velocity[:2])
    direction = velocity / length if length > 1e-6 else np.array([1.0, 0.0, 0.0])
    runner.fall = {
        "position": pose[0, :3].astype(np.float64),
        "velocity": velocity,
        "rotation": pose[0, 3:].astype(np.float64),
        "axis": np.array([-direction[1], direction[0], 0.0]),
        "angle": 0.0,
        "pose": pose.copy(),
        "lying": None,
    }
    runner.falls += 1
    floors.log("fall", runner.index, layer=runner.layer, cause=cause)


def step_fall(floors, runner):
    """A scripted tumble. MotionBricks only takes over again for the get-up."""
    import numpy as np

    fall, character = runner.fall, runner.character
    if fall["lying"] is None:
        fall["velocity"][2] -= 9.81 * DT
        fall["position"] += fall["velocity"] * DT
        fall["angle"] = min(fall["angle"] + 3.2 * DT, math.pi / 2 if runner.layer < LAYERS - 1 else math.inf)
        if fall["position"][2] <= 0.22 - LAYER_HEIGHT:
            if runner.layer == LAYERS - 1:
                if fall["position"][2] < -LAYER_HEIGHT - 6.0:
                    runner.eliminated_at = floors.time
                    floors.log("eliminated", runner.index)
                    return
            else:
                runner.layer += 1
                fall["position"][2] += LAYER_HEIGHT
                if floors.has_floor(runner.layer, fall["position"][0], fall["position"][1]):
                    fall["position"][2] = 0.22
                    fall["velocity"][:] = 0.0
                    fall["angle"] = math.pi / 2
                    fall["lying"] = floors.time
                    floors.log("land", runner.index, layer=runner.layer)
    elif floors.time - fall["lying"] >= LIE_SECONDS:
        runner.fall = None
        runner.pending_getup = floors.time
        runner.no_trigger_until = floors.time + NO_TRIGGER_SECONDS
        character.state_machine.active_locomotion_state._force_regeneration = True
        floors.log("recover", runner.index)
        return
    pose = fall["pose"].copy()
    pose[0, :3] = fall["position"]
    pose[0, 3:] = quat_mul(axis_angle(fall["axis"], fall["angle"]), fall["rotation"])
    character.current_pose.assign(pose)
    character.player.record_pose(DT, character.current_pose)
    character.character_ground_position[:2] = fall["position"][:2]
    character.actual_velocity[:] = 0.0


def simulate(engine, asset_pool, skeleton, seed, seconds, leap_reference):
    import numpy as np
    import warp as wp
    from app.runtime.common.motion import LocomotionCommand, PlaybackSource
    from app.runtime.soma.character import SOMACharacter

    Leap = make_leap_type()
    rng = np.random.default_rng(seed)
    floors = Floors()
    order = rng.permutation(RUNNERS)
    runners = []
    for index in range(RUNNERS):
        angle = 2 * math.pi * int(order[index]) / RUNNERS
        x, y = SPAWN_RADIUS * math.cos(angle), SPAWN_RADIUS * math.sin(angle)
        character = SOMACharacter(
            engine, asset_pool=asset_pool,
            initial_transform=wp.transform(
                wp.vec3(x, y, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle + math.pi)
            ),
        )
        runner = Runner(
            index, character,
            sight=float(rng.uniform(1.5, 4.5)),
            noise=float(rng.uniform(0.1, 0.8)),
            strength=float(rng.uniform(0.62, 0.8)),
            bravery=float(rng.uniform(0.15, 0.6)),
        )
        runner.direction = np.array([-math.cos(angle), -math.sin(angle)])
        runners.append(runner)
    for _ in range(int(PREROLL_SECONDS * FPS)):
        for runner in runners:
            move = np.array([runner.direction[0], runner.direction[1], 0.0], dtype=np.float32)
            runner.character.step(DT, LocomotionCommand(move, move * runner.strength, use_max_speed=True))

    hips_frames, joint_frames = [], []
    winner_at = None
    for frame in range(int(seconds * FPS)):
        floors.update()
        hips = np.full((RUNNERS, 3), np.nan, dtype=np.float32)
        relative = np.zeros((RUNNERS, len(RECORDED), 3), dtype=np.int16)
        alive = [r for r in runners if r.eliminated_at is None]
        for runner in alive:
            character = runner.character
            if runner.fall is not None:
                step_fall(floors, runner)
                if runner.eliminated_at is not None:
                    continue
            else:
                x, y = float(character.character_ground_position[0]), float(character.character_ground_position[1])
                blocked = not floors.safe(runner.layer, x + runner.direction[0] * 1.2, y + runner.direction[1] * 1.2)
                # Holding a direction matters: turning every few frames regenerates motion constantly and the runner barely moves.
                if not character.state_machine.is_interacting and frame % LEAP_LOOK_FRAMES == runner.index % LEAP_LOOK_FRAMES:
                    ready = (
                        floors.time >= runner.no_trigger_until
                        and floors.time >= runner.leap_ready_at
                        and float(np.linalg.norm(character.actual_velocity[:2])) > 1.5
                    )
                    if ready:
                        leap_direction, crosses = choose_leap(floors, runner, x, y, rng)
                        if leap_direction is not None:
                            start_leap(floors, runner, leap_direction, crosses, engine, asset_pool, leap_reference, Leap)
                            runner.leap_ready_at = floors.time + LEAP_SECONDS + LEAP_COOLDOWN_SECONDS
                if (blocked and frame - runner.last_turn >= 6) or frame - runner.last_turn >= THINK_FRAMES:
                    runner.last_turn = frame
                    others = [
                        (float(o.character.character_ground_position[0]), float(o.character.character_ground_position[1]))
                        for o in alive if o is not runner and o.layer == runner.layer
                    ]
                    if not character.state_machine.is_interacting:
                        runner.direction = choose_direction(floors, runner, x, y, runner.direction, rng, others)
                move = np.array([runner.direction[0], runner.direction[1], 0.0], dtype=np.float32)
                character.step(DT, LocomotionCommand(move, move * runner.strength, use_max_speed=True))
                if character._playback_source is PlaybackSource.CLIP:
                    raise RuntimeError("a sample clip played during the round")
            pose = character.current_pose.numpy().copy()
            if not np.isfinite(pose).all():
                raise RuntimeError(f"runner {runner.index}: non-finite pose at {floors.time:.2f} s")
            joints = skeleton.world(pose)
            if runner.fall is None:
                feet = {name: joints[name] for name in FEET}
                if floors.time >= runner.no_trigger_until:
                    for name, position in feet.items():
                        if position[2] < STEP_HEIGHT:
                            before = len(floors.events)
                            floors.step_on(runner.layer, position[0], position[1], runner.index)
                            runner.tiles_broken += len(floors.events) - before
                if runner.previous_feet is not None:
                    for name, position in feet.items():
                        if position[2] < 0.04 and runner.previous_feet[name][2] < 0.04:
                            runner.slides.append(float(np.linalg.norm((position - runner.previous_feet[name])[:2]) * FPS))
                runner.previous_feet = feet
                runner.speeds.append(float(np.linalg.norm(character.actual_velocity[:2])))
                root_x, root_y = joints["Hips"][:2]
                lowest_foot = min(float(position[2]) for position in feet.values())
                leaping = floors.time < runner.leaping_until
                # In the air a runner can cross a hole. Coming down over one is what makes a leap fail.
                grounded = lowest_foot < AIRBORNE_HEIGHT
                if runner.leap_from is not None:
                    if not grounded:
                        runner.leap_airborne = True
                    elif runner.leap_airborne:
                        from_x, from_y, from_t = runner.leap_from
                        floors.log(
                            "leap_touch", runner.index, seconds=round(floors.time - from_t, 3),
                            distance=round(math.hypot(root_x - from_x, root_y - from_y), 3),
                            floor=bool(floors.has_floor(runner.layer, root_x, root_y)),
                        )
                        runner.leap_from = None
                        if floors.has_floor(runner.layer, root_x, root_y):
                            # Hand control back at touchdown, or the runner runs blind into the next hole.
                            runner.leaps_landed += 1
                            runner.leaping_until = -1.0
                            floors.log("leap_landed", runner.index)
                            if character.state_machine.is_interacting:
                                character.state_machine.transition_to_locomotion(character)
                if grounded and floors.time >= runner.no_trigger_until - 1.0 and not floors.has_floor(runner.layer, root_x, root_y):
                    if leaping:
                        runner.leaps_failed += 1
                        runner.leaping_until = -1.0
                    begin_fall(floors, runner, pose, "leap" if leaping else ("hole" if floors.tile_at(root_x, root_y) else "edge"))
                elif runner.leaping_until > 0.0 and not leaping:
                    # The feet never came down inside the window, so this is not a clean landing.
                    runner.leaping_until = -1.0
                    runner.leap_from = None
            else:
                runner.previous_feet = None
            if runner.pending_getup is not None and joints["HeadEnd"][2] > 1.35:
                runner.getup_seconds.append(floors.time - runner.pending_getup)
                runner.pending_getup = None
            root = joints["Hips"]
            hips[runner.index] = root + np.array([0.0, 0.0, -LAYER_HEIGHT * runner.layer])
            for j, name in enumerate(RECORDED):
                relative[runner.index, j] = np.clip((joints[name] - root) * 1000.0, -32000, 32000)
        engine.synchronize()
        hips_frames.append(hips)
        joint_frames.append(relative)
        floors.time += DT
        remaining = [r for r in runners if r.eliminated_at is None]
        if len(remaining) <= 1 and winner_at is None:
            winner_at = floors.time
            winner = remaining[0] if remaining else max(runners, key=lambda r: r.eliminated_at)
            floors.log("winner", winner.index)
        if winner_at is not None and (floors.time - winner_at > WIN_RUNOFF_SECONDS or winner.fall is not None):
            break
    if winner_at is None:
        if seconds >= MAX_SECONDS:
            raise RuntimeError(f"seed {seed}: no winner after {seconds} s, {len(remaining)} left")
        floors.log("cut_short", None, left=len(remaining))

    slides = np.array([s for r in runners for s in r.slides])
    meta = {
        "seed": seed, "fps": FPS, "commit": COMMIT, "frames": len(hips_frames), "joints": RECORDED,
        "arena": {
            "layers": LAYERS, "layerHeight": LAYER_HEIGHT, "tile": TILE, "tiles": TILES,
            "half": HALF, "dropSeconds": DROP_SECONDS,
        },
        "runners": [
            {
                "name": NAMES[r.index], "color": COLORS[r.index], "gear": GEAR[r.index],
                "bulk": round(0.8 + 0.55 * r.index / (RUNNERS - 1), 3),
                "sight": r.sight, "noise": r.noise, "strength": r.strength, "falls": r.falls,
                "bravery": r.bravery, "leaps": r.leaps, "leapsLanded": r.leaps_landed, "leapsFailed": r.leaps_failed,
                "eliminatedAt": r.eliminated_at, "tilesBroken": r.tiles_broken,
                "getupSeconds": r.getup_seconds, "unrecovered": r.pending_getup is not None,
                "medianSpeed": float(np.median(r.speeds)) if r.speeds else 0.0,
            }
            for r in runners
        ],
        "footSlide": {
            "frames": int(slides.size),
            "medianMetersPerSecond": float(np.median(slides)),
            "p95MetersPerSecond": float(np.percentile(slides, 95)),
        },
        "events": floors.events,
    }
    return meta, np.stack(hips_frames).tobytes() + np.stack(joint_frames).tobytes()


# An A10 starts in seconds. Waiting for a free L4 took minutes.
@app.function(gpu="A10", volumes={CACHE: volume}, timeout=3000)
def one_round(seed: int, seconds: float) -> tuple[dict, bytes]:
    import warp as wp
    from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
    from app.runtime.common.rig import ASSETS_ROOT, load_rig_assets, model_to_runtime
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.model.contract import ModelRig
    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine

    # Compiled kernels live on the volume, so only the first container pays for them.
    wp.config.kernel_cache_dir = f"{CACHE}/warp"
    wp.init()
    startup = time.perf_counter()
    assets = load_rig_assets(ModelRig.SOMA)
    asset_pool = SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)
    leap_reference = ASSETS_ROOT / "smart_primitives/soma/jump_0_75m.smart_primitive.json"
    asset_pool.load(leap_reference)
    engine = MotionBricksSOMAEngine(
        SOMA_N1_FP32_ONNX_PATH,
        model_to_runtime=model_to_runtime(ModelRig.SOMA),
        build_options=TensorRTBuildOptions(enable_fp16=False),
        cache_dir=f"{CACHE}/trt",
    )
    skeleton = Skeleton(assets.clip_registry.skeleton)
    startup = time.perf_counter() - startup
    with wp.ScopedStream(engine.stream):
        t = time.perf_counter()
        meta, data = simulate(engine, asset_pool, skeleton, seed, seconds, leap_reference)
    meta["timing"] = {"startupSeconds": startup, "simulateSeconds": time.perf_counter() - t}
    volume.commit()
    print(f"seed {seed}: {meta['frames']} frames, {meta['timing']}")
    return meta, data


@app.local_entrypoint()
def main(seeds: str = "1", seconds: float = MAX_SECONDS):
    out = Path(__file__).resolve().parent.parent / "runs" / "spleef"
    out.mkdir(parents=True, exist_ok=True)
    # One container per seed, so three rounds take about as long as one.
    numbers = [int(s) for s in seeds.split(",")]
    for meta, data in one_round.starmap([(n, seconds) for n in numbers]):
        (out / f"seed_{meta['seed']}.json").write_text(json.dumps(meta))
        (out / f"seed_{meta['seed']}.bin").write_bytes(data)
        winner = next((e["runner"] for e in meta["events"] if e["type"] == "winner"), None)
        print(f"seed {meta['seed']}: {meta['frames'] / FPS:.1f} s, winner {winner is not None and NAMES[winner]}, {len(data) / 1e6:.1f} MB, foot slide {meta['footSlide']}, {meta['timing']}")
        for r in meta["runners"]:
            print(" ", r["name"], "out", r["eliminatedAt"] and round(r["eliminatedAt"], 1), "falls", r["falls"], "leaps", r["leaps"], r["leapsLanded"], r["leapsFailed"], "tiles", r["tilesBroken"], "speed", round(r["medianSpeed"], 2), [round(s, 2) for s in r["getupSeconds"]], r["unrecovered"])
