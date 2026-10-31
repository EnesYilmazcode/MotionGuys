"""Race eight MotionBricks runners through doors, sweepers, and tiles on Modal.

Run:  modal run scripts/modal_race.py --seeds 1,2,3

MotionBricks only generates body motion. It never sees the course. This script
owns the world, tests the generated joints against it every frame, and knocks a
runner down when they touch. Writes runs/races/seed_N.json and seed_N.bin.
"""

import heapq
import json
import math
import time
from pathlib import Path

import modal

COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"
REPO = "/opt/motionbricks"
CACHE = "/cache"
FPS = 30
DT = 1.0 / FPS
RUNNERS = 8
MAX_SECONDS = 120.0
PREROLL_SECONDS = 2.0
HALF_WIDTH = 6.0

DOOR_ROWS = ({"x": 12.0, "real": 3}, {"x": 24.0, "real": 2})
DOOR_COUNT = 6
DOOR_WIDTH = 2.0 * HALF_WIDTH / DOOR_COUNT
SWEEPERS = (
    {"id": "low", "x": 44.0, "z": 0.3, "radius": 0.14, "length": 6.0, "speed": 0.8, "angle": 0.4, "action": "jump"},
    {"id": "high", "x": 58.0, "z": 1.6, "radius": 0.15, "length": 6.0, "speed": -0.9, "angle": 1.9, "action": "duck"},
)
POST_RADIUS = 0.5
TILE_X = 72.0
TILE_SIZE = 3.0
TILE_ROWS = 8
TILE_COLUMNS = 4
TILE_DROP_SECONDS = 0.35
FINISH_X = 102.0
CHECKPOINTS = (-1.0, 30.0, 65.0)
RUNOFF_SECONDS = 3.0
INVULNERABLE_SECONDS = 2.0
PLACE_AHEAD = 2.7
HIT_MARGIN = 0.05
COMMIT_TOLERANCE = 0.15
HESITATE_STRENGTH = 0.5
TILE_STRENGTH = 0.75

BODY = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
)
RECORDED = (
    "Hips", "Spine2", "Chest", "Neck1", "Head", "HeadEnd",
    "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeEnd", "RightLeg", "RightShin", "RightFoot", "RightToeEnd",
)
FEET = ("LeftFoot", "LeftToeEnd", "RightFoot", "RightToeEnd")
NAMES = ("Ada", "Bo", "Cleo", "Dov", "Eli", "Fay", "Gus", "Hana")
COLORS = ("#ef5b5b", "#2fa8e0", "#f2b134", "#43b581", "#a06bd6", "#ff8c42", "#1fbfb8", "#e668a7")

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


def quat_mul(a, b):
    import numpy as np

    au, aw = a[:3], a[3]
    bu, bw = b[:3], b[3]
    return np.concatenate([aw * bu + bw * au + np.cross(au, bu), [aw * bw - au @ bu]])


def quat_rotate(q, v):
    import numpy as np

    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def axis_angle(axis, angle):
    import numpy as np

    return np.concatenate([np.asarray(axis) * math.sin(angle / 2), [math.cos(angle / 2)]])


class Skeleton:
    """World positions of the body joints from local transforms."""

    def __init__(self, skeleton):
        self.index = {name: skeleton.joint_index(name) for name in BODY}
        self.parent = {name: skeleton.joint_parent(self.index[name]) for name in BODY}
        by_index = {index: name for name, index in self.index.items()}
        self.chain = [(name, by_index.get(self.parent[name])) for name in BODY]

    def world(self, pose):
        import numpy as np

        positions, rotations = {}, {}
        for name, parent in self.chain:
            local = pose[self.index[name]]
            if parent is None:
                positions[name], rotations[name] = local[:3].astype(np.float64), local[3:].astype(np.float64)
            else:
                positions[name] = positions[parent] + quat_rotate(rotations[parent], local[:3].astype(np.float64))
                rotations[name] = quat_mul(rotations[parent], local[3:].astype(np.float64))
        return positions


class World:
    """The course: doors, sweepers, tiles, and everything that happened on them."""

    def __init__(self, rng):
        self.rng = rng
        self.time = 0.0
        self.events = []
        self.doors = []
        for row in DOOR_ROWS:
            real = rng.choice(DOOR_COUNT, row["real"], replace=False)
            self.doors.append({"x": row["x"], "real": sorted(int(i) for i in real), "broken": set(), "revealed": set()})
        self.safe = self._safe_path()
        self.known_safe = set()
        self.shaking = {}
        self.gone = set()

    def _safe_path(self):
        column = int(self.rng.integers(TILE_COLUMNS))
        path = set()
        for row in range(TILE_ROWS):
            path.add((row, column))
            if row < TILE_ROWS - 1 and self.rng.random() < 0.6:
                column = int(min(TILE_COLUMNS - 1, max(0, column + self.rng.choice([-1, 1]))))
                path.add((row, column))
        return path

    def log(self, kind, runner=None, **detail):
        self.events.append({"t": round(self.time, 3), "type": kind, "runner": runner, **detail})

    def arm_angle(self, sweeper, at=None):
        return sweeper["angle"] + sweeper["speed"] * (self.time if at is None else at)

    def tile_at(self, x, y):
        row = math.floor((x - TILE_X) / TILE_SIZE)
        column = math.floor((y + HALF_WIDTH) / TILE_SIZE)
        if 0 <= row < TILE_ROWS and 0 <= column < TILE_COLUMNS:
            return row, column
        return None

    def supported(self, x, y):
        if abs(y) > HALF_WIDTH:
            return False
        return self.tile_at(x, y) not in self.gone

    def update_tiles(self):
        for tile, started in list(self.shaking.items()):
            if self.time - started >= TILE_DROP_SECONDS:
                del self.shaking[tile]
                self.gone.add(tile)
                self.log("tile_drop", row=tile[0], column=tile[1])

    def step_on(self, tile, runner):
        if tile in self.safe:
            self.known_safe.add(tile)
        elif tile not in self.shaking and tile not in self.gone:
            self.shaking[tile] = self.time
            self.log("tile_shake", runner, row=tile[0], column=tile[1])

    def route(self, tile, rng):
        """Cheapest way across, preferring tiles someone has already survived."""
        noise = {
            (r, c): rng.random() * 0.6 for r in range(TILE_ROWS) for c in range(TILE_COLUMNS)
        }

        def cost(node):
            return 0.05 if node in self.known_safe else 1.0 + noise[node]

        heap, best, parent = [], {}, {}
        starts = [tile] if tile else [(0, c) for c in range(TILE_COLUMNS)]
        for node in starts:
            if node not in self.gone:
                best[node] = cost(node)
                heapq.heappush(heap, (best[node], node))
        while heap:
            value, node = heapq.heappop(heap)
            if value > best[node]:
                continue
            if node[0] == TILE_ROWS - 1:
                path = [node]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                return path[::-1]
            r, c = node
            for next_node in ((r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= next_node[0] < TILE_ROWS and 0 <= next_node[1] < TILE_COLUMNS):
                    continue
                if next_node in self.gone or next_node in self.shaking:
                    continue
                value_next = value + cost(next_node)
                if value_next < best.get(next_node, 1e9):
                    best[next_node], parent[next_node] = value_next, node
                    heapq.heappush(heap, (value_next, next_node))
        return []


class Runner:
    """One character, its intent, and its knockdown state."""

    def __init__(self, index, character, skill, strength, rng):
        self.index = index
        self.character = character
        self.skill = skill
        self.strength = strength
        self.rng = rng
        self.door_choice = {}
        self.hesitate_until = 0.0
        self.route = []
        self.route_tile = None
        self.ko = None
        self.invulnerable_until = 0.0
        self.finished_at = None
        self.previous_chest_x = None
        self.stand_pose = None
        self.pending_getup = None
        self.getup_seconds = []
        self.counts = {"door": 0, "jump": 0, "duck": 0, "tile": 0, "edge": 0}
        self.done = False


def choose_door(world, runner, row_index, y):
    row = world.doors[row_index]
    open_doors = sorted(row["broken"])
    if open_doors and runner.rng.random() < 0.75:
        return min(open_doors, key=lambda d: abs(door_center(d) - y))
    untested = [d for d in range(DOOR_COUNT) if d not in row["revealed"] and d not in row["broken"]]
    return int(runner.rng.choice(untested))


def door_center(door):
    return -HALF_WIDTH + (door + 0.5) * DOOR_WIDTH


def steering(world, runner, x, y):
    import numpy as np

    target_y, ahead = y, 4.0
    for index, row in enumerate(world.doors):
        if x < row["x"] and row["x"] - x < 11.0:
            if index not in runner.door_choice or runner.door_choice[index] in row["revealed"]:
                runner.door_choice[index] = choose_door(world, runner, index, y)
            target_y = door_center(runner.door_choice[index])
            ahead = max(row["x"] - x, 1.5)
            break
    else:
        for sweeper in SWEEPERS:
            if abs(x - sweeper["x"]) < 3.0 and abs(y) < POST_RADIUS + 1.2:
                target_y = math.copysign(POST_RADIUS + 1.4, y if y else 1.0)
        tile = world.tile_at(x, y)
        if TILE_X - 6.0 < x < TILE_X + TILE_ROWS * TILE_SIZE:
            if tile != runner.route_tile or not runner.route or any(n in world.gone for n in runner.route):
                runner.route_tile = tile
                runner.route = world.route(tile, runner.rng)
            if runner.route:
                position = runner.route.index(tile) if tile in runner.route else -1
                next_tile = runner.route[min(position + 1, len(runner.route) - 1)]
                goal_y = -HALF_WIDTH + (next_tile[1] + 0.5) * TILE_SIZE
                offset = goal_y - y
                # Line up with the next tile before crossing into it, so the path never cuts a corner.
                if tile is None:
                    forward = 2.0 if abs(offset) < 1.0 else max(TILE_X - 0.8 - x, 0.05)
                elif next_tile[1] != tile[1]:
                    forward = (TILE_X + (tile[0] + 0.5) * TILE_SIZE - x) * 0.5
                else:
                    forward = 2.0
                vector = np.array([forward, offset], dtype=np.float64)
                vector /= np.linalg.norm(vector)
                return np.array([vector[0], vector[1], 0.0], dtype=np.float32)
    target_y = min(HALF_WIDTH - 1.0, max(-HALF_WIDTH + 1.0, target_y))
    if abs(y) > HALF_WIDTH - 1.0:
        # Past a door at the edge, turn back in before momentum carries the runner off.
        target_y, ahead = math.copysign(HALF_WIDTH - 2.0, y), 1.5
    vector = np.array([ahead, target_y - y], dtype=np.float64)
    vector /= np.linalg.norm(vector)
    return np.array([vector[0], vector[1], 0.0], dtype=np.float32)


def predict_arm_hit(world, runner, sweeper, x, y, heading, speed):
    """Seconds until the runner's straight path meets the arm, seen with this runner's error."""
    if abs(x - sweeper["x"]) > sweeper["length"] + 4.0 or x > sweeper["x"] + sweeper["length"]:
        return None
    error = runner.rng.normal(0.0, runner.skill)
    for step in range(0, 54):
        ahead = step / 18.0
        px = x + heading[0] * speed * ahead - sweeper["x"]
        py = y + heading[1] * speed * ahead
        if math.hypot(px, py) > sweeper["length"]:
            continue
        angle = world.arm_angle(sweeper, world.time + ahead) + error
        if abs(px * math.sin(angle) - py * math.cos(angle)) < 0.35:
            return ahead
    return None


def begin_knockdown(world, runner, pose, impulse, cause, seconds):
    import numpy as np

    character = runner.character
    if character.state_machine.is_interacting:
        character.state_machine.transition_to_locomotion(character)
    horizontal = np.array([impulse[0], impulse[1], 0.0])
    length = np.linalg.norm(horizontal)
    direction = horizontal / length if length > 1e-6 else np.array([-1.0, 0.0, 0.0])
    runner.ko = {
        "cause": cause,
        "age": 0.0,
        "seconds": seconds,
        "position": pose[0, :3].astype(np.float64),
        "velocity": np.asarray(impulse, dtype=np.float64),
        "rotation": pose[0, 3:].astype(np.float64),
        "axis": np.array([-direction[1], direction[0], 0.0]),
        "angle": 0.0,
        "pose": pose.copy(),
        "falling": False,
    }
    runner.counts[cause] += 1
    world.log("knockdown", runner.index, cause=cause)


def step_knockdown(world, runner):
    import numpy as np

    ko, character = runner.ko, runner.character
    ko["age"] += DT
    ko["velocity"][2] -= 9.81 * DT
    ko["position"] += ko["velocity"] * DT
    x, y = ko["position"][:2]
    if not world.supported(x, y):
        if not ko["falling"]:
            world.log("fall", runner.index, cause=ko["cause"])
        ko["falling"] = True
    elif not ko["falling"] and ko["position"][2] < 0.22:
        ko["position"][2] = 0.22
        ko["velocity"] *= np.array([0.75, 0.75, 0.0])
    spin = 2.5 if not ko["falling"] else 4.0
    limit = math.pi / 2 if not ko["falling"] else math.inf
    ko["angle"] = min(limit, ko["angle"] + spin * DT)
    pose = ko["pose"].copy()
    pose[0, :3] = ko["position"]
    pose[0, 3:] = quat_mul(axis_angle(ko["axis"], ko["angle"]), ko["rotation"])
    character.current_pose.assign(pose)
    character.player.record_pose(DT, character.current_pose)
    character.character_ground_position[:2] = ko["position"][:2]
    character.actual_velocity[:] = 0.0
    if ko["falling"]:
        if ko["position"][2] < -7.0:
            respawn(world, runner)
    elif ko["age"] >= ko["seconds"]:
        runner.ko = None
        runner.pending_getup = world.time
        runner.invulnerable_until = world.time + INVULNERABLE_SECONDS
        character.state_machine.active_locomotion_state._force_regeneration = True
        world.log("recover", runner.index)


def respawn(world, runner):
    import numpy as np

    character = runner.character
    x = max(c for c in CHECKPOINTS if c <= max(runner.ko["position"][0], CHECKPOINTS[0]))
    y = -HALF_WIDTH + (runner.index + 0.5) * (2 * HALF_WIDTH / RUNNERS)
    pose = runner.stand_pose.copy()
    pose[0, 0], pose[0, 1] = x, y
    character.current_pose.assign(pose)
    character.player.seed_history_from_pose(character.current_pose)
    character.character_ground_position[:] = np.array([x, y, 0.0], dtype=np.float32)
    character.actual_velocity[:] = 0.0
    runner.ko = None
    runner.route, runner.route_tile = [], None
    runner.invulnerable_until = world.time + INVULNERABLE_SECONDS
    character.state_machine.active_locomotion_state._force_regeneration = True
    world.log("respawn", runner.index, x=x)


def start_action(runner, action, x, y, heading, engine, asset_pool, references):
    import warp as wp
    from app.runtime.common.smart_primitive_behaviors import VelocityMatchingSequenceBehavior

    # A primitive clamps its pivot to limits around its first placement, so reusing
    # one drags the runner back there. Every action gets a new instance.
    primitive = asset_pool.instantiate(references[action], VelocityMatchingSequenceBehavior)
    yaw = math.atan2(heading[1], heading[0]) - math.pi / 2
    transform = wp.transform(
        wp.vec3(x + heading[0] * PLACE_AHEAD, y + heading[1] * PLACE_AHEAD, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
    )
    primitive.set_node_world_transform(0, transform, stream=engine.stream)
    runner.character.state_machine.start_interaction(runner.character, primitive)


def check_contacts(world, runner, joints, pose, chest_x_before):
    import numpy as np

    chest = joints["Chest"]
    for index, row in enumerate(world.doors):
        if chest_x_before is not None and chest_x_before < row["x"] <= chest[0]:
            door = int(min(DOOR_COUNT - 1, max(0, (chest[1] + HALF_WIDTH) // DOOR_WIDTH)))
            if door in row["broken"]:
                continue
            if door in row["real"]:
                row["broken"].add(door)
                world.log("door_break", runner.index, row=index, door=door)
            else:
                row["revealed"].add(door)
                world.log("door_solid", runner.index, row=index, door=door)
                begin_knockdown(world, runner, pose, (-2.5, 0.0, 1.2), "door", 0.9)
                return
    if world.time < runner.invulnerable_until:
        return
    for sweeper in SWEEPERS:
        angle = world.arm_angle(sweeper)
        direction = np.array([math.cos(angle), math.sin(angle)])
        center = np.array([sweeper["x"], 0.0])
        for name, position in joints.items():
            offset = position[:2] - center
            along = float(np.clip(offset @ direction, -sweeper["length"], sweeper["length"]))
            closest = center + direction * along
            distance = math.hypot(*(position[:2] - closest), position[2] - sweeper["z"])
            if distance < sweeper["radius"] + HIT_MARGIN:
                normal = np.array([-direction[1], direction[0]]) * math.copysign(1.0, along * sweeper["speed"])
                tip_speed = abs(sweeper["speed"]) * abs(along)
                push = 1.5 + 0.6 * tip_speed
                impulse = (normal[0] * push, normal[1] * push, 2.6)
                world.log("arm_hit", runner.index, sweeper=sweeper["id"], joint=name, height=round(float(position[2]), 3))
                begin_knockdown(world, runner, pose, impulse, sweeper["action"], 1.1)
                return
    for name in FEET:
        position = joints[name]
        if position[2] < 0.2:
            tile = world.tile_at(position[0], position[1])
            if tile is not None:
                world.step_on(tile, runner.index)


def calibrate(engine, asset_pool, references, skeleton):
    """Measure how long after starting an action the body is clear of a sweeper."""
    import numpy as np
    import warp as wp
    from app.runtime.common.motion import LocomotionCommand
    from app.runtime.soma.character import SOMACharacter

    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    windows = {}
    for action, sweeper in (("jump", SWEEPERS[0]), ("duck", SWEEPERS[1])):
        character = SOMACharacter(engine, asset_pool=asset_pool, initial_transform=wp.transform_identity())
        command = LocomotionCommand(facing_direction=forward, movement=forward, use_max_speed=True)
        for _ in range(int(2.5 * FPS)):
            character.step(DT, command)
        position = character.character_ground_position
        runner = Runner(0, character, 0.0, 1.0, None)
        start_action(runner, action, float(position[0]), float(position[1]), forward, engine, asset_pool, references)
        clear = []
        for frame in range(int(2.5 * FPS)):
            character.step(DT, command)
            joints = skeleton.world(character.current_pose.numpy())
            heights = [p[2] for p in joints.values()]
            if action == "jump":
                ok = min(joints[n][2] for n in FEET) > sweeper["z"] + sweeper["radius"] + HIT_MARGIN
            else:
                ok = max(heights) < sweeper["z"] - sweeper["radius"] - HIT_MARGIN
            if ok:
                clear.append((frame + 1) * DT)
        if not clear:
            raise RuntimeError(f"{action}: the body never clears the sweeper height")
        windows[action] = {"start": clear[0], "end": clear[-1], "frames": len(clear)}
    return windows


def simulate(engine, asset_pool, references, skeleton, windows, seed):
    import numpy as np
    import warp as wp
    from app.runtime.common.motion import LocomotionCommand, PlaybackSource
    from app.runtime.soma.character import SOMACharacter

    rng = np.random.default_rng(seed)
    world = World(rng)
    runners = []
    for index in range(RUNNERS):
        y = -HALF_WIDTH + (index + 0.5) * (2 * HALF_WIDTH / RUNNERS)
        character = SOMACharacter(
            engine,
            asset_pool=asset_pool,
            initial_transform=wp.transform(wp.vec3(-6.5, y, 0.0), wp.quat_identity()),
        )
        runners.append(Runner(
            index, character,
            skill=float(rng.uniform(0.04, 0.3)),
            strength=float(rng.uniform(0.9, 1.0)),
            rng=np.random.default_rng(seed * 100 + index),
        ))

    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(int(PREROLL_SECONDS * FPS)):
        for runner in runners:
            runner.character.step(
                DT, LocomotionCommand(forward, forward * runner.strength, use_max_speed=True)
            )
    for runner in runners:
        runner.stand_pose = runner.character.current_pose.numpy().copy()

    index_of = {name: i for i, name in enumerate(BODY)}
    hips_frames, joint_frames = [], []
    generated_frames = 0
    max_frames = int(MAX_SECONDS * FPS)
    for _ in range(max_frames):
        world.update_tiles()
        hips = np.full((RUNNERS, 3), np.nan, dtype=np.float32)
        relative = np.zeros((RUNNERS, len(RECORDED), 3), dtype=np.int16)
        positions = [r.character.character_ground_position.copy() for r in runners]
        for runner in runners:
            if runner.done:
                continue
            character = runner.character
            if runner.ko is not None:
                step_knockdown(world, runner)
            else:
                x, y = float(character.character_ground_position[0]), float(character.character_ground_position[1])
                heading = steering(world, runner, x, y)
                for other, position in enumerate(positions):
                    offset = np.array([y - position[1]])
                    if other != runner.index and not runners[other].done and abs(position[0] - x) < 1.2 and abs(offset[0]) < 0.8:
                        heading[1] += float(np.sign(offset[0]) or 1.0) * 0.5
                heading /= np.linalg.norm(heading)
                on_tiles = TILE_X - 1.0 < x < TILE_X + TILE_ROWS * TILE_SIZE
                strength = runner.strength * (TILE_STRENGTH if on_tiles else 1.0)
                if runner.hesitate_until > world.time:
                    strength *= HESITATE_STRENGTH
                character.step(
                    DT, LocomotionCommand(heading, heading * strength, use_max_speed=True)
                )
                if character._playback_source is PlaybackSource.CLIP:
                    raise RuntimeError("a sample clip played during the race")
                generated_frames += 1
                if not character.state_machine.is_interacting and world.time >= runner.invulnerable_until - 1.0:
                    speed = max(float(np.linalg.norm(character.actual_velocity)), 1.0)
                    for sweeper in SWEEPERS:
                        window = windows[sweeper["action"]]
                        middle = (window["start"] + window["end"]) / 2
                        lead = predict_arm_hit(world, runner, sweeper, x, y, heading, speed)
                        if lead is None:
                            continue
                        if lead < middle - COMMIT_TOLERANCE:
                            # Too late to clear this pass, so ease off and take the next one.
                            runner.hesitate_until = world.time + 0.3
                        elif lead <= middle + COMMIT_TOLERANCE:
                            start_action(runner, sweeper["action"], x, y, heading, engine, asset_pool, references)
                            world.log("commit", runner.index, action=sweeper["action"], seen_lead=round(lead, 3), needed_lead=round(middle, 3))
                            break
            pose = character.current_pose.numpy().copy()
            if not np.isfinite(pose).all():
                raise RuntimeError(f"runner {runner.index}: non-finite pose at {world.time:.2f} s")
            joints = skeleton.world(pose)
            if runner.ko is None:
                check_contacts(world, runner, joints, pose, runner.previous_chest_x)
                x, y = joints["Hips"][:2]
                if runner.ko is None and not world.supported(x, y):
                    cause = "edge" if abs(y) > HALF_WIDTH else "tile"
                    begin_knockdown(world, runner, pose, (0.0, 0.0, 0.0), cause, 1.0)
            runner.previous_chest_x = joints["Chest"][0] if runner.ko is None else None
            if runner.pending_getup is not None and joints["HeadEnd"][2] > 1.35:
                runner.getup_seconds.append(world.time - runner.pending_getup)
                runner.pending_getup = None
            root = joints["Hips"]
            hips[runner.index] = root
            for j, name in enumerate(RECORDED):
                relative[runner.index, j] = np.clip((joints[name] - root) * 1000.0, -32000, 32000)
            if runner.finished_at is None and root[0] >= FINISH_X:
                runner.finished_at = world.time
                world.log("finish", runner.index, place=sum(r.finished_at is not None for r in runners))
            if runner.finished_at is not None and world.time - runner.finished_at > RUNOFF_SECONDS:
                runner.done = True
        engine.synchronize()
        hips_frames.append(hips)
        joint_frames.append(relative)
        world.time += DT
        if all(r.done for r in runners):
            break

    meta = {
        "seed": seed,
        "fps": FPS,
        "commit": COMMIT,
        "frames": len(hips_frames),
        "joints": RECORDED,
        "course": {
            "halfWidth": HALF_WIDTH, "finishX": FINISH_X,
            "doorRows": [{"x": r["x"], "real": r["real"]} for r in world.doors],
            "doorCount": DOOR_COUNT, "sweepers": SWEEPERS, "postRadius": POST_RADIUS,
            "tiles": {"x": TILE_X, "size": TILE_SIZE, "rows": TILE_ROWS, "columns": TILE_COLUMNS, "safe": sorted(world.safe)},
        },
        "runners": [
            {
                "name": NAMES[r.index], "color": COLORS[r.index], "skill": r.skill, "strength": r.strength,
                "finishedAt": r.finished_at, "counts": r.counts, "getupSeconds": r.getup_seconds,
                "unrecovered": r.pending_getup is not None,
            }
            for r in runners
        ],
        "windows": windows,
        "events": world.events,
    }
    return meta, np.stack(hips_frames).tobytes() + np.stack(joint_frames).tobytes()


@app.function(gpu="L4", volumes={CACHE: volume}, timeout=3000)
def race(seeds: list[int]) -> list[tuple[dict, bytes]]:
    import warp as wp
    from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
    from app.runtime.common.rig import ASSETS_ROOT, load_rig_assets, model_to_runtime
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.model.contract import ModelRig
    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine

    wp.init()
    assets = load_rig_assets(ModelRig.SOMA)
    asset_pool = SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)
    references = {
        "jump": ASSETS_ROOT / "smart_primitives/soma/jump_0_75m.smart_primitive.json",
        "duck": ASSETS_ROOT / "smart_primitives/soma/avoidance.smart_primitive.json",
    }
    for reference in references.values():
        asset_pool.load(reference)
    engine = MotionBricksSOMAEngine(
        SOMA_N1_FP32_ONNX_PATH,
        model_to_runtime=model_to_runtime(ModelRig.SOMA),
        build_options=TensorRTBuildOptions(enable_fp16=False),
        cache_dir=f"{CACHE}/trt",
    )
    skeleton = Skeleton(assets.clip_registry.skeleton)
    results = []
    with wp.ScopedStream(engine.stream):
        windows = calibrate(engine, asset_pool, references, skeleton)
        print("windows", json.dumps(windows))
        for seed in seeds:
            t = time.perf_counter()
            meta, data = simulate(engine, asset_pool, references, skeleton, windows, seed)
            meta["simSeconds"] = time.perf_counter() - t
            print(f"seed {seed}: {meta['frames']} frames in {meta['simSeconds']:.0f} s")
            results.append((meta, data))
    return results


@app.function(gpu="L4", volumes={CACHE: volume}, timeout=1200)
def measure_jumps(repeats: int = 5) -> dict:
    """How far each jump carries the feet, the widest hole it could cross."""
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

    wp.init()
    assets = load_rig_assets(ModelRig.SOMA)
    asset_pool = SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)
    references = {
        name: ASSETS_ROOT / f"smart_primitives/soma/{name}.smart_primitive.json"
        for name in ("jump_0_5m", "jump_0_75m")
    }
    for reference in references.values():
        asset_pool.load(reference)
    engine = MotionBricksSOMAEngine(
        SOMA_N1_FP32_ONNX_PATH,
        model_to_runtime=model_to_runtime(ModelRig.SOMA),
        build_options=TensorRTBuildOptions(enable_fp16=False),
        cache_dir=f"{CACHE}/trt",
    )
    skeleton = Skeleton(assets.clip_registry.skeleton)
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = {}
    with wp.ScopedStream(engine.stream):
        for name in references:
            flights = []
            for _ in range(repeats):
                character = SOMACharacter(engine, asset_pool=asset_pool, initial_transform=wp.transform_identity())
                command = LocomotionCommand(forward, forward, use_max_speed=True)
                for _ in range(int(2.5 * FPS)):
                    character.step(DT, command)
                position = character.character_ground_position
                start_action(Runner(0, character, 0.0, 1.0, None), name, float(position[0]), float(position[1]), forward, engine, asset_pool, references)
                frames = []
                for _ in range(int(4.0 * FPS)):
                    character.step(DT, command)
                    joints = skeleton.world(character.current_pose.numpy())
                    frames.append([[joints[n][0], joints[n][2]] for n in FEET])
                frames = np.array(frames)
                airborne = frames[:, :, 1].min(axis=1) > 0.15
                runs, begin = [], None
                for f, up in enumerate([*airborne, False]):
                    if up and begin is None:
                        begin = f
                    elif not up and begin is not None:
                        runs.append((begin, f - 1))
                        begin = None
                if not runs:
                    raise RuntimeError(f"{name}: the feet never left the ground")
                a, b = max(runs, key=lambda r: r[1] - r[0])
                takeoff = frames[max(a - 1, 0), :, 0].max()
                landing = frames[min(b + 1, len(frames) - 1), :, 0].min()
                flights.append({
                    "seconds": (b - a + 1) * DT,
                    "gap": float(landing - takeoff),
                    "peak_feet": float(frames[a:b + 1, :, 1].min(axis=1).max()),
                })
            results[name] = flights
    return results


@app.local_entrypoint()
def jumps(repeats: int = 5):
    for name, flights in measure_jumps.remote(repeats).items():
        gaps = [f["gap"] for f in flights]
        print(name, "clearable gap m:", [round(g, 2) for g in gaps], "air s:", [round(f["seconds"], 2) for f in flights], "feet peak m:", [round(f["peak_feet"], 2) for f in flights])


@app.local_entrypoint()
def main(seeds: str = "1"):
    out = Path(__file__).resolve().parent.parent / "runs" / "races"
    out.mkdir(parents=True, exist_ok=True)
    for meta, data in race.remote([int(s) for s in seeds.split(",")]):
        (out / f"seed_{meta['seed']}.json").write_text(json.dumps(meta))
        (out / f"seed_{meta['seed']}.bin").write_bytes(data)
        print(f"seed {meta['seed']}: {meta['frames'] / FPS:.1f} s, {len(data) / 1e6:.1f} MB")
        for runner in meta["runners"]:
            print(" ", runner["name"], runner["finishedAt"], runner["counts"], [round(s, 2) for s in runner["getupSeconds"]], runner["unrecovered"])
