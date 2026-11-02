"""Eight robots on six floors of hexagons above a pool of slime.

Run:  modal run scripts/modal_hex.py --seeds 1,2,3
Try:  modal run scripts/modal_hex.py --seconds 20     a quick look, no winner needed
Test: modal run scripts/modal_hex.py --seeds 1,2,3 --brains half     half trained, half hand-written
Solo: modal run scripts/modal_hex.py --seeds 1,2,3 --policy solo55slow --solo-radius 8.3     one robot on a small floor, until its first fall

A trained network decides where each robot goes and how fast (hex_brain.py), MotionBricks generates its running, leaping,
and getting up. A robot that drops through a hole keeps running and lands on its feet on the next floor. MuJoCo takes over when
it gets knocked down and for the last plunge into the slime (hex_physics.py).

Layout and tile behavior are taken from gameplay footage, see docs/HEXAGONS.md. A touched
tile stays solid, whitens, and vanishes in place. It never drops. Its timer keeps the footage's
ratio of timer to the time spent crossing a tile, because these runners are slower.
Writes runs/hex/seed_N.json and seed_N.bin.
"""

import json
import math
import time
from pathlib import Path

import modal
from modal_leap import make_leap_type
import torch
from modal_race import (
    BODY, CACHE, COLORS, COMMIT, DT, FEET, FPS, NAMES, RECORDED, RUNNERS, Skeleton, axis_angle, image, quat_mul, volume,
)
from modal_spleef import AIRBORNE_HEIGHT, GEAR, LEAP_COOLDOWN_SECONDS, LEAP_LOOK_FRAMES, LEAP_SECONDS, STEP_HEIGHT, Runner, choose_direction, choose_leap, start_leap
from hex_brain import (
    BLOCKED_LOOKS, DROP, FLOOR_RADII, GAIT_STRENGTH, GAP_LOOKS, HEADINGS, HEX_RADIUS, HEX_WIDTH, LAYER_HEIGHT, LAYERS, LEAP, LEAP_COOLDOWN, MOVE_LEAP, OUT,
    PEDESTAL_RING, PERSONALITIES, REACH, RUN, STUN, THINK_SECONDS, TOUCH_SECONDS, Policy, Sim,
)
from hex_physics import Bodies, rail_capsules

SLIME_DEPTH = 4.0
COUNTDOWN_SECONDS = 3.0
PEDESTAL_HEIGHT = 1.5
BLOCK_AHEAD = 2.2
THINK_FRAMES = 15
BRAIN_FRAMES = round(THINK_SECONDS * FPS)
NO_TRIGGER_SECONDS = 1.2
# Picking among the network's choices. Lower than 1 steadies it, because turning all the time slows the generated run.
BRAIN_TEMPERATURE = 0.5
# Relative mass per robot, roster order, roughly following the real machines.
MASSES = (1.15, 1.0, 0.7, 1.05, 1.25, 1.5, 0.65, 1.4)
BODY_RADIUS = 0.3
KNOCK_CLOSING_SPEED = 5.0
KNOCK_BY_LIMP_SPEED = 2.2
REST_MIN_SECONDS = 0.3
REST_MAX_SECONDS = 1.1
REST_SPEED = 1.0
MAX_SECONDS = 180.0
WIN_RUNOFF_SECONDS = 5.0
# Rare rounds stall with robots left. Then the floors go, like a game's sudden death, and the log records who was stuck.
SUDDEN_DEATH_SECONDS = 150.0
# Unrecorded: run in, then stop. Idle clips are under an evaluation-only license, so the first recorded frame is already generated.
PREROLL_RUN_SECONDS = 2.0
PREROLL_STOP_SECONDS = 1.5
PREROLL_REACH = 4.5
# Zero input drops into those idle clips and any other input walks, so a waiting runner holds its last generated pose.
SLOW_STRENGTH = 0.1
WIN_HOLD_AFTER_SECONDS = 0.6
# A foot can land just off the tile the hips landed on.
LAND_GRACE_SECONDS = 0.4

app = modal.App(
    "motionguys-hex",
    image=image.pip_install("mujoco==3.3.7")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("modal_race", "modal_leap", "modal_spleef", "hex_brain", "hex_physics"),
)


def hex_center(q, r):
    return HEX_WIDTH * (q + r / 2), 1.5 * HEX_RADIUS * r


def hex_at(x, y):
    """The honeycomb cell holding a point, by cube rounding."""
    fq = (x * math.sqrt(3) / 3 - y / 3) / HEX_RADIUS
    fr = (y * 2 / 3) / HEX_RADIUS
    fs = -fq - fr
    q, r, s = round(fq), round(fr), round(fs)
    dq, dr, ds = abs(q - fq), abs(r - fr), abs(s - fs)
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    return q, r


def layer_tiles(radius):
    reach = int(radius / HEX_WIDTH) + 2
    return {
        (q, r)
        for q in range(-reach, reach + 1)
        for r in range(-reach, reach + 1)
        if math.hypot(*hex_center(q, r)) <= radius
    }


class Floors:
    """Which hexagons are touched or gone, and everything that happened."""

    def __init__(self, radii=FLOOR_RADII):
        self.time = 0.0
        self.tiles = [layer_tiles(radius) for radius in radii]
        self.gone = [set() for _ in range(LAYERS)]
        self.touched = [{} for _ in range(LAYERS)]
        self.events = []

    def log(self, kind, runner=None, **detail):
        self.events.append({"t": round(self.time, 3), "type": kind, "runner": runner, **detail})

    def tile_at(self, layer, x, y):
        tile = hex_at(x, y)
        return tile if tile in self.tiles[layer] else None

    def has_floor(self, layer, x, y):
        tile = self.tile_at(layer, x, y)
        return tile is not None and tile not in self.gone[layer]

    def safe(self, layer, x, y):
        tile = self.tile_at(layer, x, y)
        return tile is not None and tile not in self.gone[layer] and tile not in self.touched[layer]

    def step_on(self, layer, x, y, runner):
        tile = self.tile_at(layer, x, y)
        if tile is None or tile in self.gone[layer] or tile in self.touched[layer]:
            return False
        self.touched[layer][tile] = self.time
        self.log("tile_touch", runner, layer=layer, q=tile[0], r=tile[1], x=round(x, 2), y=round(y, 2))
        return True

    def update(self):
        vanished = []
        for layer, touched in enumerate(self.touched):
            for tile, started in list(touched.items()):
                if self.time - started >= TOUCH_SECONDS:
                    del touched[tile]
                    self.gone[layer].add(tile)
                    self.log("tile_gone", None, layer=layer, q=tile[0], r=tile[1])
                    vanished.append((layer, tile))
        return vanished


class Brain:
    """Fills a one-round copy of the training sim with the real round's state and asks the network."""

    def __init__(self, weights, seed, heading_temperature, move_temperature, radii=FLOOR_RADII):
        self.heading_temperature, self.move_temperature = heading_temperature, move_temperature
        self.sim = Sim(1, "cpu", seed=seed, radii=radii)
        self.policy = Policy()
        self.policy.load_state_dict(torch.load(weights, map_location="cpu"))
        self.policy.eval()
        self.sim.personality[0] = torch.tensor(PERSONALITIES)

    def sync(self, floors, runners):
        sim = self.sim
        sim.solid[0] = sim.exists
        sim.touched_at[0] = math.inf
        sim.time[0] = floors.time
        for layer in range(LAYERS):
            for q, r in floors.gone[layer]:
                sim.solid[0, layer, q + REACH, r + REACH] = False
            for (q, r), started in floors.touched[layer].items():
                sim.touched_at[0, layer, q + REACH, r + REACH] = started
        for runner in runners:
            i = runner.index
            character = runner.character
            if runner.eliminated_at is not None:
                mode = OUT
            elif runner.limp is not None:
                mode = STUN if runner.limp["landed"] is not None else DROP
            elif runner.dropping:
                mode = DROP
            elif floors.time < runner.leaping_until:
                mode = LEAP
            else:
                mode = RUN
            sim.mode[0, i] = mode
            sim.layer[0, i] = runner.layer
            sim.x[0, i], sim.y[0, i] = float(runner.ground[0]), float(runner.ground[1])
            sim.heading[0, i] = math.atan2(runner.direction[1], runner.direction[0])
            sim.speed[0, i] = float(math.hypot(character.actual_velocity[0], character.actual_velocity[1]))
            # The body trails the command, and the network has to know by how much.
            sim.move_angle[0, i] = math.atan2(character.actual_velocity[1], character.actual_velocity[0]) if sim.speed[0, i] > 0.3 else sim.heading[0, i]
            sim.leap_ready[0, i] = runner.leap_ready_at
            sim.landed_at[0, i] = getattr(runner, "landed_at", -math.inf)
            sim.no_touch_until[0, i] = runner.no_trigger_until

    @torch.no_grad()
    def decide(self, floors, runners):
        self.sync(floors, runners)
        heading_logits, gait_logits, move_logits, _ = self.policy(self.sim.observe())
        pick = lambda logits, temperature: torch.distributions.Categorical(logits=logits[0] / temperature).sample().tolist()
        return pick(heading_logits, self.heading_temperature), pick(gait_logits, BRAIN_TEMPERATURE), pick(move_logits, self.move_temperature)


def go_limp(floors, bodies, runner, velocity, cause, by=None):
    """Physics takes over from here, starting at the robot's current pose and speed."""
    import numpy as np

    character = runner.character
    if character.state_machine.is_interacting:
        character.state_machine.transition_to_locomotion(character)
    runner.leaping_until = -1.0
    runner.leap_from = None
    runner.leap_release = None
    forward = np.array([velocity[0], velocity[1], 0.0])
    length = np.linalg.norm(forward)
    forward = forward / length if length > 1e-6 else np.array([1.0, 0.0, 0.0])
    # Tip gently the way it is going, so a fall starts as a stumble and not a spin.
    spin = np.cross([0.0, 0.0, 1.0], forward) * (0.5 + length * 0.25)
    bodies.go_limp(runner.index, velocity=velocity, spin=spin, stiffness=25.0)
    runner.limp = {"started": floors.time, "landed": None, "touching": 0.0, "phase": runner.index * 1.7}
    runner.falls += 1
    if cause == "knock":
        floors.log("knock", runner.index, by=by, layer=runner.layer)
    else:
        floors.log("fall", runner.index, layer=runner.layer, cause=cause)


def drop_through(floors, runner, cause):
    """Through a hole to the next floor. No ragdoll and no get-up: it keeps its stride and lands on its feet, like stepping off the start."""
    character = runner.character
    if character.state_machine.is_interacting:
        character.state_machine.transition_to_locomotion(character)
    runner.leaping_until = -1.0
    runner.leap_from = None
    runner.leap_release = None
    runner.falls += 1
    floors.log("fall", runner.index, layer=runner.layer, cause=cause)
    runner.layer += 1
    floors.log("pass", runner.index, layer=runner.layer)
    runner.lift = LAYER_HEIGHT
    runner.lift_speed = 0.0
    runner.dropping = True


def scramble(bodies, runner, time):
    """A falling robot's arms and legs move a little around the pose it fell in, the way anyone's would."""
    phase = time * 5.0 + runner.limp["phase"]
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        bodies.sway(runner.index, f"{side}Arm", 0.35 * math.sin(phase * sign), 18.0)
        bodies.sway(runner.index, f"{side}ForeArm", 0.25 * math.sin(phase + sign), 8.0)
        bodies.sway(runner.index, f"{side}Leg", 0.22 * math.sin(phase * 0.8 * sign + 1.0), 30.0)
        bodies.sway(runner.index, f"{side}Shin", 0.25 * math.sin(phase * 0.8 + sign), 12.0)


def step_limp(floors, bodies, runner, tiles_touched, tile_list, index):
    """A limp body falls, lands, and hands back to MotionBricks for the get-up once it comes to rest."""
    import numpy as np

    limp, character = runner.limp, runner.character
    hips = bodies.hips_position(runner.index)
    while hips[2] < -LAYER_HEIGHT * runner.layer - 0.6 and runner.layer < LAYERS - 1:
        runner.layer += 1
        floors.log("pass", runner.index, layer=runner.layer)
    slime = -LAYER_HEIGHT * (LAYERS - 1) - SLIME_DEPTH
    if not limp.get("splashed") and hips[2] <= slime:
        limp["splashed"] = True
        floors.log("splash", runner.index, x=round(float(hips[0]), 2), y=round(float(hips[1]), 2))
    if hips[2] < slime - 1.6:
        runner.eliminated_at = floors.time
        floors.log("eliminated", runner.index)
        bodies.park(runner.index)
        return None
    touching = bool(tiles_touched)
    if touching:
        limp["touching"] = floors.time
        if limp["landed"] is None:
            limp["landed"] = floors.time
            floors.log("land", runner.index, layer=runner.layer, x=round(float(hips[0]), 2), y=round(float(hips[1]), 2))
            bodies.set_stiffness(runner.index, 6.0)
    elif limp["landed"] is not None and floors.time - limp["touching"] > 0.2:
        limp["landed"] = None  # Slid off an edge or the tile went. Still falling.
    if limp["landed"] is None:
        anchor = limp.setdefault("anchor", (floors.time, hips.copy()))
        if np.linalg.norm(hips - anchor[1]) > 0.5:
            limp["anchor"] = (floors.time, hips.copy())
        elif floors.time - anchor[0] > 2.5:
            # Wedged on a rail, where flailing limbs keep it moving but going nowhere.
            bodies.pass_through_rails(runner.index)
            floors.log("unstuck", runner.index)
            limp["anchor"] = (floors.time, hips.copy())
        scramble(bodies, runner, floors.time)
    height = -LAYER_HEIGHT * runner.layer
    pose = character.current_pose.numpy().copy()
    bodies.read_pose(runner.index, pose, index, height, (0.0, 0.0))
    done = False
    if limp["landed"] is not None:
        rested = floors.time - limp["landed"]
        speed = float(np.linalg.norm(bodies.hips_velocity(runner.index)))
        done = rested >= REST_MAX_SECONDS or (rested >= REST_MIN_SECONDS and speed < REST_SPEED)
    if done:
        runner.limp = None
        bodies.wake(runner.index)
        pose[index["Hips"], 2] = max(float(pose[index["Hips"], 2]), 0.15)
        runner.pending_getup = floors.time
        runner.no_trigger_until = floors.time + NO_TRIGGER_SECONDS
        character.state_machine.active_locomotion_state._force_regeneration = True
        floors.log("recover", runner.index)
    character.current_pose.assign(pose)
    character.player.record_pose(DT, character.current_pose)
    character.character_ground_position[:2] = pose[index["Hips"], :2]
    character.actual_velocity[:] = 0.0
    runner.shift[:] = 0.0
    return pose


def separate(floors, bodies, runners):
    """Running robots shove each other instead of passing through. A hard enough head-on knocks the lighter one down."""
    import numpy as np

    standing = [r for r in runners if r.eliminated_at is None and r.limp is None and r.lift <= 0.0]
    for i, a in enumerate(standing):
        for b in standing[i + 1:]:
            if a.layer != b.layer:
                continue
            gap = a.ground - b.ground
            distance = float(np.linalg.norm(gap))
            overlap = 2 * BODY_RADIUS - distance
            if overlap <= 0.0:
                continue
            normal = gap / max(distance, 1e-4)
            a_mass, b_mass = MASSES[a.index], MASSES[b.index]
            a.shift += normal * overlap * b_mass / (a_mass + b_mass)
            b.shift -= normal * overlap * a_mass / (a_mass + b_mass)
            closing = -float((a.character.actual_velocity[:2] - b.character.actual_velocity[:2]) @ normal)
            if closing > KNOCK_CLOSING_SPEED and floors.time >= max(a.no_trigger_until, b.no_trigger_until):
                loser, winner = (a, b) if a_mass < b_mass else (b, a)
                push = winner.character.actual_velocity[:2] * (MASSES[winner.index] / MASSES[loser.index]) * 0.5
                go_limp(floors, bodies, loser, np.array([push[0], push[1], 0.6]), "knock", by=winner.index)


def gap_ahead(floors, layer, x, y, direction):
    """The same test the training sim pays a save bonus for, so the log can say which leaps were real saves."""
    return not all(floors.safe(layer, x + direction[0] * d, y + direction[1] * d) for d in GAP_LOOKS)


def way_blocked(floors, layer, x, y, direction, sectors=1):
    """No way forward on foot, the test for a real save. The tile it stands on does not count."""
    angle = math.atan2(direction[1], direction[0])
    own = hex_at(x, y)
    for offset in [k * 2 * math.pi / HEADINGS for k in range(-sectors, sectors + 1)]:
        cos, sin = math.cos(angle + offset), math.sin(angle + offset)
        if all(hex_at(x + cos * d, y + sin * d) == own or floors.safe(layer, x + cos * d, y + sin * d) for d in BLOCKED_LOOKS):
            return False
    return True


def simulate(engine, asset_pool, skeleton, reference, seed, seconds, leap_reference, brains, policy, heading_temperature, move_temperature, leap_cooldown, land_hold, leaps_allowed, leap_after, block_sectors, leap_shape, solo_radius):
    solo = solo_radius > 0
    radii = (solo_radius, *FLOOR_RADII[1:]) if solo else FLOOR_RADII
    ring = 0.0 if solo else PEDESTAL_RING
    if solo:
        # The solo brains learned their inputs with the longer leap's look distances.
        import hex_brain
        hex_brain.LANDING_LOOKS, hex_brain.RUNOUT_LOOKS = (4.0, 4.6, 5.2, 5.8, 6.4), (7.0, 8.0, 9.0, 10.0)
    import modal_spleef
    modal_spleef.LEAP_PLACE, modal_spleef.LEAP_STRETCH = (float(v) for v in leap_shape.split(","))
    import numpy as np
    import warp as wp
    from app.runtime.common.motion import LocomotionCommand, PlaybackSource
    from app.runtime.soma.character import SOMACharacter

    Leap = make_leap_type()
    rng = np.random.default_rng(seed)
    floors = Floors(radii)
    order = rng.permutation(RUNNERS)
    runners = []
    for index in range(RUNNERS):
        around = 2 * math.pi * int(order[index]) / RUNNERS
        x, y = (ring + PREROLL_REACH) * math.cos(around), (ring + PREROLL_REACH) * math.sin(around)
        angle = around + math.pi
        character = SOMACharacter(
            engine, asset_pool=asset_pool,
            initial_transform=wp.transform(
                wp.vec3(x, y, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
            ),
        )
        runner = Runner(
            index, character,
            sight=float(rng.uniform(3.0, 8.0)),
            noise=float(rng.uniform(0.1, 0.9)),
            strength=float(rng.uniform(0.75, 1.0)),
            bravery=float(rng.uniform(0.15, 0.6)),
        )
        runner.direction = np.array([math.cos(angle), math.sin(angle)])
        runner.lift = PEDESTAL_HEIGHT
        runner.lift_speed = 0.0
        runner.limp = None
        runner.knocks_dealt = 0
        runner.gait = None
        runner.shift = np.zeros(2)
        runner.ground = np.zeros(2)
        runner.leap_release = None
        runner.dropping = False
        runner.land_grace_until = -1.0
        runners.append(runner)
    for frame in range(int((PREROLL_RUN_SECONDS + PREROLL_STOP_SECONDS) * FPS)):
        for runner in runners:
            move = np.array([runner.direction[0], runner.direction[1], 0.0], dtype=np.float32)
            running = frame < PREROLL_RUN_SECONDS * FPS
            runner.character.step(DT, LocomotionCommand(move, move * (0.6 if running else SLOW_STRENGTH), use_max_speed=running))
    for runner in runners:
        # Each start hexagon goes wherever its runner came to rest.
        runner.pedestal = tuple(float(v) for v in runner.character.character_ground_position[:2])
    pedestals = [{"x": r.pedestal[0], "y": r.pedestal[1]} for r in runners]
    for runner in runners:
        runner.ground = np.array(runner.pedestal)

    # Half and half is the fair test of the network against the hand-written policy.
    if solo:
        thinking = {0}
    elif brains == "all":
        thinking = set(range(RUNNERS))
    elif brains == "half":
        thinking = {int(i) for i in rng.permutation(RUNNERS)[:RUNNERS // 2]}
    else:
        thinking = set()
    brain = Brain(f"{CACHE}/brain/policy_{policy}.pt", seed, heading_temperature, move_temperature, radii) if thinking else None

    index = {name: skeleton.index[name] for name in BODY}
    rest_pose = runners[0].character.current_pose.numpy()
    rest = {name: (rest_pose[index[name], :3].astype(np.float64), reference[index[name], 3:].astype(np.float64)) for name in BODY}
    tile_keys, tile_positions = [], []
    for layer, tiles in enumerate(floors.tiles):
        for q, r in sorted(tiles):
            x, y = hex_center(q, r)
            tile_keys.append((layer, q, r))
            tile_positions.append((x, y, -LAYER_HEIGHT * layer))
    tile_number = {key: i for i, key in enumerate(tile_keys)}
    chain = [(name, parent) for name, parent in skeleton.chain]
    bodies = Bodies(chain, rest, tile_positions, HEX_RADIUS, MASSES, rail_capsules(radii, LAYER_HEIGHT))
    for runner in runners:
        bodies.write_pose(runner.index, runner.character.current_pose.numpy(), index, PEDESTAL_HEIGHT, (0.0, 0.0))
    if solo:
        for runner in runners[1:]:
            runner.eliminated_at = 0.0
            bodies.park(runner.index)

    hips_frames, joint_frames = [], []
    winner_at = None
    solo_fell = None
    sudden_death = False
    floors.log("countdown")
    for frame in range(int(seconds * FPS)):
        # Once there is a winner the floor stops vanishing, like the end of a round.
        if winner_at is None:
            for layer, (q, r) in floors.update():
                bodies.remove_tile(tile_number[(layer, q, r)])
        if winner_at is None and not sudden_death and floors.time >= SUDDEN_DEATH_SECONDS:
            sudden_death = True
            floors.log("sudden_death", None)
            for r in runners:
                if r.eliminated_at is None:
                    print("STUCK", seed, r.index, "limp", r.limp, "layer", r.layer, "lift", r.lift, "ground", r.ground, "direction", r.direction, "speed", r.character.actual_velocity[:2], "no_trigger", r.no_trigger_until, "leaping", r.leaping_until, "hips", bodies.hips_position(r.index), "vel", bodies.hips_velocity(r.index), "tiles", r.tiles_broken, "pending", r.pending_getup, flush=True)
                    # Nothing else ends a body that came to rest somewhere it should not be.
                    if r.limp is not None:
                        v = bodies.root_v[r.index]
                        bodies.data.qvel[v:v + 3] = 0.0, 0.0, -5.0
            for layer, tiles in enumerate(floors.tiles):
                for q, r in tiles:
                    if (q, r) not in floors.gone[layer]:
                        floors.gone[layer].add((q, r))
                        floors.touched[layer].pop((q, r), None)
                        floors.log("tile_gone", None, layer=layer, q=q, r=r)
                        bodies.remove_tile(tile_number[(layer, q, r)])
            bodies.remove_rails()
        started = floors.time >= COUNTDOWN_SECONDS and winner_at is None
        if frame == int(COUNTDOWN_SECONDS * FPS):
            floors.log("go")
        hips = np.full((RUNNERS, 3), np.nan, dtype=np.float32)
        relative = np.zeros((RUNNERS, len(RECORDED), 3), dtype=np.int16)
        alive = [r for r in runners if r.eliminated_at is None]
        decisions = None
        if brain is not None and started and frame % BRAIN_FRAMES == 0:
            decisions = brain.decide(floors, runners)
        tiles_touched, hits = bodies.step_frame()
        for (first, second), speed in hits.items():
            # Whoever is flying hits whoever is standing, and passes on some speed.
            for flying, target in ((runners[first], runners[second]), (runners[second], runners[first])):
                if flying.limp is None or target.limp is not None or target.eliminated_at is not None or speed < KNOCK_BY_LIMP_SPEED:
                    continue
                if target.lift > 0.0 or floors.time < target.no_trigger_until or winner_at is not None:
                    continue
                push = bodies.hips_velocity(flying.index) * (MASSES[flying.index] / MASSES[target.index]) * 0.5
                push[2] = max(push[2], 0.5)
                push[:2] += target.character.actual_velocity[:2] * 0.5
                flying.knocks_dealt += 1
                go_limp(floors, bodies, target, push, "knock", by=flying.index)
        for runner in alive:
            character = runner.character
            limp_pose = None
            if runner.limp is not None:
                limp_pose = step_limp(floors, bodies, runner, tiles_touched[runner.index], tile_keys, index)
                if runner.eliminated_at is not None:
                    continue
            else:
                x, y = float(runner.ground[0]), float(runner.ground[1])
                if runner.pedestal is not None and math.hypot(x - runner.pedestal[0], y - runner.pedestal[1]) > HEX_RADIUS * 0.8:
                    floors.log("pedestal_gone", runner.index)
                    runner.pedestal = None
                if runner.pedestal is None and runner.lift > 0.0:
                    # Stepping off the start hexagon. The legs keep running on the way down.
                    runner.lift_speed -= 9.81 * DT
                    runner.lift = max(runner.lift + runner.lift_speed * DT, 0.0)
                    if runner.dropping and runner.lift <= 0.0:
                        if floors.has_floor(runner.layer, x, y):
                            runner.dropping = False
                            runner.land_grace_until = floors.time + LAND_GRACE_SECONDS
                            floors.log("land", runner.index, layer=runner.layer, x=round(x, 2), y=round(y, 2))
                        elif runner.layer < LAYERS - 1:
                            runner.layer += 1
                            floors.log("pass", runner.index, layer=runner.layer)
                            runner.lift = LAYER_HEIGHT
                        else:
                            # Nothing left below but the slime, so physics takes the last plunge.
                            runner.dropping = False
                            go_limp(floors, bodies, runner, np.array([character.actual_velocity[0] * 0.7, character.actual_velocity[1] * 0.7, runner.lift_speed]), "slime")
                on_floor = runner.lift <= 0.0
                blocked = on_floor and not floors.safe(runner.layer, x + runner.direction[0] * BLOCK_AHEAD, y + runner.direction[1] * BLOCK_AHEAD)
                # Holding a direction matters: turning every few frames regenerates motion constantly and the runner barely moves.
                brained = runner.index in thinking
                if decisions is not None and brained and on_floor and not character.state_machine.is_interacting:
                    turn = decisions[0][runner.index]
                    gait = decisions[1][runner.index]
                    if gait != runner.gait:
                        runner.gait = gait
                        runner.strength = GAIT_STRENGTH[gait]
                        floors.log("gait", runner.index, gait=gait)
                    turn = (turn - HEADINGS if turn > HEADINGS // 2 else turn) * (2 * math.pi / HEADINGS)
                    angle = math.atan2(runner.direction[1], runner.direction[0]) + turn
                    runner.direction = np.array([math.cos(angle), math.sin(angle)])
                    moving = float(np.linalg.norm(character.actual_velocity[:2]))
                    if leaps_allowed and (solo or (floors.time - COUNTDOWN_SECONDS >= leap_after and way_blocked(floors, runner.layer, x, y, runner.direction, block_sectors))) and decisions[2][runner.index] == MOVE_LEAP and abs(turn) < 1.5 * (2 * math.pi / HEADINGS) and moving > 1.5 and floors.time >= runner.leap_ready_at and floors.time >= runner.no_trigger_until:
                        start_leap(floors, runner, runner.direction, gap_ahead(floors, runner.layer, x, y, runner.direction), engine, asset_pool, leap_reference, Leap)
                        floors.events[-1]["blocked"] = True
                        runner.leap_ready_at = floors.time + LEAP_SECONDS + leap_cooldown
                if not brained and started and on_floor and not character.state_machine.is_interacting and frame % LEAP_LOOK_FRAMES == runner.index % LEAP_LOOK_FRAMES:
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
                if not brained and started and on_floor and ((blocked and frame - runner.last_turn >= 5) or frame - runner.last_turn >= THINK_FRAMES):
                    runner.last_turn = frame
                    others = [
                        (float(o.ground[0]), float(o.ground[1]))
                        for o in alive if o is not runner and o.layer == runner.layer
                    ]
                    if not character.state_machine.is_interacting:
                        runner.direction = choose_direction(floors, runner, x, y, runner.direction, rng, others)
                if runner.limp is None:
                    move = np.array([runner.direction[0], runner.direction[1], 0.0], dtype=np.float32)
                    if started:
                        character.step(DT, LocomotionCommand(move, move * runner.strength, use_max_speed=True))
                    elif winner_at is not None and (floors.time - winner_at < WIN_HOLD_AFTER_SECONDS or runner.pending_getup is not None):
                        # A winner caught mid-tumble still has to stand up for its crown.
                        character.step(DT, LocomotionCommand(move, move * SLOW_STRENGTH))
                    if character._playback_source is PlaybackSource.CLIP:
                        raise RuntimeError(f"a sample clip played at {floors.time:.2f} s")
            pose = limp_pose if limp_pose is not None else character.current_pose.numpy().copy()
            if not np.isfinite(pose).all():
                raise RuntimeError(f"runner {runner.index}: non-finite pose at {floors.time:.2f} s")
            joints = skeleton.world(pose)
            shift3 = np.array([runner.shift[0], runner.shift[1], 0.0])
            joints = {name: position + shift3 for name, position in joints.items()}
            runner.ground = joints["Hips"][:2].copy()
            if runner.limp is None and limp_pose is None:
                bodies.write_pose(runner.index, pose, index, -LAYER_HEIGHT * runner.layer + runner.lift, runner.shift)
                root_v = bodies.root_v[runner.index]
                bodies.data.qvel[root_v:root_v + 2] = character.actual_velocity[:2]
            if runner.limp is None:
                feet = {name: joints[name] for name in FEET}
                if winner_at is None and runner.lift <= 0.0 and floors.time >= runner.no_trigger_until:
                    for name, position in feet.items():
                        if position[2] < STEP_HEIGHT and floors.step_on(runner.layer, position[0], position[1], runner.index):
                            runner.tiles_broken += 1
                if runner.previous_feet is not None:
                    for name, position in feet.items():
                        if position[2] < 0.04 and runner.previous_feet[name][2] < 0.04:
                            runner.slides.append(float(np.linalg.norm((position - runner.previous_feet[name])[:2]) * FPS))
                runner.previous_feet = feet
                if started:
                    runner.speeds.append(float(np.linalg.norm(character.actual_velocity[:2])))
                root_x, root_y = joints["Hips"][:2]
                leaping = floors.time < runner.leaping_until
                planted = [position for position in feet.values() if position[2] < AIRBORNE_HEIGHT]
                grounded = bool(planted)
                # A planted foot on a tile holds the runner up, even with the hips over a hole.
                supported = any(floors.has_floor(runner.layer, position[0], position[1]) for position in planted)
                # A body getting up sprawls past its tile, so while it rises only the hips count.
                if (runner.pending_getup is not None or floors.time < runner.land_grace_until) and floors.has_floor(runner.layer, root_x, root_y):
                    supported = True
                if runner.leap_from is not None:
                    if not grounded:
                        runner.leap_airborne = True
                    elif runner.leap_airborne:
                        from_x, from_y, from_t = runner.leap_from
                        over_floor = supported
                        floors.log("leap_touch", runner.index, seconds=round(floors.time - from_t, 3), distance=round(math.hypot(root_x - from_x, root_y - from_y), 3), floor=bool(over_floor))
                        runner.leap_from = None
                        if over_floor:
                            runner.leaps_landed += 1
                            runner.leaping_until = -1.0
                            runner.leap_ready_at = floors.time + leap_cooldown
                            runner.landed_at = floors.time
                            floors.log("leap_landed", runner.index)
                            # The generated landing can play on for a moment before steering comes back.
                            runner.leap_release = floors.time + land_hold
                if runner.leap_release is not None and floors.time >= runner.leap_release:
                    runner.leap_release = None
                    if character.state_machine.is_interacting:
                        character.state_machine.transition_to_locomotion(character)
                on_floor = runner.lift <= 0.0
                winding_up = leaping and not runner.leap_airborne
                if winner_at is None and on_floor and grounded and not supported and not winding_up and floors.time >= runner.no_trigger_until - 1.0:
                    if leaping:
                        runner.leaps_failed += 1
                        runner.leaping_until = -1.0
                    cause = "leap" if leaping else ("hole" if floors.tile_at(runner.layer, root_x, root_y) else "edge")
                    if runner.layer < LAYERS - 1:
                        drop_through(floors, runner, cause)
                    else:
                        go_limp(floors, bodies, runner, np.array([character.actual_velocity[0] * 0.7, character.actual_velocity[1] * 0.7, 0.0]), cause)
                elif runner.leaping_until > 0.0 and not leaping:
                    runner.leaping_until = -1.0
                    runner.leap_from = None
            else:
                runner.previous_feet = None
            if runner.pending_getup is not None and joints["HeadEnd"][2] > 1.35:
                runner.getup_seconds.append(floors.time - runner.pending_getup)
                runner.pending_getup = None
            root = joints["Hips"]
            hips[runner.index] = root + np.array([0.0, 0.0, -LAYER_HEIGHT * runner.layer + runner.lift])
            for j, name in enumerate(RECORDED):
                relative[runner.index, j] = np.clip((joints[name] - root) * 1000.0, -32000, 32000)
        separate(floors, bodies, runners)
        engine.synchronize()
        hips_frames.append(hips)
        joint_frames.append(relative)
        floors.time += DT
        remaining = [r for r in runners if r.eliminated_at is None]
        if solo:
            if runners[0].falls > 0 and solo_fell is None:
                solo_fell = floors.time
                floors.log("solo_over", 0, seconds=round(floors.time - COUNTDOWN_SECONDS, 2), tiles=runners[0].tiles_broken)
            if solo_fell is not None and floors.time > solo_fell + 2.5:
                break
        elif len(remaining) <= 1 and winner_at is None:
            winner_at = floors.time
            winner = remaining[0] if remaining else max(runners, key=lambda r: r.eliminated_at)
            floors.log("winner", winner.index)
        if winner_at is not None and (floors.time - winner_at > WIN_RUNOFF_SECONDS or winner.eliminated_at is not None):
            break
    if winner_at is None and not solo:
        if seconds >= MAX_SECONDS:
            raise RuntimeError(f"seed {seed}: no winner after {seconds} s, {len(remaining)} left")
        floors.log("cut_short", None, left=len(remaining))

    slides = np.array([s for r in runners for s in r.slides])
    meta = {
        "seed": seed, "fps": FPS, "commit": COMMIT, "frames": len(hips_frames), "joints": RECORDED,
        "arena": {
            "layers": LAYERS, "layerHeight": LAYER_HEIGHT, "hexRadius": HEX_RADIUS, "slimeDepth": SLIME_DEPTH,
            "touchSeconds": TOUCH_SECONDS, "countdownSeconds": COUNTDOWN_SECONDS, "pedestalHeight": PEDESTAL_HEIGHT,
            "radii": radii, "tiles": [sorted(tiles) for tiles in floors.tiles],
            "pedestals": pedestals,
        },
        "runners": [
            {
                "name": NAMES[r.index], "color": COLORS[r.index], "gear": GEAR[r.index], "bulk": 1.0,
                "sight": r.sight, "noise": r.noise, "strength": r.strength, "falls": r.falls,
                "leaps": r.leaps, "leapsLanded": r.leaps_landed, "leapsFailed": r.leaps_failed,
                "eliminatedAt": r.eliminated_at, "tilesTouched": r.tiles_broken,
                "getupSeconds": r.getup_seconds, "unrecovered": r.pending_getup is not None,
                "brain": r.index in thinking, "knocksDealt": r.knocks_dealt, "mass": MASSES[r.index],
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
def one_round(seed: int, seconds: float, brains: str, policy: str, heading_temperature: float, move_temperature: float, leap_cooldown: float, land_hold: float, leaps_allowed: bool, leap_after: float, block_sectors: int, leap_shape: str, solo_radius: float) -> tuple[dict, bytes]:
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
    reference = assets.reference_pose.numpy() if hasattr(assets.reference_pose, "numpy") else assets.reference_pose
    startup = time.perf_counter() - startup
    with wp.ScopedStream(engine.stream):
        t = time.perf_counter()
        meta, data = simulate(engine, asset_pool, skeleton, reference, seed, seconds, leap_reference, brains, policy, heading_temperature, move_temperature, leap_cooldown, land_hold, leaps_allowed, leap_after, block_sectors, leap_shape, solo_radius)
    meta["timing"] = {"startupSeconds": startup, "simulateSeconds": time.perf_counter() - t}
    volume.commit()
    print(f"seed {seed}: {meta['frames']} frames, {meta['timing']}")
    return meta, data


@app.local_entrypoint()
def main(seeds: str = "1", seconds: float = MAX_SECONDS, brains: str = "all", policy: str = "hop", folder: str = "hex", heading_temperature: float = BRAIN_TEMPERATURE, move_temperature: float = BRAIN_TEMPERATURE, leap_cooldown: float = 1.0, land_hold: float = 0.0, no_leaps: bool = False, leap_after: float = 10.0, block_sectors: int = 1, leap_shape: str = "1.6,1.2", solo_radius: float = 0.0):
    out = Path(__file__).resolve().parent.parent / "runs" / folder
    out.mkdir(parents=True, exist_ok=True)
    # One container per seed, so three rounds take about as long as one.
    numbers = [int(s) for s in seeds.split(",")]
    failures = []
    for result in one_round.starmap([(n, seconds, brains, policy, heading_temperature, move_temperature, leap_cooldown, land_hold, not no_leaps, leap_after, block_sectors, leap_shape, solo_radius) for n in numbers], return_exceptions=True):
        if isinstance(result, BaseException):
            failures.append(result)
            print("FAILED", result)
            continue
        meta, data = result
        (out / f"seed_{meta['seed']}.json").write_text(json.dumps(meta))
        (out / f"seed_{meta['seed']}.bin").write_bytes(data)
        winner = next((e["runner"] for e in meta["events"] if e["type"] == "winner"), None)
        kinds = {}
        for event in meta["events"]:
            kinds[event["type"]] = kinds.get(event["type"], 0) + 1
        print(f"seed {meta['seed']}: {meta['frames'] / FPS:.1f} s, winner {winner}, {len(data) / 1e6:.1f} MB, slide {meta['footSlide']}, {kinds}")
        for r in meta["runners"]:
            print(" ", r["name"], "out", r["eliminatedAt"] and round(r["eliminatedAt"], 1), "falls", r["falls"], "leaps", r["leaps"], r["leapsLanded"], r["leapsFailed"], "tiles", r["tilesTouched"], "speed", round(r["medianSpeed"], 2), [round(s, 2) for s in r["getupSeconds"]], r["unrecovered"])
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(numbers)} rounds failed")
