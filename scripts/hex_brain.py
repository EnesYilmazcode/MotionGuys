"""The trained brain: a fast copy of the hexagon round, and the network that plays it.

The copy treats each robot as a dot, so thousands of rounds run at once on a GPU. Its numbers are measured from
MotionBricks (scripts/modal_turn.py and recorded rounds): speed per gait, a stride of about 0.9 m, feet landing up to 0.5 m from
the hips, a command that takes about 0.3 s to start showing and then turns the body at about 260 degrees per second, a leap
of 3.9 m to 6.8 m in 1.2 s, a get-up of about 1.1 s. A body keeps its old heading for a moment, so a late turn walks off an edge.
The real round fills in the same state and asks `Sim.observe`, so training and play see the same inputs.

A tile is 2.08 m across and lasts 0.9 s, so about 2.3 m/s is the slowest straight crossing that survives. Slower robots burn
fewer tiles, and the network has to find out by itself how slow it can safely go.
"""

import math

import torch
from torch import nn

HEX_RADIUS = 1.2
HEX_WIDTH = math.sqrt(3) * HEX_RADIUS
FLOOR_RADII = (11.0, 12.5, 14.0, 15.5, 17.0, 18.5)
LAYERS = len(FLOOR_RADII)
LAYER_HEIGHT = 5.5
TOUCH_SECONDS = 0.9
PEDESTAL_RING = 8.2
AGENTS = 8

STEP = 0.1
SUBSTEPS = 3
THINK_SECONDS = STEP * SUBSTEPS
REACH = 11
GRID = 2 * REACH + 1

# Measured on the real model: the command strength for each gait and the speed it holds on a straight line.
GAIT_STRENGTH = (0.6, 0.7, 1.0)
GAIT_SPEED = (2.65, 3.13, 4.47)
GAIT_REACTION_SECONDS = 0.25
GAITS = len(GAIT_SPEED)
STRIDE_METERS = 0.9
FOOT_SIDE = 0.2
FOOT_REACH = (-0.1, 0.4)
TURN_RADIANS_PER_SECOND = math.radians(260)
LEAP_DISTANCE = (5.2, 1.1, 3.8, 6.9)  # Mean, spread, least, most, as measured over 4,987 real leaps.
LEAP_SECONDS = 1.2
LEAP_WINDUP = 0.4
LEAP_COOLDOWN = 1.0
# Measured over 376 real landings: a body touches down near 5.9 m/s, runs on about 4 m, and is down to 2 m/s a second later.
LAND_SPEED = 5.9
LAND_SLOW_SPEED = 2.0
LAND_CARRY_SECONDS = 1.0
LAND_NO_STEER_SECONDS = 0.5
SAVE_HOLD_SECONDS = 2.0
STUN_SECONDS = 1.7
DROP_SECONDS = 1.06
BODY_RADIUS = 0.3
KNOCK_CLOSING_SPEED = 5.0
MAX_ROUND_SECONDS = 120.0
# What burning one tile costs in training, scaled by each robot's thrift. Tripling it saved tiles but lost rounds, see docs/HEXAGONS.md.
TILE_COST = 0.04
# Every change of direction makes MotionBricks regenerate its motion, which costs real speed. Fraction lost per sector turned.
TURN_SPEED_LOSS = 0.1
# A leap earns this if it cleared a real gap and landed, and pays this if the floor ahead was fine and it leapt anyway.
SAVE_BONUS = 0.5
PLAIN_LEAP_COST = 0.15

RUN, LEAP, STUN, DROP, OUT = range(5)
HEADINGS = 12
MOVES = 2
MOVE_RUN, MOVE_LEAP = range(MOVES)

NEAR_SIDE, NEAR_STEP = 13, 1.0
CLOSE_SIDE, CLOSE_STEP = 9, 0.5
BELOW_SIDE, BELOW_STEP = 7, 2.5
RIVALS_SEEN = 3
PERSONALITY_SIZE = 3
RUN_LOOKS = 16
RUN_LOOK_STEP = 0.5
GAP_LOOKS = (1.5, 2.0, 2.5, 3.0, 3.5)
# A save means no way forward on foot: straight on and a sector to either side all break within this reach.
BLOCKED_LOOKS = (1.0, 1.5, 2.0, 2.5)
LANDING_LOOKS = (4.0, 4.6, 5.2, 5.8, 6.4)
# A landing keeps going, so what lies beyond it decides whether the leap was any good.
RUNOUT_LOOKS = (7.0, 8.0, 9.0, 10.0)
OBS_SIZE = NEAR_SIDE**2 + CLOSE_SIDE**2 + BELOW_SIDE**2 + RIVALS_SEEN * 4 + 14 + PERSONALITY_SIZE + HEADINGS * 4

# Dread of falling, thrift with tiles, aggression toward rivals. Roster order. Mass-class robots lean into contact.
PERSONALITIES = (
    (0.8, 0.8, 0.1),  # Optimus, methodical
    (0.6, 0.6, 0.3),  # Figure, balanced
    (0.5, 0.9, 0.2),  # G1, light and thrifty
    (0.4, 0.3, 0.3),  # H1, the sprinter
    (1.0, 0.7, 0.0),  # Apollo, careful
    (0.4, 0.3, 1.0),  # Atlas, the bully
    (0.6, 1.0, 0.1),  # T1, a miser
    (0.3, 0.4, 0.9),  # Petman, heavy and pushy
)


def grid_of_tiles(device):
    """Which honeycomb cells exist on each floor."""
    exists = torch.zeros(LAYERS, GRID, GRID, dtype=torch.bool, device=device)
    for layer, radius in enumerate(FLOOR_RADII):
        for q in range(-REACH, REACH + 1):
            for r in range(-REACH, REACH + 1):
                if math.hypot(HEX_WIDTH * (q + r / 2), 1.5 * HEX_RADIUS * r) <= radius:
                    exists[layer, q + REACH, r + REACH] = True
    return exists


def cell_of(x, y):
    """The honeycomb cell holding each point, by cube rounding, in grid indices. Far points clamp to an empty corner."""
    fq = (x * math.sqrt(3) / 3 - y / 3) / HEX_RADIUS
    fr = (y * 2 / 3) / HEX_RADIUS
    fs = -fq - fr
    q, r, s = fq.round(), fr.round(), fs.round()
    dq, dr, ds = (q - fq).abs(), (r - fr).abs(), (s - fs).abs()
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    q = torch.where(fix_q, -r - s, q)
    r = torch.where(fix_r, -q - s, r)
    outside = (q.abs() > REACH) | (r.abs() > REACH)
    return (q + REACH).clamp(0, GRID - 1).long(), (r + REACH).clamp(0, GRID - 1).long(), outside


class Sim:
    def __init__(self, envs, device="cpu", seed=0, save_bonus=SAVE_BONUS, plain_leap_cost=PLAIN_LEAP_COST, tile_cost=TILE_COST, strict_save=False, leap_cooldown=LEAP_COOLDOWN, land_fall_cost=0.0):
        self.strict_save, self.leap_cooldown, self.land_fall_cost = strict_save, leap_cooldown, land_fall_cost
        self.envs, self.device = envs, device
        self.save_bonus, self.plain_leap_cost, self.tile_cost = save_bonus, plain_leap_cost, tile_cost
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.exists = grid_of_tiles(device)
        self.env_index = torch.arange(envs, device=device)[:, None].expand(envs, AGENTS)
        shape = (envs, AGENTS)
        zeros = lambda: torch.zeros(shape, device=device)
        self.x, self.y, self.heading, self.speed, self.timer, self.target = zeros(), zeros(), zeros(), zeros(), zeros(), zeros()
        self.pace, self.leap_ready, self.no_touch_until, self.next_step = zeros(), zeros(), zeros(), zeros()
        self.drift_x, self.drift_y = zeros(), zeros()
        # Where the body is actually going and where it has been told to go by now. Both trail the newest command.
        self.move_angle, self.aim, self.previous_command = zeros(), zeros(), zeros()
        self.foot_x, self.foot_y, self.leap_speed = zeros(), zeros(), zeros()
        self.leap_gap = torch.zeros(shape, dtype=torch.bool, device=device)
        self.landed_at = torch.full(shape, -math.inf, device=device)
        self.save_due = torch.full(shape, math.inf, device=device)
        self.foot_side = torch.ones(shape, device=device)
        self.layer = torch.zeros(shape, dtype=torch.long, device=device)
        self.mode = torch.zeros(shape, dtype=torch.long, device=device)
        self.personality = torch.zeros(*shape, PERSONALITY_SIZE, device=device)
        self.time = torch.zeros(envs, device=device)
        self.solid = self.exists[None].repeat(envs, 1, 1, 1)
        self.touched_at = torch.full((envs, LAYERS, GRID, GRID), math.inf, device=device)
        self.reward = zeros()
        self.stats = {"falls": 0.0, "knocks": 0.0, "leaps": 0.0, "leaps_landed": 0.0, "gap_leaps": 0.0, "gap_landed": 0.0, "rounds": 0.0, "round_seconds": 0.0, "touches": 0.0, "slow": 0.0, "mid": 0.0, "fast": 0.0}
        self.reset(torch.ones(envs, dtype=torch.bool, device=device))

    def rand(self, *shape):
        return torch.rand(shape, device=self.device, generator=self.gen)

    def reset(self, mask):
        count = int(mask.sum())
        if not count:
            return
        order = self.rand(count, AGENTS).argsort(dim=1).float()
        around = 2 * math.pi * order / AGENTS
        self.x[mask], self.y[mask] = PEDESTAL_RING * around.cos(), PEDESTAL_RING * around.sin()
        self.heading[mask] = around + math.pi
        self.move_angle[mask] = around + math.pi
        self.aim[mask] = around + math.pi
        self.previous_command[mask] = around + math.pi
        self.foot_x[mask], self.foot_y[mask] = self.x[mask], self.y[mask]
        self.leap_gap[mask] = False
        self.landed_at[mask] = -math.inf
        self.save_due[mask] = math.inf
        self.speed[mask] = 0.0
        self.target[mask] = GAIT_SPEED[-1]
        self.timer[mask] = 0.0
        # Real speed varies a little from the command, so nothing can ride the edge exactly.
        self.pace[mask] = 0.95 + 0.1 * self.rand(count, AGENTS)
        self.leap_ready[mask] = 0.0
        self.no_touch_until[mask] = 0.0
        self.next_step[mask] = 0.0
        self.drift_x[mask], self.drift_y[mask] = 0.0, 0.0
        self.layer[mask] = 0
        self.mode[mask] = RUN
        self.personality[mask] = self.rand(count, AGENTS, PERSONALITY_SIZE)
        self.time[mask] = 0.0
        self.solid[mask] = self.exists
        self.touched_at[mask] = math.inf

    def lookup(self, table, layer, x, y):
        q, r, outside = cell_of(x, y)
        env = self.env_index
        for _ in range(x.dim() - env.dim()):
            env = env[..., None]
        return table[env.expand_as(x), layer.clamp(max=LAYERS - 1).expand_as(x), q, r], outside

    def has_floor(self, layer, x, y):
        value, outside = self.lookup(self.solid, layer, x, y)
        return value & ~outside

    def touch(self, x, y, mask):
        """Mark untouched tiles under these points, and charge whoever stepped on them."""
        q, r, outside = cell_of(x, y)
        env, layer = self.env_index[mask], self.layer[mask]
        agents = mask.nonzero()[:, 1]
        q, r, keep = q[mask], r[mask], ~outside[mask]
        env, layer, q, r, agents = env[keep], layer[keep], q[keep], r[keep], agents[keep]
        fresh = self.solid[env, layer, q, r] & self.touched_at[env, layer, q, r].isinf()
        self.touched_at[env[fresh], layer[fresh], q[fresh], r[fresh]] = self.time[env[fresh]]
        self.reward[env[fresh], agents[fresh]] -= self.tile_cost * self.personality[env[fresh], agents[fresh], 1]
        self.stats["touches"] += float(fresh.sum())

    def begin_drop(self, mask):
        # A leap that ends in a fall right away saved nothing.
        now = self.time[:, None].expand_as(self.x)
        self.reward -= (mask & (now - self.landed_at < SAVE_HOLD_SECONDS)) * self.land_fall_cost
        self.save_due[mask] = math.inf
        self.mode[mask] = DROP
        self.timer[mask] = DROP_SECONDS
        # A dropping robot keeps running through the air and lands on its feet.
        self.drift_x[mask] = self.move_angle[mask].cos() * self.speed[mask]
        self.drift_y[mask] = self.move_angle[mask].sin() * self.speed[mask]
        self.speed[mask] = 0.0
        self.stats["falls"] += float(mask.sum())

    def knock(self, mask):
        self.mode[mask] = STUN
        self.timer[mask] = STUN_SECONDS
        self.no_touch_until[mask] = self.time[:, None].expand_as(mask)[mask] + STUN_SECONDS
        self.speed[mask] = 0.0

    def apply_actions(self, heading_action, gait_action, move_action):
        acting = self.mode == RUN
        turn = (heading_action.float() - (heading_action > HEADINGS // 2).float() * HEADINGS) * (2 * math.pi / HEADINGS)
        self.heading = torch.where(acting, self.heading + turn, self.heading)
        # The newest command starts to show only after a moment, so the body answers the one before it.
        self.aim = torch.where(acting, self.previous_command, self.aim)
        self.previous_command = torch.where(acting, self.heading, self.previous_command)
        sectors = turn.abs() / (2 * math.pi / HEADINGS)
        self.speed = torch.where(acting, self.speed * (1 - TURN_SPEED_LOSS * sectors).clamp(min=0.2), self.speed)
        speeds = torch.tensor(GAIT_SPEED, device=self.device)
        self.target = torch.where(acting, speeds[gait_action], self.target)
        for name, gait in (("slow", 0), ("mid", 1), ("fast", 2)):
            self.stats[name] += float((acting & (gait_action == gait)).sum())
        now = self.time[:, None].expand_as(self.x)
        straight = turn.abs() < 2 * math.pi / HEADINGS * 1.5
        leap = acting & (move_action == MOVE_LEAP) & (now >= self.leap_ready) & straight & (self.speed > 1.5)
        # Legal only with no way forward on foot, so a robot can never leap across open floor.
        if self.strict_save:
            leap = leap & self.way_is_blocked()
        self.mode[leap] = LEAP
        self.timer[leap] = LEAP_SECONDS
        # A leap goes where it was pointed, without the usual turning lag.
        self.move_angle = torch.where(leap, self.heading, self.move_angle)
        self.aim = torch.where(leap, self.heading, self.aim)
        self.previous_command = torch.where(leap, self.heading, self.previous_command)
        mean, spread, least, most = LEAP_DISTANCE
        distance = (mean + spread * torch.randn(self.x.shape, device=self.device, generator=self.gen)).clamp(least, most)
        self.leap_speed = torch.where(leap, distance / LEAP_SECONDS, self.leap_speed)
        gap = self.way_is_blocked() if self.strict_save else self.path_is_broken(GAP_LOOKS)
        self.leap_gap = torch.where(leap, gap, self.leap_gap)
        # Jumping for show is what looked dumb, so only a leap over a real gap pays.
        self.reward -= (leap & ~gap) * self.plain_leap_cost
        self.stats["leaps"] += float(leap.sum())
        self.stats["gap_leaps"] += float((leap & gap).sum())

    def safe(self, layer, x, y):
        solid, outside = self.lookup(self.solid, layer, x, y)
        touched, _ = self.lookup(self.touched_at, layer, x, y)
        return solid & ~outside & touched.isinf()

    def way_is_blocked(self):
        sector = 2 * math.pi / HEADINGS
        return self.path_is_broken(BLOCKED_LOOKS) & self.path_is_broken(BLOCKED_LOOKS, sector) & self.path_is_broken(BLOCKED_LOOKS, -sector)

    def safe_past_own_tile(self, layer, x, y):
        """Untouched, or the tile this robot is on. Its own tile is always touched, and says nothing about the way ahead."""
        q, r, _ = cell_of(x, y)
        own_q, own_r, _ = cell_of(self.x, self.y)
        extra = (slice(None), slice(None)) + (None,) * (x.dim() - 2)
        return self.safe(layer, x, y) | ((q == own_q[extra]) & (r == own_r[extra]) & self.has_floor(layer, x, y))

    def path_is_broken(self, distances, offset=0.0):
        """Is anything missing or about to go on the way straight ahead, or at this angle from it?"""
        cos, sin = (self.heading + offset).cos()[..., None], (self.heading + offset).sin()[..., None]
        along = torch.tensor(distances, device=self.device)[None, None]
        return ~self.safe_past_own_tile(self.layer[..., None], self.x[..., None] + cos * along, self.y[..., None] + sin * along).all(dim=-1)

    def substep(self):
        self.time += STEP
        vanished = self.touched_at + TOUCH_SECONDS <= self.time[:, None, None, None]
        self.solid &= ~vanished
        self.touched_at[vanished] = math.inf

        now = self.time[:, None].expand_as(self.x)
        run, leap, drop = self.mode == RUN, self.mode == LEAP, self.mode == DROP
        # Measured: a hard turn costs most of the speed at first, and a body comes round at a limited rate.
        miss = torch.remainder(self.aim - self.move_angle + math.pi, 2 * math.pi) - math.pi
        limit = TURN_RADIANS_PER_SECOND * STEP
        age = now - self.landed_at
        steering = run & (age >= LAND_NO_STEER_SECONDS)
        self.move_angle = self.move_angle + torch.where(steering, miss.clamp(-limit, limit), torch.zeros_like(miss))
        slowed = 0.3 + 0.7 * ((1 + miss.cos()) / 2).sqrt()
        goal = run * self.target * self.pace * slowed
        self.speed = self.speed + (goal - self.speed) * min(STEP / GAIT_REACTION_SECONDS, 1.0)
        carried = LAND_SPEED + (LAND_SLOW_SPEED - LAND_SPEED) * (age / LAND_CARRY_SECONDS)
        self.speed = torch.where(run & (age < LAND_CARRY_SECONDS), carried, self.speed)
        speed = torch.where(leap, self.leap_speed, self.speed)
        self.x += self.move_angle.cos() * speed * STEP * (run | leap)
        self.y += self.move_angle.sin() * speed * STEP * (run | leap)
        self.x += self.drift_x * STEP * drop
        self.y += self.drift_y * STEP * drop

        # Robots on the same floor push each other apart. A hard head-on hit knocks one over.
        dx = self.x[:, :, None] - self.x[:, None, :]
        dy = self.y[:, :, None] - self.y[:, None, :]
        distance = (dx * dx + dy * dy).sqrt().clamp(min=1e-4)
        same = (self.layer[:, :, None] == self.layer[:, None, :]) & ~torch.eye(AGENTS, dtype=torch.bool, device=self.device)
        grounded = run | (self.mode == STUN) | leap
        pair = same & grounded[:, :, None] & grounded[:, None, :]
        overlap = (2 * BODY_RADIUS - distance).clamp(min=0) * pair
        self.x += (overlap * 0.5 * dx / distance).sum(dim=2)
        self.y += (overlap * 0.5 * dy / distance).sum(dim=2)

        vx, vy = self.move_angle.cos() * speed, self.move_angle.sin() * speed
        closing = -((vx[:, :, None] - vx[:, None, :]) * dx + (vy[:, :, None] - vy[:, None, :]) * dy) / distance
        hard = (overlap > 0) & (closing > KNOCK_CLOSING_SPEED) & run[:, :, None] & run[:, None, :]
        loser = hard & (self.rand(self.envs, AGENTS, AGENTS) < 0.5)
        knocked = loser.any(dim=2)
        self.reward += (loser.float() * self.personality[:, None, :, 2].expand_as(loser)).sum(dim=1) * 1.0
        self.stats["knocks"] += float(knocked.sum())
        self.knock(knocked)
        run = self.mode == RUN

        # A foot lands every stride, a little ahead and to one side of the hips. Only a foot touches a tile.
        winding_up = leap & (self.timer > LEAP_SECONDS - LEAP_WINDUP)
        footfall = (run | winding_up) & (now >= self.next_step)
        cos, sin = self.move_angle.cos(), self.move_angle.sin()
        least, most = FOOT_REACH
        reach = least + (most - least) * self.rand(self.envs, AGENTS)
        foot_x = self.x + cos * reach - sin * FOOT_SIDE * self.foot_side
        foot_y = self.y + sin * reach + cos * FOOT_SIDE * self.foot_side
        self.foot_x = torch.where(footfall, foot_x, self.foot_x)
        self.foot_y = torch.where(footfall, foot_y, self.foot_y)
        self.touch(foot_x, foot_y, footfall & (now >= self.no_touch_until))
        pace = speed.clamp(min=1.5)
        self.next_step = torch.where(footfall, now + STRIDE_METERS / pace, self.next_step)
        self.foot_side = torch.where(footfall, -self.foot_side, self.foot_side)

        centre = self.has_floor(self.layer, self.x, self.y)
        # A planted foot with no tile under it is a fall, the same test the real round uses. The windup of a leap is exempt.
        self.begin_drop(run & ~self.has_floor(self.layer, self.foot_x, self.foot_y))
        self.begin_drop((self.mode == STUN) & ~centre)

        self.timer -= STEP
        ended = self.timer <= 0

        landed = (self.mode == LEAP) & ended & centre
        self.mode[landed] = RUN
        self.leap_ready[landed] = now[landed] + self.leap_cooldown
        self.speed[landed] = LAND_SPEED
        self.landed_at = torch.where(landed, now, self.landed_at)
        self.foot_x = torch.where(landed, self.x, self.foot_x)
        self.foot_y = torch.where(landed, self.y, self.foot_y)
        self.next_step = torch.where(landed, now, self.next_step)
        self.stats["leaps_landed"] += float(landed.sum())
        # A save only pays once the robot is still on its feet a moment after landing.
        self.save_due = torch.where(landed & self.leap_gap, now + SAVE_HOLD_SECONDS, self.save_due)
        paid = (self.mode == RUN) & (now >= self.save_due)
        self.reward += paid * self.save_bonus
        self.save_due = torch.where(paid, torch.full_like(self.save_due, math.inf), self.save_due)
        self.stats["gap_landed"] += float(paid.sum())
        self.begin_drop((self.mode == LEAP) & ended)

        up = (self.mode == STUN) & ended
        self.mode[up] = RUN
        self.foot_x = torch.where(up, self.x, self.foot_x)
        self.foot_y = torch.where(up, self.y, self.foot_y)
        self.next_step = torch.where(up, now, self.next_step)
        self.move_angle = torch.where(up, self.heading, self.move_angle)
        self.aim = torch.where(up, self.heading, self.aim)
        self.previous_command = torch.where(up, self.heading, self.previous_command)
        self.heading[up] += (self.rand(self.envs, AGENTS)[up] - 0.5) * 0.5

        through = (self.mode == DROP) & ended
        before = self.layer.clone()
        self.layer[through] += 1
        self.reward -= through * (0.5 + 1.5 * self.personality[..., 0])
        gone = through & (before == LAYERS - 1)
        self.mode[gone] = OUT
        self.layer.clamp_(max=LAYERS - 1)
        self.reward -= gone * 2.0
        caught = through & ~gone & self.has_floor(self.layer, self.x, self.y)
        self.mode[caught] = RUN
        self.speed[caught] = GAIT_SPEED[1]
        self.foot_x = torch.where(caught, self.x, self.foot_x)
        self.foot_y = torch.where(caught, self.y, self.foot_y)
        self.next_step = torch.where(caught, now, self.next_step)
        self.aim = torch.where(caught, self.heading, self.aim)
        self.previous_command = torch.where(caught, self.heading, self.previous_command)
        self.move_angle = torch.where(caught, self.heading, self.move_angle)
        passing = through & ~gone & ~caught
        self.timer[passing] = DROP_SECONDS * 0.8

    def tick(self, heading_action, gait_action, move_action):
        """One decision. Returns rewards and which rounds ended."""
        self.reward = torch.zeros_like(self.x)
        alive_before = self.mode != OUT
        self.apply_actions(heading_action, gait_action, move_action)
        for _ in range(SUBSTEPS):
            self.substep()
        alive = self.mode != OUT
        self.reward += alive * 0.02
        left = alive.sum(dim=1)
        over = (left <= 1) | (self.time >= MAX_ROUND_SECONDS)
        winners = over[:, None] & alive & (left[:, None] == 1)
        self.reward += winners * 5.0
        self.reward *= alive_before
        reward = self.reward
        self.stats["rounds"] += float(over.sum())
        self.stats["round_seconds"] += float(self.time[over].sum())
        done = over.clone()
        self.reset(over)
        return reward, done

    def tile_state(self, layer, x, y):
        """1 for a fresh tile, down to 0 as a touched one runs out, 0 for nothing there."""
        solid, outside = self.lookup(self.solid, layer, x, y)
        touched, _ = self.lookup(self.touched_at, layer, x, y)
        extra = (slice(None), None) + (None,) * (x.dim() - 2)
        left = ((touched - self.time[extra]) + TOUCH_SECONDS).clamp(0, TOUCH_SECONDS) / TOUCH_SECONDS
        return (solid & ~outside).float() * torch.where(touched.isinf(), torch.ones_like(left), 0.5 * left)

    def observe(self):
        """What each robot sees, turned to face where it is heading."""
        cos, sin = self.heading.cos(), self.heading.sin()

        def sample(side, step):
            axis = (torch.arange(side, device=self.device) - side // 2).float() * step
            ahead, across = torch.meshgrid(axis, axis, indexing="ij")
            ahead, across = ahead.flatten()[None, None], across.flatten()[None, None]
            return self.x[..., None] + ahead * cos[..., None] - across * sin[..., None], self.y[..., None] + ahead * sin[..., None] + across * cos[..., None]

        layer = self.layer[..., None]
        near = self.tile_state(layer, *sample(NEAR_SIDE, NEAR_STEP))
        close = self.tile_state(layer, *sample(CLOSE_SIDE, CLOSE_STEP))
        x, y = sample(BELOW_SIDE, BELOW_STEP)
        below, outside = self.lookup(self.solid, (layer + 1).clamp(max=LAYERS - 1), x, y)
        below = (below & ~outside).float() * (layer < LAYERS - 1)

        dx = self.x[:, None, :] - self.x[:, :, None]
        dy = self.y[:, None, :] - self.y[:, :, None]
        distance = (dx * dx + dy * dy).sqrt()
        hidden = (self.layer[:, None, :] != self.layer[:, :, None]) | (self.mode[:, None, :] == OUT) | torch.eye(AGENTS, dtype=torch.bool, device=self.device)
        order = (distance + hidden * 1000.0).argsort(dim=2)[:, :, :RIVALS_SEEN]
        seen = hidden.gather(2, order)
        rx, ry = dx.gather(2, order), dy.gather(2, order)
        ahead = rx * cos[..., None] + ry * sin[..., None]
        across = -rx * sin[..., None] + ry * cos[..., None]
        rival_mode = self.mode[:, None, :].expand(-1, AGENTS, -1).gather(2, order)
        rivals = torch.stack([ahead / 10, across / 10, (rival_mode == LEAP).float(), (rival_mode == RUN).float()], dim=-1) * ~seen[..., None]

        # Where this robot stands on its own tile and how long that tile has left: the numbers smart pacing needs.
        q, r, _ = cell_of(self.x, self.y)
        centre_x = HEX_WIDTH * ((q - REACH).float() + (r - REACH).float() / 2)
        centre_y = 1.5 * HEX_RADIUS * (r - REACH).float()
        off_x, off_y = self.x - centre_x, self.y - centre_y
        here = self.tile_state(self.layer, self.x, self.y)
        now = self.time[:, None].expand_as(self.x)
        own = torch.stack([
            self.layer.float() / (LAYERS - 1),
            self.speed / GAIT_SPEED[-1],
            (self.mode == RUN).float(),
            (now >= self.leap_ready).float(),
            (self.mode != OUT).sum(dim=1, keepdim=True).expand_as(self.x).float() / AGENTS,
            (self.x * self.x + self.y * self.y).sqrt() / FLOOR_RADII[-1],
            here,
            (off_x * cos + off_y * sin) / HEX_RADIUS,
            (-off_x * sin + off_y * cos) / HEX_RADIUS,
            (now < self.no_touch_until).float(),
            (self.move_angle - self.heading).sin(),
            (self.move_angle - self.heading).cos(),
            self.floor_left(self.layer),
            self.floor_left((self.layer + 1).clamp(max=LAYERS - 1)) * (self.layer < LAYERS - 1),
        ], dim=-1)
        return torch.cat([near, close, below, rivals.flatten(2), own, self.personality, self.lookahead()], dim=-1)

    def floor_left(self, layer):
        """How much of a floor is left, because a robot should know when its floor is nearly gone."""
        left = self.solid.flatten(2).sum(dim=2).float() / self.exists.flatten(1).sum(dim=1).float()
        return left.gather(1, layer)

    def lookahead(self):
        """The rules of the game, spelled out per direction: how far a safe run goes, whether the way breaks, whether a leap lands well."""
        angles = self.heading[..., None] + torch.arange(HEADINGS, device=self.device).float() * (2 * math.pi / HEADINGS)
        cos, sin = angles.cos()[..., None], angles.sin()[..., None]
        layer = self.layer[..., None, None]

        def clear(distances):
            along = torch.tensor(distances, device=self.device)[None, None, None]
            return self.safe_past_own_tile(layer, self.x[..., None, None] + cos * along, self.y[..., None, None] + sin * along)

        run = clear([(k + 1) * RUN_LOOK_STEP for k in range(RUN_LOOKS)]).float().cumprod(dim=-1).sum(dim=-1) / RUN_LOOKS
        broken = (~clear(GAP_LOOKS).all(dim=-1)).float()
        landing = clear(LANDING_LOOKS).float().mean(dim=-1)
        runout = clear(RUNOUT_LOOKS).float().cumprod(dim=-1).mean(dim=-1)
        return torch.cat([run, broken, landing, runout], dim=-1)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_SIZE, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.heading = nn.Linear(256, HEADINGS)
        self.gait = nn.Linear(256, GAITS)
        self.move = nn.Linear(256, MOVES)
        self.value = nn.Linear(256, 1)

    def forward(self, obs):
        hidden = self.body(obs)
        return self.heading(hidden), self.gait(hidden), self.move(hidden), self.value(hidden).squeeze(-1)


def scripted_action(sim, sight, noise):
    """The old hand-written policy, for comparison: head for the longest safe run ahead."""
    angles = sim.heading[..., None] + torch.arange(HEADINGS, device=sim.device).float() * (2 * math.pi / HEADINGS)
    cos, sin = angles.cos(), angles.sin()
    distances = (torch.arange(16, device=sim.device).float() + 1) * 0.5
    x = sim.x[..., None, None] + cos[..., None] * distances
    y = sim.y[..., None, None] + sin[..., None] * distances
    solid, outside = sim.lookup(sim.solid, sim.layer[..., None, None], x, y)
    touched, _ = sim.lookup(sim.touched_at, sim.layer[..., None, None], x, y)
    safe = solid & ~outside & touched.isinf() & (distances <= sight[..., None, None])
    run = safe.float().cumprod(dim=-1).sum(dim=-1) * 0.5
    score = run + 1.5 * cos * sim.heading.cos()[..., None] + 1.5 * sin * sim.heading.sin()[..., None]
    score = score + noise[..., None] * torch.randn(score.shape, device=sim.device, generator=sim.gen)
    return score.argmax(dim=-1)
