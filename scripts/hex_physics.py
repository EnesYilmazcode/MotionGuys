"""Real bodies for the moments MotionBricks cannot do: knockdowns, dives, and falls.

Every robot exists in one MuJoCo scene, one rigid capsule per skeleton bone, on real hexagon floors. While a robot runs
its joints are simply copied in from MotionBricks and it can only be hit. When it goes limp the copy stops, gravity and
contact take over, and the resulting pose goes back to MotionBricks for a generated get-up.
"""

import math

import mujoco
import numpy as np

BODY_FPS_SUBSTEPS = 16
TIMESTEP = 1.0 / 30.0 / BODY_FPS_SUBSTEPS
TILE_THICKNESS = 0.42
TILE_BIT = 1 << 8
RAIL_BIT = 1 << 9
ALL_ROBOTS = (1 << 8) - 1

# Radius, and the bone it spans, in meters. Heavier parts follow from volume at one density.
BONE_RADII = {
    "Hips": 0.14, "Spine1": 0.13, "Spine2": 0.14, "Chest": 0.16, "Neck1": 0.05, "Neck2": 0.05, "Head": 0.11, "HeadEnd": 0.11,
    "Shoulder": 0.05, "Arm": 0.05, "ForeArm": 0.04, "Hand": 0.05, "Leg": 0.075, "Shin": 0.055, "Foot": 0.05, "ToeBase": 0.04, "ToeEnd": 0.03,
}
# How far a joint may bend from its rest pose, in degrees, which is what MuJoCo reads from a range by default.
BONE_RANGE = {
    "Spine1": 35, "Spine2": 35, "Chest": 35, "Neck1": 45, "Neck2": 45, "Head": 45, "Shoulder": 25, "Arm": 150, "Hand": 80,
    "Leg": 125, "Foot": 60, "ToeBase": 45,
}
# Knees and elbows only bend one way, a positive turn around local z in every recorded pose.
HINGE_RANGE = {"ForeArm": 150, "Shin": 150}


def bone_kind(name):
    return name.removeprefix("Left").removeprefix("Right")


def to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def hex_prism_vertices(radius, thickness):
    # Same corners as the replay: a corner points along +y of the honeycomb, which has rows along x.
    corners = [(radius * math.sin(k * math.pi / 3), radius * math.cos(k * math.pi / 3)) for k in range(6)]
    return " ".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y in corners for z in (0.0, -thickness))


def rail_capsules(radii, layer_height, offset=2.4):
    """The padded octagons from the replay, as (from, to, radius). Same corners as makeCage in world.ts, with x and y swapped."""
    rails = []
    for layer, radius in enumerate(radii):
        z = -layer * layer_height
        corners = [(math.sin((i + 0.5) * math.pi / 4) * (radius + offset), math.cos((i + 0.5) * math.pi / 4) * (radius + offset)) for i in range(8)]
        for i, (x, y) in enumerate(corners):
            nx, ny = corners[(i + 1) % 8]
            for lift, size in ((0.15, 0.21), (0.55, 0.21), (0.95, 0.26)):
                rails.append(((x, y, z + lift), (nx, ny, z + lift), size))
    return rails


def build_scene(chain, rest, tile_positions, hex_radius, masses, rails):
    """chain is (name, parent name) from the root down, rest is each joint's local offset and rotation at rest."""
    children = {}
    for name, parent in chain:
        children.setdefault(parent, []).append(name)

    def geoms(name, robot, density):
        kind = bone_kind(name)
        radius = BONE_RADII[kind]
        parts = []
        below = children.get(name, [])
        if not below:
            parts.append(f'<geom name="r{robot}_{name}" type="sphere" size="{radius}" density="{density}"/>')
        for child in below:
            # A hip or spine fork would pass through the torso, so only the first child of a fork gets a full capsule.
            offset = rest[child][0]
            length = float(np.linalg.norm(offset))
            if length < 1e-3:
                continue
            parts.append(
                f'<geom name="r{robot}_{name}_{child}" type="capsule" size="{min(radius, length * 0.6):.4f}" density="{density}" '
                f'fromto="0 0 0 {offset[0]:.4f} {offset[1]:.4f} {offset[2]:.4f}"/>'
            )
        return "".join(parts)

    def body(name, robot, density):
        position, rotation = rest[name]
        quat = to_wxyz(rotation)
        attributes = f'name="r{robot}_{name}" pos="{position[0]:.5f} {position[1]:.5f} {position[2]:.5f}" quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}"'
        kind = bone_kind(name)
        if name == chain[0][0]:
            joint = f'<freejoint name="r{robot}_root"/>'
            attributes = f'name="r{robot}_{name}" pos="{float(robot) * 3:.3f} 0 50"'
        elif kind in HINGE_RANGE:
            joint = f'<joint name="r{robot}_{name}" type="hinge" axis="0 0 1" limited="true" range="0 {HINGE_RANGE[kind]}" damping="1" armature="0.01"/>'
        elif kind in BONE_RANGE:
            joint = f'<joint name="r{robot}_{name}" type="ball" limited="true" range="0 {BONE_RANGE[kind]}" damping="2" armature="0.01"/>'
        else:
            joint = ""
        inner = "".join(body(child, robot, density) for child in children.get(name, []))
        return f"<body {attributes}>{joint}{geoms(name, robot, density)}{inner}</body>"

    robots = "".join(body(chain[0][0], robot, 450.0 * masses[robot]) for robot in range(len(masses)))
    tiles = "".join(
        f'<geom name="t{i}" type="mesh" mesh="hex" pos="{x:.3f} {y:.3f} {z:.3f}" contype="{TILE_BIT}" conaffinity="0" group="1"/>'
        for i, (x, y, z) in enumerate(tile_positions)
    )
    tiles += "".join(
        f'<geom name="rail{i}" type="capsule" size="{size}" fromto="{a[0]:.3f} {a[1]:.3f} {a[2]:.3f} {b[0]:.3f} {b[1]:.3f} {b[2]:.3f}" contype="{RAIL_BIT}" conaffinity="0" group="1"/>'
        for i, (a, b, size) in enumerate(rails)
    )
    return f"""<mujoco>
<option timestep="{TIMESTEP:.6f}" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic"/>
<default><geom friction="0.9 0.02 0.002" solref="0.02 1" solimp="0.9 0.95 0.002"/></default>
<size memory="256M"/>
<asset><mesh name="hex" vertex="{hex_prism_vertices(hex_radius * 0.975, TILE_THICKNESS)}"/></asset>
<worldbody>{tiles}{robots}</worldbody>
</mujoco>"""


class Bodies:
    def __init__(self, skeleton_chain, rest, tile_positions, hex_radius, masses, rails=()):
        self.chain = skeleton_chain
        self.robots = len(masses)
        self.model = mujoco.MjModel.from_xml_string(build_scene(skeleton_chain, rest, tile_positions, hex_radius, masses, rails))
        self.data = mujoco.MjData(self.model)
        m = self.model
        id_of = lambda kind, name: mujoco.mj_name2id(m, kind, name)
        self.root_q = [m.jnt_qposadr[id_of(mujoco.mjtObj.mjOBJ_JOINT, f"r{i}_root")] for i in range(self.robots)]
        self.root_v = [m.jnt_dofadr[id_of(mujoco.mjtObj.mjOBJ_JOINT, f"r{i}_root")] for i in range(self.robots)]
        self.joints = []
        for robot in range(self.robots):
            joints = {}
            for name, _ in skeleton_chain[1:]:
                joint = id_of(mujoco.mjtObj.mjOBJ_JOINT, f"r{robot}_{name}")
                if joint >= 0:
                    joints[name] = joint
            self.joints.append(joints)
        self.rest = {name: rest[name][1] for name, _ in skeleton_chain}
        self.geoms = [[g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"r{i}_")] for i in range(self.robots)]
        self.tile_geoms = np.array([id_of(mujoco.mjtObj.mjOBJ_GEOM, f"t{i}") for i in range(len(tile_positions))])
        self.tile_of_geom = {int(g): i for i, g in enumerate(self.tile_geoms)}
        self.robot_of_geom = np.full(m.ngeom, -1)
        self.bodies_of = [sorted({int(m.geom_bodyid[g]) for g in geoms}) for geoms in self.geoms]
        for robot, geoms in enumerate(self.geoms):
            self.robot_of_geom[geoms] = robot
            m.geom_contype[geoms] = 1 << robot
            # MuJoCo also filters by a per-body copy of these bits, so a change has to reach both.
            m.body_contype[self.bodies_of[robot]] = 1 << robot
        self.limp = [False] * self.robots
        self.ghost = [False] * self.robots
        self.held = [{} for _ in range(self.robots)]
        self.targets = [{} for _ in range(self.robots)]
        self.gains = [{} for _ in range(self.robots)]
        for robot in range(self.robots):
            self.set_collision(robot)

    def set_collision(self, robot):
        # A limp body hits floors and other robots. A running body is only something to be hit.
        solid = TILE_BIT if self.ghost[robot] else TILE_BIT | RAIL_BIT
        affinity = (ALL_ROBOTS & ~(1 << robot)) | solid if self.limp[robot] else 0
        self.model.geom_conaffinity[self.geoms[robot]] = affinity
        self.model.body_conaffinity[self.bodies_of[robot]] = affinity

    def pass_through_rails(self, robot):
        """A body that went limp already overlapping a rail is held there by the contacts. Let it drop."""
        self.ghost[robot] = True
        self.set_collision(robot)

    def remove_rails(self):
        for g in range(self.model.ngeom):
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("rail"):
                self.model.geom_contype[g] = 0

    def remove_tile(self, tile):
        self.model.geom_contype[self.tile_geoms[tile]] = 0

    def relative_to_rest(self, name, local):
        """A skeleton joint's local rotation as a MuJoCo ball joint value, which measures from rest."""
        rest = self.rest[name]
        conjugate = np.array([-rest[0], -rest[1], -rest[2], rest[3]])
        return to_wxyz(quat_mul(conjugate, local))

    def write_pose(self, robot, pose, index, height_offset, shift):
        """Copy a MotionBricks pose in. height_offset moves it from its floor's height to the real one."""
        q = self.data.qpos
        root = self.root_q[robot]
        hips = pose[index[self.chain[0][0]]]
        q[root:root + 3] = hips[0] + shift[0], hips[1] + shift[1], hips[2] + height_offset
        q[root + 3:root + 7] = to_wxyz(hips[3:].astype(np.float64))
        for name, joint in self.joints[robot].items():
            adr = self.model.jnt_qposadr[joint]
            value = self.relative_to_rest(name, pose[index[name]][3:].astype(np.float64))
            if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                # value is w, x, y, z. Going the short way around keeps the angle in range.
                angle = 2.0 * math.atan2(value[3], value[0])
                q[adr] = (angle + math.pi) % (2 * math.pi) - math.pi
            else:
                q[adr:adr + 4] = value

    def read_pose(self, robot, pose, index, height_offset, shift):
        """Copy a limp body back into a MotionBricks pose."""
        q = self.data.qpos
        root = self.root_q[robot]
        hips = index[self.chain[0][0]]
        pose[hips, 0] = q[root] - shift[0]
        pose[hips, 1] = q[root + 1] - shift[1]
        pose[hips, 2] = q[root + 2] - height_offset
        pose[hips, 3:] = to_xyzw(q[root + 3:root + 7])
        for name, joint in self.joints[robot].items():
            adr = self.model.jnt_qposadr[joint]
            if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                local = np.array([0.0, 0.0, math.sin(q[adr] / 2), math.cos(q[adr] / 2)])
            else:
                local = to_xyzw(q[adr:adr + 4])
            pose[index[name], 3:] = quat_mul(self.rest[name], local)

    def go_limp(self, robot, velocity=None, spin=None, stiffness=6.0):
        """From now on physics owns this robot, starting at its last motion, holding its pose softly."""
        self.limp[robot] = True
        self.set_collision(robot)
        v = self.root_v[robot]
        if velocity is not None:
            self.data.qvel[v:v + 3] = velocity
        if spin is not None:
            self.data.qvel[v + 3:v + 6] = spin
        self.hold_current_pose(robot, stiffness)

    def hold_current_pose(self, robot, stiffness):
        """Pull every joint toward the pose the robot was in, so it falls as itself and never relaxes to the rest pose."""
        self.held[robot] = {}
        for name, joint in self.joints[robot].items():
            adr = self.model.jnt_qposadr[joint]
            size = self.joint_size(joint)
            self.held[robot][name] = self.data.qpos[adr:adr + size].copy()
            self.targets[robot][name] = self.held[robot][name].copy()
            self.gains[robot][name] = stiffness

    def sway(self, robot, name, angle, stiffness):
        """Move a joint's target by a small turn away from the held pose, around the axis knees and elbows bend on."""
        joint = self.joints[robot][name]
        held = self.held[robot][name]
        if self.joint_size(joint) == 1:
            self.targets[robot][name][0] = max(float(held[0]) + angle, 0.0)
        else:
            turn = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
            mujoco.mju_mulQuat(self.targets[robot][name], held, turn)
        self.gains[robot][name] = stiffness

    def apply_joint_torques(self, robot):
        """Torque toward each target."""
        error = np.zeros(3)
        for name, joint in self.joints[robot].items():
            adr, dof = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
            gain, target = self.gains[robot][name], self.targets[robot][name]
            if self.joint_size(joint) == 1:
                self.data.qfrc_applied[dof] = gain * (target[0] - self.data.qpos[adr])
            else:
                mujoco.mju_subQuat(error, target, self.data.qpos[adr:adr + 4])
                self.data.qfrc_applied[dof:dof + 3] = gain * error

    def joint_size(self, joint):
        return 1 if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE else 4

    def set_stiffness(self, robot, stiffness):
        for name in self.joints[robot]:
            self.gains[robot][name] = stiffness

    def wake(self, robot):
        """Hand the robot back to MotionBricks."""
        self.limp[robot] = False
        self.ghost[robot] = False
        self.set_collision(robot)
        self.data.qfrc_applied[self.dof_slice(robot)] = 0.0

    def park(self, robot):
        """Out of the round. Far away and out of everything's way."""
        self.limp[robot] = False
        self.model.geom_contype[self.geoms[robot]] = 0
        self.model.geom_conaffinity[self.geoms[robot]] = 0
        self.data.qfrc_applied[self.dof_slice(robot)] = 0.0
        self.model.body_contype[self.bodies_of[robot]] = 0
        self.model.body_conaffinity[self.bodies_of[robot]] = 0
        root = self.root_q[robot]
        self.data.qpos[root:root + 3] = 1000.0 + robot * 10, 1000.0, 1000.0
        self.data.qvel[self.root_v[robot]:self.root_v[robot] + 6] = 0.0

    def dof_slice(self, robot):
        """All of a robot's velocities, which sit together because each robot is one chain of bodies."""
        start = self.root_v[robot]
        end = self.root_v[robot + 1] if robot + 1 < self.robots else self.model.nv
        return slice(start, end)

    def hips_position(self, robot):
        root = self.root_q[robot]
        return self.data.qpos[root:root + 3].copy()

    def hips_velocity(self, robot):
        v = self.root_v[robot]
        return self.data.qvel[v:v + 3].copy()

    def step_frame(self):
        """Advance one 30 Hz frame. Returns contacts: tiles under each limp robot and hits between robots."""
        tiles_touched = [set() for _ in range(self.robots)]
        hits = {}
        if not any(self.limp):
            return tiles_touched, hits
        # Running robots are frozen for the frame, so they act as solid obstacles of unlimited mass.
        frozen = [r for r in range(self.robots) if not self.limp[r]]
        saved = {r: (self.data.qpos[self.root_q[r]:self.root_q[r] + 7].copy()) for r in frozen}
        joint_saved = {r: [(self.model.jnt_qposadr[j], self.data.qpos[self.model.jnt_qposadr[j]:self.model.jnt_qposadr[j] + self.joint_size(j)].copy()) for j in self.joints[r].values()] for r in frozen}
        for _ in range(BODY_FPS_SUBSTEPS):
            for r in frozen:
                self.data.qpos[self.root_q[r]:self.root_q[r] + 7] = saved[r]
                self.data.qvel[self.dof_slice(r)] = 0.0
                for adr, value in joint_saved[r]:
                    self.data.qpos[adr:adr + len(value)] = value
            for r in range(self.robots):
                if self.limp[r]:
                    self.apply_joint_torques(r)
            mujoco.mj_step(self.model, self.data)
            for c in range(self.data.ncon):
                contact = self.data.contact[c]
                a, b = contact.geom1, contact.geom2
                ra, rb = self.robot_of_geom[a], self.robot_of_geom[b]
                if ra >= 0 and rb >= 0:
                    key = (min(ra, rb), max(ra, rb))
                    relative = np.linalg.norm(self.hips_velocity(ra) - self.hips_velocity(rb))
                    hits[key] = max(hits.get(key, 0.0), relative)
                elif ra >= 0 and b in self.tile_of_geom:
                    tiles_touched[ra].add(self.tile_of_geom[int(b)])
                elif rb >= 0 and a in self.tile_of_geom:
                    tiles_touched[rb].add(self.tile_of_geom[int(a)])
        for r in frozen:
            self.data.qvel[self.dof_slice(r)] = 0.0
        return tiles_touched, hits


def quat_mul(a, b):
    au, aw = a[:3], a[3]
    bu, bw = b[:3], b[3]
    return np.concatenate([aw * bu + bw * au + np.cross(au, bu), [aw * bw - au @ bu]])
