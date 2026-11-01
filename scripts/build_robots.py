"""Cut robot models into rigid parts that follow the recorded skeleton.

Run:  python scripts/build_robots.py <models dir> [public/robots] [robot ...]

The recording stores joint positions, not bone rotations, and robots are rigid links anyway.
So each model is cut into one part per bone. A part is written in its bone's own frame: the origin
at the first joint, y = 1 at the second joint, x along the side hint, sizes in SOMA meters. The
replay stretches y to the bone's length that frame, so any model fits the same skeleton.

The model files are not in the repo. See CREDITS.md for where each one comes from.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

SOMA_HEIGHT = 1.78
# Part, first joint, second joint, side hint. The replay in src/robots.ts builds its frames from the same table.
PARTS = [
    ("Pelvis", "Hips", "Spine2", "pelvis"),
    ("Torso", "Spine2", "Neck1", "shoulders"),
    ("Head", "Neck1", "HeadEnd", "shoulders"),
    *[
        row
        for side in ("Left", "Right")
        for row in (
            (f"{side}UpperArm", f"{side}Arm", f"{side}ForeArm", "forward"),
            (f"{side}LowerArm", f"{side}ForeArm", f"{side}Hand", "forward"),
            (f"{side}Thigh", f"{side}Leg", f"{side}Shin", "pelvis"),
            (f"{side}Calf", f"{side}Shin", f"{side}Foot", "pelvis"),
            (f"{side}Sole", f"{side}Foot", f"{side}ToeEnd", "shin"),
        )
    ],
]
# How wide each kind of part is when deciding which part a triangle belongs to.
THICKNESS = {"Pelvis": 1.5, "Torso": 2.6, "Head": 1.5, "UpperArm": 0.7, "LowerArm": 0.7, "Thigh": 1.1, "Calf": 0.9, "Sole": 0.7}


def mirrored(joints):
    """Fill in the right side from the left, across x = center."""
    center = joints.pop("center", 0.0)
    out = {}
    for name, position in joints.items():
        position = np.array(position, dtype=float)
        if name.startswith("Left"):
            out[name] = position
            right = position.copy()
            right[0] = 2 * center - right[0]
            out[name.replace("Left", "Right")] = right
        else:
            out[name] = position
    return out


# Sketchfab meshes, with joints read off plots of each mesh. Facing +z and y up, with the character's left at +x.
MESHES = {
    "optimus": {
        "file": "optimus50k.glb",
        "joints": {
            "Hips": (0, -0.05, 0.0), "Spine2": (0, 0.2, 0.0), "Neck1": (0, 0.72, 0.0), "HeadEnd": (0, 1.0, 0.02),
            "LeftArm": (0.3, 0.6, 0.0), "LeftForeArm": (0.335, 0.22, 0.0), "LeftHand": (0.31, -0.08, 0.0),
            "LeftLeg": (0.13, -0.12, 0.0), "LeftShin": (0.15, -0.55, 0.0), "LeftFoot": (0.16, -0.9, -0.02), "LeftToeEnd": (0.16, -0.99, 0.2),
        },
    },
    "figure": {
        "file": "optimus2.glb",
        "budget": 42000,
        "paint": [{"color": (0.035, 0.037, 0.045, 1.0), "metal": 0.55, "rough": 0.3}],
        "joints": {
            "Hips": (0, 9.0, -0.3), "Spine2": (0, 10.2, -0.3), "Neck1": (0, 14.0, -0.3), "HeadEnd": (0, 16.65, 0.0),
            "LeftArm": (2.3, 13.0, -0.3), "LeftForeArm": (2.4, 10.3, -0.3), "LeftHand": (2.6, 7.7, -0.3),
            "LeftLeg": (1.0, 8.9, -0.3), "LeftShin": (1.0, 5.0, -0.3), "LeftFoot": (1.0, 1.3, -0.35), "LeftToeEnd": (1.0, 0.42, 1.6),
        },
    },
    "petman": {
        "file": "petman.glb",
        "joints": {
            "Hips": (0, 0.98, -0.07), "Spine2": (0, 1.14, -0.07), "Neck1": (0, 1.56, -0.04), "HeadEnd": (0, 1.755, -0.02),
            "LeftArm": (0.22, 1.55, -0.05), "LeftForeArm": (0.27, 1.14, -0.05), "LeftHand": (0.27, 0.85, -0.05),
            "LeftLeg": (0.12, 0.96, -0.07), "LeftShin": (0.12, 0.5, -0.07), "LeftFoot": (0.12, 0.08, -0.12), "LeftToeEnd": (0.12, 0.0, 0.02),
        },
    },
    "atlas": {
        "file": "atlas.glb",
        "shoulder_cut": 0.26,
        "paint": [{"color": (0.62, 0.64, 0.68, 1.0), "metal": 0.35, "rough": 0.4}],
        "joints": {
            "center": -0.99,
            "Hips": (-0.99, 0.75, -0.22), "Spine2": (-0.99, 0.95, -0.22), "Neck1": (-0.99, 1.38, -0.22), "HeadEnd": (-0.99, 1.63, -0.22),
            "LeftArm": (-0.79, 1.31, -0.22), "LeftForeArm": (-0.53, 1.31, -0.22), "LeftHand": (-0.27, 1.31, -0.22),
            "LeftLeg": (-0.87, 0.72, -0.22), "LeftShin": (-0.89, 0.41, -0.22), "LeftFoot": (-0.89, 0.08, -0.22), "LeftToeEnd": (-0.89, 0.0, -0.08),
        },
    },
}

# Official MuJoCo Menagerie models, posed by their own joints. Facing +x and z up, with the left at +y.
MJCF = {
    "t1": {
        "file": "menagerie/booster_t1/t1.xml",
        "joints": {"Leg": "Hip_Pitch_Left", "Shin": "Shank_Left", "Foot": "Ankle_Cross_Left", "Arm": "AL1", "ForeArm": "AL3", "Hand": "left_hand_link"},
        "spine": "Waist",
        "neck": "H1",
        "links": {
            r"Trunk": "Torso", r"^H[12]$": "Head", r"^A[LR][12]$": "UpperArm", r"^A[LR]3$|hand": "LowerArm",
            r"Waist": "Pelvis", r"Hip": "Thigh", r"Shank": "Calf", r"Ankle|foot": "Sole",
        },
    },
    "g1": {
        "file": "menagerie/unitree_g1/g1.xml",
        "joints": {"Leg": "left_hip_pitch_link", "Shin": "left_knee_link", "Foot": "left_ankle_pitch_link", "Arm": "left_shoulder_roll_link", "ForeArm": "left_elbow_link", "Hand": "left_wrist_yaw_link"},
        "spine": "waist_yaw_link",
        "links": {
            r"pelvis": "Pelvis", r"waist|torso": "Torso", r"shoulder_pitch|shoulder_roll|shoulder_yaw": "UpperArm",
            r"elbow|wrist": "LowerArm", r"hip": "Thigh", r"knee": "Calf", r"ankle": "Sole",
        },
    },
    "h1": {
        "file": "menagerie/unitree_h1/h1.xml",
        "joints": {"Leg": "left_hip_pitch_link", "Shin": "left_knee_link", "Foot": "left_ankle_link", "Arm": "left_shoulder_roll_link", "ForeArm": "left_elbow_link"},
        "spine": "torso_link",
        "links": {
            r"pelvis": "Pelvis", r"torso": "Torso", r"shoulder": "UpperArm", r"elbow": "LowerArm",
            r"hip": "Thigh", r"knee": "Calf", r"ankle": "Sole",
        },
    },
    "apollo": {
        "file": "menagerie/apptronik_apollo/apptronik_apollo.xml",
        "joints": {"Leg": "l_hip_fe_link", "Shin": "l_knee_fe_link", "Foot": "l_ankle_ie_link", "Arm": "l_shoulder_fe_link", "ForeArm": "l_elbow_fe_link", "Hand": "l_wrist_pitch_link"},
        "spine": "torso_roll_link",
        "neck": "neck_yaw_link",
        "skip": r"oak_d|world_link",
        "links": {
            r"base": "Pelvis", r"neck": "Head", r"torso": "Torso", r"shoulder": "UpperArm", r"elbow|wrist": "LowerArm",
            r"hip": "Thigh", r"knee": "Calf", r"ankle|foot": "Sole",
        },
    },
}


def bone_frame(start, end, hint):
    """The part's axes in the model. y along the bone, x as close to the hint as stays square to it."""
    y = end - start
    length = float(np.linalg.norm(y))
    y = y / length
    x = hint - y * float(hint @ y)
    x = x / np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z]), length


def hints(joints):
    up = joints["Neck1"] - joints["Hips"]
    shoulders = joints["RightArm"] - joints["LeftArm"]
    pelvis = joints["RightLeg"] - joints["LeftLeg"]
    # The up cross right is forward when the left is +x and forward is +z. The replay uses the same formula.
    forward = np.cross(up, shoulders)
    return {"pelvis": pelvis, "shoulders": shoulders, "forward": forward}


def to_part(vertices, joints, part, side_hints, scale):
    name, first, second, hint = part
    start, end = joints[first], joints[second]
    if hint == "shin":
        hint_vector = joints[first.replace("Foot", "Shin")] - start
    else:
        hint_vector = side_hints[hint]
    axes, length = bone_frame(start, end, hint_vector)
    local = (vertices - start) @ axes.T
    return np.column_stack([local[:, 0] * scale, local[:, 1] / length, local[:, 2] * scale])


def segment_distance(points, start, end):
    along = end - start
    t = np.clip(((points - start) @ along) / (along @ along), 0.0, 1.0)
    return np.linalg.norm(points - (start + t[:, None] * along), axis=1)


def kind_of(part):
    return re.sub(r"^(Left|Right)", "", part)


def assign_faces(centers, joints, center_x, shoulder_cut=0.0):
    """The part each triangle belongs to: the bone it is closest to, for that bone's thickness."""
    scores = []
    for name, first, second, _ in PARTS:
        start, end = joints[first], joints[second]
        # The hand hangs past the wrist joint, often right beside the thigh.
        if kind_of(name) == "LowerArm":
            end = end + (end - start) * 0.8
        distance = segment_distance(centers, start, end) / THICKNESS[kind_of(name)]
        # A T-pose shoulder stays on the torso, or it swings out as a spike when the arm comes down.
        if kind_of(name) == "UpperArm":
            distance = np.where(np.abs(centers[:, 0] - center_x) < shoulder_cut, np.inf, distance)
        # Limbs never reach across the body.
        if name.startswith("Left"):
            distance = np.where(centers[:, 0] < center_x - 0.01, np.inf, distance)
        if name.startswith("Right"):
            distance = np.where(centers[:, 0] > center_x + 0.01, np.inf, distance)
        scores.append(distance)
    return np.argmin(np.stack(scores, axis=1), axis=1)


def shrink_texture(material, size=1024):
    for field in ("baseColorTexture", "metallicRoughnessTexture", "normalTexture", "emissiveTexture", "occlusionTexture"):
        image = getattr(material, field, None)
        if isinstance(image, Image.Image) and max(image.size) > size:
            setattr(material, field, image.resize((size, size), Image.LANCZOS))


def write_parts(pieces, joints, out, title):
    """pieces: (part index, vertices, faces, visual or color) in model space."""
    side_hints = hints(joints)
    scale = SOMA_HEIGHT / float(joints["HeadEnd"][1] - min(joints["LeftToeEnd"][1], joints["LeftFoot"][1]))
    scene = trimesh.Scene()
    count = {}
    faces = 0
    for index, vertices, triangles, visual in pieces:
        part = PARTS[index]
        local = to_part(vertices, joints, part, side_hints, scale)
        mesh = trimesh.Trimesh(local, triangles, process=False)
        mesh.visual = visual
        number = count.get(part[0], 0)
        count[part[0]] = number + 1
        scene.add_geometry(mesh, node_name=f"{part[0]}__{number}", geom_name=f"{part[0]}__{number}")
        faces += len(triangles)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(scene.export(file_type="glb"))
    print(f"{title}: {out.stat().st_size / 1e6:.2f} MB, {faces} faces, parts {sorted(count)}")


def build_mesh(name, spec, models, out_dir):
    scene = trimesh.load(models / spec["file"])
    joints = mirrored(dict(spec["joints"]))
    center_x = spec["joints"].get("center", 0.0)
    joints = {k: np.array(v, dtype=float) for k, v in joints.items()}
    paint = spec.get("paint")
    pieces = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name]
        vertices = trimesh.transform_points(mesh.vertices, transform)
        centers = vertices[mesh.faces].mean(axis=1)
        owner = assign_faces(centers, joints, center_x, spec.get("shoulder_cut", 0.0))
        visual = mesh.visual
        if paint:
            material = trimesh.visual.material.PBRMaterial(baseColorFactor=paint[0]["color"], metallicFactor=paint[0]["metal"], roughnessFactor=paint[0]["rough"], doubleSided=True)
        elif hasattr(visual, "material"):
            material = visual.material
            shrink_texture(material)
        for index in range(len(PARTS)):
            selected = np.nonzero(owner == index)[0]
            if not len(selected):
                continue
            used, inverse = np.unique(mesh.faces[selected], return_inverse=True)
            triangles = inverse.reshape(-1, 3)
            uv = getattr(visual, "uv", None)
            part_visual = trimesh.visual.TextureVisuals(uv=None if uv is None or paint else uv[used], material=material)
            part_vertices = vertices[used]
            # Only painted models can be simplified, since this drops texture coordinates.
            if paint and "budget" in spec:
                target = max(int(len(triangles) * spec["budget"] / max(sum(len(scene.geometry[scene.graph[n][1]].faces) for n in scene.graph.nodes_geometry), 1)), 40)
                if target < len(triangles):
                    simple = trimesh.Trimesh(part_vertices, triangles, process=True).simplify_quadric_decimation(face_count=target)
                    part_vertices, triangles = np.asarray(simple.vertices), np.asarray(simple.faces)
            pieces.append((index, part_vertices, triangles, part_visual))
    write_parts(pieces, joints, out_dir / f"{name}.glb", name)
    return joints


def build_mjcf(name, spec, models, out_dir):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(models / spec["file"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def canonical(p):
        # MuJoCo is +x forward, +z up, +y left. This is the matching cyclic swap, so handedness survives.
        p = np.asarray(p)
        return np.stack([p[..., 1], p[..., 2], p[..., 0]], axis=-1)

    def body_position(body):
        return canonical(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])

    groups = []
    skip = spec.get("skip")
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[g] or model.geom_conaffinity[g]:
            continue
        body = model.body(model.geom_bodyid[g]).name
        if skip and re.search(skip, body):
            continue
        mesh_id = model.geom_dataid[g]
        v0, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        f0, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        vertices = model.mesh_vert[v0:v0 + vn] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        vertices = canonical(vertices)
        faces = model.mesh_face[f0:f0 + fn].copy()
        material = model.geom_matid[g]
        color = model.mat_rgba[material] if material >= 0 else model.geom_rgba[g]
        groups.append((body, vertices, faces, tuple(float(c) for c in color)))
        low, high = np.minimum(low, vertices.min(0)), np.maximum(high, vertices.max(0))

    named = spec["joints"]
    left = {key: body_position(body) for key, body in named.items()}
    if "Hand" not in left:
        left["Hand"] = left["ForeArm"] + np.array([0.0, -0.25, 0.1])
    foot = left["Foot"]
    joints = {
        f"Left{key}": value for key, value in left.items()
    }
    joints["LeftToeEnd"] = np.array([foot[0], low[1], foot[2] + 0.14])
    for key in list(joints):
        right = joints[key].copy()
        right[0] = -right[0]
        joints[key.replace("Left", "Right")] = right
    spine = body_position(spec["spine"])
    hips_y = left["Leg"][1]
    joints["Hips"] = np.array([0.0, hips_y, left["Leg"][2]])
    joints["Spine2"] = np.array([0.0, max(spine[1], hips_y + 0.08), left["Leg"][2]])
    # The neck sits just above the shoulders, and the head ends at the top of the model.
    neck_y = body_position(spec["neck"])[1] if "neck" in spec else left["Arm"][1] + 0.1
    joints["Neck1"] = np.array([0.0, neck_y, left["Arm"][2]])
    joints["HeadEnd"] = np.array([0.0, high[1], left["Arm"][2]])

    links = [(re.compile(pattern), kind) for pattern, kind in spec["links"].items()]
    pieces_by_key = {}
    index_of = {part[0]: i for i, part in enumerate(PARTS)}
    for body, vertices, faces, color in groups:
        kind = next((k for pattern, k in links if pattern.search(body)), None)
        if kind is None:
            raise RuntimeError(f"{name}: no part for link {body}")
        if kind == "Torso":
            # The head is cast into the torso link on these robots.
            centers = vertices[faces].mean(axis=1)
            owners = np.where(centers[:, 1] > joints["Neck1"][1], index_of["Head"], index_of["Torso"])
        else:
            is_left = re.search(r"^left_|^l_|_Left$|^AL", body)
            is_right = re.search(r"^right_|^r_|_Right$|^AR", body)
            part = ("Left" if is_left else "Right" if is_right else "") + kind
            owners = np.full(len(faces), index_of[part])
        for index in np.unique(owners):
            key = (int(index), color)
            pieces_by_key.setdefault(key, []).append((vertices, faces[owners == index]))

    pieces = []
    budget = 42000
    total = sum(len(f) for chunks in pieces_by_key.values() for _, f in chunks)
    for (index, color), chunks in pieces_by_key.items():
        merged = trimesh.util.concatenate([trimesh.Trimesh(v, f, process=False) for v, f in chunks])
        merged.merge_vertices()
        target = max(int(len(merged.faces) * budget / total), 60)
        if target < len(merged.faces):
            merged = merged.simplify_quadric_decimation(face_count=target)
        material = trimesh.visual.material.PBRMaterial(baseColorFactor=color, metallicFactor=0.25, roughnessFactor=0.42, doubleSided=True)
        pieces.append((index, np.asarray(merged.vertices), np.asarray(merged.faces), trimesh.visual.TextureVisuals(material=material)))
    write_parts(pieces, joints, out_dir / f"{name}.glb", name)
    return joints


def build_crown(source, out, face_budget=45000):
    """A crown, normalized to 1 unit wide with its base at y = 0 and the opening along y. Simplified with its textures kept."""
    import fast_simplification

    scene = trimesh.load(source)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        meshes.append((trimesh.transform_points(mesh.vertices, transform), np.asarray(mesh.faces), mesh.visual))
    every = np.concatenate([v for v, _, _ in meshes])
    low, high = every.min(0), every.max(0)
    center = (low + high) / 2
    width = max(high[0] - low[0], high[2] - low[2])
    total = sum(len(f) for _, f, _ in meshes)
    result = trimesh.Scene()
    for i, (vertices, faces, visual) in enumerate(meshes):
        ratio = 1.0 - min(face_budget / total, 1.0)
        points, triangles, collapses = fast_simplification.simplify(vertices.astype(np.float32), faces.astype(np.int32), target_reduction=ratio, agg=4, return_collapses=True)
        points, triangles, mapping = fast_simplification.replay_simplification(vertices.astype(np.float32), faces.astype(np.int32), collapses)
        # Each kept vertex takes the texture position of the first original vertex that collapsed into it.
        uv = np.zeros((len(points), 2))
        seen = np.zeros(len(points), dtype=bool)
        for old, new in enumerate(mapping):
            if 0 <= new < len(points) and not seen[new]:
                uv[new] = visual.uv[old]
                seen[new] = True
        material = visual.material
        shrink_texture(material, 1024)
        normalized = (points - np.array([center[0], low[1], center[2]])) / width
        mesh = trimesh.Trimesh(normalized, triangles, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        result.add_geometry(mesh, node_name=f"crown_{i}", geom_name=f"crown_{i}")
    out.write_bytes(result.export(file_type="glb"))
    print(f"crown: {out.stat().st_size / 1e6:.2f} MB, {sum(len(g.faces) for g in result.geometry.values())} faces, height {(high[1] - low[1]) / width:.2f} of its width")


def main():
    models = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("public/robots")
    wanted = sys.argv[3:] or [*MESHES, *MJCF, "crown"]
    rest = {}
    for name in wanted:
        if name == "crown":
            build_crown(models / "crown.glb", out_dir / "crown.glb")
            continue
        joints = build_mesh(name, MESHES[name], models, out_dir) if name in MESHES else build_mjcf(name, MJCF[name], models, out_dir)
        rest[name] = {k: [round(float(c), 4) for c in v] for k, v in joints.items()}
    (models / "rest_joints.json").write_text(json.dumps(rest, indent=1))


if __name__ == "__main__":
    main()
