import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { Recording, Vec3 } from './game/playback.ts';

// Same table as scripts/build_robots.py. A part's y runs 0 to 1 along its bone, x follows the side hint, sizes are meters.
type Hint = 'pelvis' | 'shoulders' | 'forward' | 'shin';
const SIDES = ['Left', 'Right'];
const PARTS: [string, string, string, Hint][] = [
  ['Pelvis', 'Hips', 'Spine2', 'pelvis'],
  ['Torso', 'Spine2', 'Neck1', 'shoulders'],
  ['Head', 'Neck1', 'HeadEnd', 'shoulders'],
  ...SIDES.flatMap((side): [string, string, string, Hint][] => [
    [`${side}UpperArm`, `${side}Arm`, `${side}ForeArm`, 'forward'],
    [`${side}LowerArm`, `${side}ForeArm`, `${side}Hand`, 'forward'],
    [`${side}Thigh`, `${side}Leg`, `${side}Shin`, 'pelvis'],
    [`${side}Calf`, `${side}Shin`, `${side}Foot`, 'pelvis'],
    [`${side}Sole`, `${side}Foot`, `${side}ToeEnd`, 'shin'],
  ]),
];
const ROUND_JOINTS = ['LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg', 'LeftShin', 'RightShin'];

export interface RobotSkin {
  name: string;
  color: string;
  model: string;
  joints?: { radius: number; color: string };
}

function material(color: string, metalness = 0.1, roughness = 0.5) {
  return new THREE.MeshStandardMaterial({ color, metalness, roughness });
}

export const ROSTER: RobotSkin[] = [
  { name: 'Optimus', color: '#e8eaee', model: 'optimus' },
  { name: 'Figure', color: '#3a3d46', model: 'figure', joints: { radius: 0.062, color: '#aeb6c2' } },
  { name: 'G1', color: '#6b7a8c', model: 'g1' },
  { name: 'H1', color: '#2f89d9', model: 'h1' },
  { name: 'Apollo', color: '#f06b3a', model: 'apollo' },
  { name: 'Atlas', color: '#a56ddb', model: 'atlas', joints: { radius: 0.055, color: '#3e4149' } },
  { name: 'T1', color: '#d8a83a', model: 't1' },
  { name: 'Petman', color: '#c0392b', model: 'petman' },
];

const loader = new GLTFLoader();

export async function loadSkin(skin: RobotSkin, base = 'robots/') {
  const gltf = await loader.loadAsync(`${base}${skin.model}.glb`);
  const parts = new Map<string, THREE.Object3D[]>();
  gltf.scene.traverse(node => {
    if (!(node instanceof THREE.Mesh)) return;
    const part = node.name.split('__')[0];
    if (!parts.has(part)) parts.set(part, []);
    parts.get(part)!.push(node);
  });
  return parts;
}

const start = new THREE.Vector3();
const end = new THREE.Vector3();
const axisX = new THREE.Vector3();
const axisY = new THREE.Vector3();
const axisZ = new THREE.Vector3();
const scratch: Vec3 = [0, 0, 0];

export class Robot {
  readonly group = new THREE.Group();
  readonly head = new THREE.Vector3();
  readonly chest = new THREE.Vector3();
  readonly feet = new THREE.Vector3();
  private readonly points: THREE.Vector3[];
  private readonly index: Map<string, number>;
  private readonly parts: { pivot: THREE.Group; a: number; b: number; hint: Hint; last: THREE.Vector3 }[] = [];
  private readonly rounds: { mesh: THREE.Mesh; joint: number }[] = [];

  constructor(scene: THREE.Object3D, private readonly recording: Recording, private readonly runner: number, skin: RobotSkin, meshes: Map<string, THREE.Object3D[]>) {
    const names = recording.meta.joints;
    this.index = new Map(names.map((name, i) => [name, i]));
    this.points = names.map(() => new THREE.Vector3());
    for (const [part, from, to, hint] of PARTS) {
      const pivot = new THREE.Group();
      pivot.matrixAutoUpdate = false;
      for (const piece of meshes.get(part) ?? []) {
        piece.traverse(child => {
          child.frustumCulled = false;
          if (child instanceof THREE.Mesh) child.castShadow = true;
        });
        pivot.add(piece);
      }
      this.group.add(pivot);
      this.parts.push({ pivot, a: this.index.get(from)!, b: this.index.get(to)!, hint, last: new THREE.Vector3(1, 0, 0) });
    }
    if (skin.joints) {
      const ball = new THREE.SphereGeometry(skin.joints.radius, 16, 12);
      const surface = material(skin.joints.color, 0.7, 0.3);
      for (const name of ROUND_JOINTS) {
        const round = new THREE.Mesh(ball, surface);
        round.castShadow = true;
        round.frustumCulled = false;
        this.group.add(round);
        this.rounds.push({ mesh: round, joint: this.index.get(name)! });
      }
    }
    scene.add(this.group);
  }

  private at(name: string) { return this.points[this.index.get(name)!]; }

  draw(time: number) {
    const { recording, runner, points } = this;
    if (!recording.joint(time, runner, 0, scratch)) { this.group.visible = false; return false; }
    this.group.visible = true;
    for (let j = 0; j < points.length; j++) {
      recording.joint(time, runner, j, scratch);
      // The round records X and Y across the floor and Z up. Three.js is Y up.
      points[j].set(scratch[1], scratch[2], scratch[0]);
    }
    const pelvis = end.copy(this.at('RightLeg')).sub(this.at('LeftLeg')).clone();
    const shoulders = end.copy(this.at('RightArm')).sub(this.at('LeftArm')).clone();
    const upward = end.copy(this.at('Neck1')).sub(this.at('Hips')).clone();
    const forward = new THREE.Vector3().crossVectors(upward, shoulders);
    for (const part of this.parts) {
      start.copy(points[part.a]);
      axisY.copy(points[part.b]).sub(start);
      const length = axisY.length();
      axisY.divideScalar(length || 1);
      if (part.hint === 'pelvis') axisX.copy(pelvis);
      else if (part.hint === 'shoulders') axisX.copy(shoulders);
      else if (part.hint === 'forward') axisX.copy(forward);
      else axisX.copy(points[part.a - 1]).sub(start);
      axisX.addScaledVector(axisY, -axisX.dot(axisY));
      // A bone can line up with its hint for a frame. Keep the last good direction.
      if (axisX.lengthSq() < 1e-5) axisX.copy(part.last);
      else axisX.normalize();
      part.last.copy(axisX);
      axisZ.crossVectors(axisX, axisY);
      part.pivot.matrix.set(
        axisX.x, axisY.x * length, axisZ.x, start.x,
        axisX.y, axisY.y * length, axisZ.y, start.y,
        axisX.z, axisY.z * length, axisZ.z, start.z,
        0, 0, 0, 1,
      );
      part.pivot.matrixWorldNeedsUpdate = true;
    }
    for (const round of this.rounds) round.mesh.position.copy(points[round.joint]);
    this.head.copy(this.at('HeadEnd'));
    this.chest.copy(this.at('Chest'));
    this.feet.copy(this.at('LeftFoot')).add(this.at('RightFoot')).multiplyScalar(0.5);
    return true;
  }

  /** Shift weight slowly around the feet. Upstream's idle clips cannot be recorded, so waiting robots do this instead of freezing. */
  wiggle(time: number, amount: number) {
    const lean = Math.sin(time * 1.3 + this.runner * 1.9) * 0.025 * amount;
    const nod = Math.sin(time * 0.9 + this.runner * 0.7) * 0.018 * amount;
    this.group.rotation.set(nod, 0, lean);
    const tilted = this.feet.clone().applyEuler(this.group.rotation);
    this.group.position.copy(this.feet).sub(tilted);
  }
}
