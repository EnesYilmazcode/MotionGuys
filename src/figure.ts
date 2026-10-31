import * as THREE from 'three';
import type { Recording, Vec3 } from './game/playback.ts';

// An original mannequin: chunky capsule limbs, mitten hands, a round helmet with a visor.
const bones: [string, string, number][] = [
  ['Hips', 'Chest', 0.2], ['Chest', 'Neck1', 0.17],
  ['Chest', 'LeftArm', 0.08], ['Chest', 'RightArm', 0.08], ['Hips', 'LeftLeg', 0.1], ['Hips', 'RightLeg', 0.1],
  ['LeftLeg', 'LeftShin', 0.1], ['LeftShin', 'LeftFoot', 0.085], ['LeftFoot', 'LeftToeEnd', 0.075],
  ['RightLeg', 'RightShin', 0.1], ['RightShin', 'RightFoot', 0.085], ['RightFoot', 'RightToeEnd', 0.075],
  ['LeftArm', 'LeftForeArm', 0.07], ['LeftForeArm', 'LeftHand', 0.062],
  ['RightArm', 'RightForeArm', 0.07], ['RightForeArm', 'RightHand', 0.062],
];
const unitCapsule = new THREE.CapsuleGeometry(1, 1, 6, 14);
const up = new THREE.Vector3(0, 1, 0);
const a: Vec3 = [0, 0, 0];
const b: Vec3 = [0, 0, 0];
const start = new THREE.Vector3();
const end = new THREE.Vector3();

function makeGear(kind: string, material: THREE.Material) {
  const gear = new THREE.Group();
  const add = (geometry: THREE.BufferGeometry, y: number) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.y = y;
    mesh.castShadow = true;
    gear.add(mesh);
  };
  if (kind === 'cone') add(new THREE.ConeGeometry(0.13, 0.3, 16), 0.26);
  if (kind === 'bun') add(new THREE.SphereGeometry(0.075, 12, 10), 0.2);
  if (kind === 'brim') { add(new THREE.CylinderGeometry(0.27, 0.27, 0.025, 20), 0.1); add(new THREE.CylinderGeometry(0.13, 0.15, 0.13, 16), 0.17); }
  if (kind === 'cap') add(new THREE.SphereGeometry(0.175, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), 0.03);
  return gear;
}

export interface FigureStyle { color: string; bulk?: number; gear?: string; }

export class Figure {
  readonly group = new THREE.Group();
  private readonly limbs: { a: number; b: number; radius: number; mesh: THREE.Mesh }[];
  private readonly helmet: THREE.Mesh;
  private readonly visor: THREE.Mesh;
  private readonly gear: THREE.Group;
  private readonly head: number;
  private readonly headEnd: number;
  private readonly chest: number;

  constructor(scene: THREE.Object3D, private readonly recording: Recording, private readonly runner: number, style: FigureStyle) {
    const index = new Map(recording.meta.joints.map((name, i) => [name, i]));
    const bulk = style.bulk ?? 1;
    const suit = new THREE.MeshStandardMaterial({ color: style.color, roughness: 0.45 });
    const trim = new THREE.MeshStandardMaterial({ color: '#25354a', roughness: 0.5 });
    this.limbs = bones.map(([from, to, radius]) => {
      const mesh = new THREE.Mesh(unitCapsule, from.endsWith('Foot') ? trim : suit);
      mesh.castShadow = true;
      this.group.add(mesh);
      return { a: index.get(from)!, b: index.get(to)!, radius: radius * bulk, mesh };
    });
    this.helmet = new THREE.Mesh(new THREE.SphereGeometry(0.17, 20, 14), suit);
    this.visor = new THREE.Mesh(new THREE.SphereGeometry(0.135, 16, 10), new THREE.MeshStandardMaterial({ color: '#1c2738', roughness: 0.15 }));
    this.helmet.castShadow = true;
    this.gear = makeGear(style.gear ?? '', trim);
    this.group.add(this.helmet, this.visor, this.gear);
    this.head = index.get('Head')!;
    this.headEnd = index.get('HeadEnd')!;
    this.chest = index.get('Chest')!;
    scene.add(this.group);
  }

  draw(time: number) {
    const { recording, runner } = this;
    if (!recording.joint(time, runner, 0, a)) { this.group.visible = false; return; }
    this.group.visible = true;
    for (const limb of this.limbs) {
      recording.joint(time, runner, limb.a, a);
      recording.joint(time, runner, limb.b, b);
      start.set(a[1], a[2], a[0]);
      end.set(b[1], b[2], b[0]);
      limb.mesh.position.lerpVectors(start, end, 0.5);
      end.sub(start);
      const length = end.length();
      // The unit capsule is 3 long, so this makes it reach both joints plus its radius.
      limb.mesh.scale.set(limb.radius, (length + limb.radius * 2) / 3, limb.radius);
      limb.mesh.quaternion.setFromUnitVectors(up, end.normalize());
    }
    recording.joint(time, runner, this.head, a);
    recording.joint(time, runner, this.headEnd, b);
    this.helmet.position.set((a[1] + b[1]) / 2, (a[2] + b[2]) / 2, (a[0] + b[0]) / 2);
    this.gear.position.copy(this.helmet.position);
    // The headgear leans with the head, and the visor sits on the side the head leans toward.
    end.set(b[1] - a[1], b[2] - a[2], b[0] - a[0]).normalize();
    this.gear.quaternion.setFromUnitVectors(up, end);
    recording.joint(time, runner, this.chest, b);
    start.set(a[1] - b[1], 0, a[0] - b[0]);
    if (start.lengthSq() < 1e-4) start.set(0, 0, 1);
    this.visor.position.copy(this.helmet.position).addScaledVector(start.normalize(), 0.06);
  }
}
