import * as THREE from 'three';
import type { HexArena } from './game/hex.ts';

// Colors sampled from gameplay footage. See docs/HEXAGONS.md.
const YELLOW = ['#f8b823', '#cc8f25'];
const BLUE = ['#147ac5', '#0e5f9e'];
const PURPLE = ['#94429c', '#743280'];
// The footage shows blue under the yellow top and a blue bottom floor.
export const FLOOR_COLORS = [YELLOW, BLUE, PURPLE, YELLOW, PURPLE, BLUE];
const RAIL_COLORS = ['#ec6416', '#147ac5', '#94429c', '#ec6416', '#94429c', '#147ac5'];

function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** A pink pool with slow lighter swirls. */
function makeSlime(y: number) {
  const material = new THREE.ShaderMaterial({
    uniforms: { time: { value: 0 } },
    vertexShader: 'varying vec2 vPos; void main() { vPos = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `
      uniform float time; varying vec2 vPos;
      float wave(vec2 p) { return sin(p.x + sin(p.y * 1.3 + time * 0.2) * 1.4) * cos(p.y * 0.8 + sin(p.x * 0.7 - time * 0.15) * 1.7); }
      void main() {
        vec2 p = vPos * 0.09;
        float w = wave(p + wave(p * 0.5) * 0.9);
        float line = smoothstep(0.035, 0.0, abs(w - 0.15)) + smoothstep(0.03, 0.0, abs(w + 0.45)) * 0.6;
        vec3 color = mix(vec3(0.992, 0.56, 0.74), vec3(1.0, 0.66, 0.81), smoothstep(-0.7, 0.8, w));
        gl_FragColor = vec4(mix(color, vec3(1.0, 0.82, 0.9), line * 0.75), 1.0);
      }`,
  });
  const slime = new THREE.Mesh(new THREE.CircleGeometry(420, 64).rotateX(-Math.PI / 2), material);
  // CircleGeometry was rotated, so the shader's xy comes from the world floor plane.
  material.vertexShader = 'varying vec2 vPos; void main() { vPos = position.xz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }';
  slime.position.y = y;
  return { slime, material };
}

function blobShape(rng: () => number, radius: number) {
  const shape = new THREE.Shape();
  const points = 14;
  const bumps = Array.from({ length: points }, () => radius * (0.78 + rng() * 0.3));
  const at = (i: number) => {
    const angle = (i / points) * Math.PI * 2;
    return new THREE.Vector2(Math.cos(angle) * bumps[i % points], Math.sin(angle) * bumps[i % points]);
  };
  shape.setFromPoints(new THREE.SplineCurve(Array.from({ length: points + 1 }, (_, i) => at(i))).getPoints(70));
  return shape;
}

/** A flat grass island with the white line the slime leaves around it. */
function makeIsland(rng: () => number, radius: number, height: number, y: number, x: number, z: number) {
  const group = new THREE.Group();
  const shape = blobShape(rng, radius);
  const grass = new THREE.ExtrudeGeometry(shape, { depth: height, bevelEnabled: true, bevelSize: Math.min(radius * 0.06, 1.2), bevelThickness: Math.min(radius * 0.05, 0.8), bevelSegments: 3, curveSegments: 6 }).rotateX(-Math.PI / 2);
  const land = new THREE.Mesh(grass, new THREE.MeshStandardMaterial({ color: '#b4de78', roughness: 0.9 }));
  land.receiveShadow = true;
  const line = new THREE.Mesh(new THREE.ShapeGeometry(blobShape(rng, radius * 1.07)).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: '#fff0f6' }));
  line.position.y = 0.02;
  group.add(line, land);
  group.position.set(x, y, z);
  return group;
}

function makeTree(rng: () => number, x: number, y: number, z: number) {
  const tree = new THREE.Group();
  const colors = ['#f2698f', '#ef5f7f', '#8fce62', '#73bb58', '#6f8fa4', '#7bc26c'];
  const color = colors[Math.floor(rng() * colors.length)];
  const leaves = new THREE.MeshStandardMaterial({ color, roughness: 0.75 });
  const scale = 1.6 + rng() * 2.2;
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.34, 2.4, 8), new THREE.MeshStandardMaterial({ color: '#8d6a55', roughness: 0.9 }));
  trunk.position.y = 1.2;
  tree.add(trunk);
  for (const [dx, dy, dz, size] of [[0, 3.4, 0, 1.5], [0.9, 2.8, 0.2, 1], [-0.8, 2.9, -0.3, 1.05], [0.1, 4.3, 0.2, 1]]) {
    const ball = new THREE.Mesh(new THREE.SphereGeometry(size, 14, 10), leaves);
    ball.position.set(dx, dy, dz);
    tree.add(ball);
  }
  tree.scale.setScalar(scale);
  tree.position.set(x, y, z);
  return tree;
}

/** A soft cone in stripes with a snow cap, in vertex colors. */
function makeMountain(rng: () => number, radius: number, height: number, x: number, y: number, z: number) {
  const geometry = new THREE.ConeGeometry(radius, height, 28, 12);
  const position = geometry.getAttribute('position');
  const colors = new Float32Array(position.count * 3);
  const color = new THREE.Color();
  const bands = ['#79c9a1', '#8fd6ad', '#ee87a6', '#f7a8c0', '#ffd0dc', '#fff6f4', '#ffffff'];
  for (let i = 0; i < position.count; i++) {
    const level = position.getY(i) / height + 0.5;
    color.set(bands[Math.min(Math.floor(level * bands.length), bands.length - 1)]);
    colors.set([color.r, color.g, color.b], i * 3);
    // Round the point off so it reads as a soft candy hill.
    const squash = 1 - Math.pow(Math.max(level - 0.72, 0) / 0.28, 2) * 0.2;
    position.setY(i, position.getY(i) * squash);
    const wobble = 1 + Math.sin(i * 1.7) * 0.03 * (1 - level);
    position.setX(i, position.getX(i) * wobble);
    position.setZ(i, position.getZ(i) * wobble);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeVertexNormals();
  const mountain = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, flatShading: false }));
  mountain.position.set(x, y + height * 0.42, z);
  mountain.rotation.y = rng() * 6;
  return mountain;
}

function makePanelTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  const draw = canvas.getContext('2d')!;
  const fill = draw.createLinearGradient(0, 0, 1024, 512);
  fill.addColorStop(0, '#16b3e0');
  fill.addColorStop(0.5, '#ff5fa2');
  fill.addColorStop(1, '#ffc22e');
  draw.fillStyle = fill;
  draw.fillRect(0, 0, 1024, 512);
  draw.fillStyle = '#ffffff33';
  for (let i = 0; i < 9; i++) {
    draw.beginPath();
    draw.moveTo(512, 256);
    draw.arc(512, 256, 800, (i / 9) * Math.PI * 2, ((i + 0.5) / 9) * Math.PI * 2);
    draw.fill();
  }
  draw.textAlign = 'center';
  draw.lineJoin = 'round';
  draw.font = '150px "Titan One", sans-serif';
  draw.lineWidth = 26;
  draw.strokeStyle = '#3a1457';
  draw.fillStyle = '#ffffff';
  for (const [text, y] of [['HEX-A', 215], ['BOTS', 390]] as const) {
    draw.strokeText(text, 512, y);
    draw.fillText(text, 512, y);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** The propeller on a post or a panel. The caller spins the returned hub. */
function makePropeller(color: string) {
  const hub = new THREE.Group();
  const blade = new THREE.MeshStandardMaterial({ color, roughness: 0.4 });
  for (const angle of [0, Math.PI]) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.04, 0.16), blade);
    mesh.position.set(Math.cos(angle) * 0.85, 0.22, Math.sin(angle) * 0.85);
    hub.add(mesh);
  }
  hub.add(new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.3, 10).translate(0, 0.1, 0), new THREE.MeshStandardMaterial({ color: '#ff8a1e', roughness: 0.4 })));
  return hub;
}

/** One octagon of padded rails for a floor, with posts down to the next one. */
function makeCage(arena: HexArena, layer: number, propellers: THREE.Object3D[]) {
  const group = new THREE.Group();
  const y = -layer * arena.layerHeight;
  const radius = arena.radii[layer] + 2.4;
  const color = RAIL_COLORS[layer % RAIL_COLORS.length];
  const surface = new THREE.MeshStandardMaterial({ color, roughness: 0.42 });
  const trim = new THREE.MeshStandardMaterial({ color: new THREE.Color(color).multiplyScalar(0.72), roughness: 0.5 });
  const corners = Array.from({ length: 8 }, (_, i) => {
    const angle = (i + 0.5) * Math.PI / 4;
    return new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius);
  });
  corners.forEach((corner, i) => {
    const next = corners[(i + 1) % 8];
    const middle = corner.clone().add(next).multiplyScalar(0.5);
    const length = corner.distanceTo(next);
    const angle = -Math.atan2(next.z - corner.z, next.x - corner.x);
    [0.15, 0.55, 0.95].forEach((lift, bar) => {
      const rail = new THREE.Mesh(new THREE.CapsuleGeometry(bar === 2 ? 0.26 : 0.21, length, 4, 10).rotateZ(Math.PI / 2), bar === 2 ? surface : trim);
      rail.position.copy(middle).setY(y + lift);
      rail.rotation.y = angle;
      rail.castShadow = true;
      group.add(rail);
    });
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, arena.layerHeight + 1.2, 12), surface);
    post.position.copy(corner).setY(y + 0.8 - (arena.layerHeight + 1.2) / 2);
    group.add(post);
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.5, 14, 10).scale(1, 0.8, 1), trim);
    cap.position.copy(corner).setY(y + 1.25);
    group.add(cap);
    if (layer === 0 && i % 2 === 0) {
      const hub = makePropeller('#ff4fa3');
      hub.position.copy(corner).setY(y + 1.6);
      group.add(hub);
      propellers.push(hub);
    }
  });
  return group;
}

export interface World {
  group: THREE.Group;
  update(time: number): void;
}

export function buildWorld(arena: HexArena): World {
  const group = new THREE.Group();
  const rng = random(11);
  const slimeY = -(arena.layers - 1) * arena.layerHeight - arena.slimeDepth;
  const { slime, material: slimeMaterial } = makeSlime(slimeY);
  group.add(slime);
  const propellers: THREE.Object3D[] = [];
  for (let layer = 0; layer < arena.layers; layer++) group.add(makeCage(arena, layer, propellers));

  // Lily-pad islands near the arena, and big meadows farther out.
  const outer = arena.radii[arena.layers - 1];
  for (let i = 0; i < 9; i++) {
    const angle = (i / 9) * Math.PI * 2 + rng();
    const distance = outer + 8 + rng() * 26;
    group.add(makeIsland(rng, 3 + rng() * 5, 0.5, slimeY, Math.cos(angle) * distance, Math.sin(angle) * distance));
  }
  const meadowY = slimeY;
  for (let i = 0; i < 12; i++) {
    const angle = (i / 12) * Math.PI * 2 + rng() * 0.3;
    const distance = outer + 78 + rng() * 24;
    const x = Math.cos(angle) * distance;
    const z = Math.sin(angle) * distance;
    group.add(makeIsland(rng, 40 + rng() * 22, 3 + rng() * 5, meadowY, x, z));
    for (let t = 0; t < 8; t++) {
      const a = rng() * Math.PI * 2;
      const r = 8 + rng() * 26;
      group.add(makeTree(rng, x + Math.cos(a) * r, meadowY + 6, z + Math.sin(a) * r));
    }
  }
  const ground = new THREE.Mesh(new THREE.RingGeometry(outer + 118, 460, 64).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ color: '#b4de78', roughness: 0.95 }));
  ground.position.y = slimeY + 1;
  group.add(ground);
  for (let i = 0; i < 16; i++) {
    const angle = (i / 16) * Math.PI * 2 + rng() * 0.2;
    const distance = outer + 165 + rng() * 60;
    group.add(makeMountain(rng, 48 + rng() * 36, 62 + rng() * 44, Math.cos(angle) * distance, slimeY, Math.sin(angle) * distance));
  }

  // A floating sign held up by propellers, in the arena's sky.
  const panelMaterial = new THREE.MeshStandardMaterial({ map: makePanelTexture(), roughness: 0.5 });
  const frame = new THREE.MeshStandardMaterial({ color: '#4a2a91', roughness: 0.4 });
  const panel = new THREE.Group();
  panel.add(new THREE.Mesh(new THREE.BoxGeometry(12, 6, 6), [frame, frame, frame, frame, panelMaterial, panelMaterial]));
  for (const [x, z] of [[-6, -3], [6, -3], [-6, 3], [6, 3]]) {
    const hub = makePropeller('#7b4ff0');
    hub.position.set(x, 3.05, z);
    panel.add(hub);
    propellers.push(hub);
  }
  panel.position.set(outer + 12, -4, -outer + 4);
  group.add(panel);

  return {
    group,
    update(time) {
      slimeMaterial.uniforms.time.value = time;
      propellers.forEach((hub, i) => { hub.rotation.y = time * (9 + (i % 3)); });
      panel.rotation.y = -0.7 + Math.sin(time * 0.25) * 0.25;
      panel.position.y = -4 + Math.sin(time * 0.6) * 0.5;
    },
  };
}
