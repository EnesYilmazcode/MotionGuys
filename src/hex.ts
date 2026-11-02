import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { actionShots, brainStateAt, describeHex, GAIT_NAMES, hexCenter, hexKey, tileTimes, tileWhiteness, type HexMeta } from './game/hex.ts';
import { Recording, type RaceMeta, type Vec3 } from './game/playback.ts';
import { ranking, roundStateAt, shotAt, spectatorShots, type SpleefMeta } from './game/spleef.ts';
import { loadSkin, Robot, ROSTER } from './robots.ts';
import { buildWorld, FLOOR_COLORS } from './world.ts';
import './hex.css';

const params = new URLSearchParams(location.search);
const seed = params.get('seed') ?? '6';
const base = `${params.get('dir') ?? 'hex'}/seed_${seed}`;
const [meta, buffer] = await Promise.all([
  fetch(`${base}.json`).then(r => r.json() as Promise<HexMeta>),
  fetch(`${base}.bin`).then(r => r.arrayBuffer()),
  document.fonts.load('40px "Titan One"'),
]);
const names = ROSTER.map(skin => skin.name);
meta.runners.forEach((runner, i) => { runner.name = names[i]; runner.color = ROSTER[i].color; });
const spleefMeta = meta as unknown as SpleefMeta;
const recording = new Recording(meta as unknown as RaceMeta, buffer);
const { arena: floors } = meta;

const stage = document.querySelector<HTMLElement>('#stage')!;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#bfeef6');
scene.fog = new THREE.Fog('#d9f3f4', 190, 430);
const camera = new THREE.PerspectiveCamera(50, 1, 0.15, 900);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: params.has('capture') });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
stage.prepend(renderer.domElement);
const environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environment = environment;
scene.environmentIntensity = 0.35;
const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;

scene.add(new THREE.HemisphereLight('#ffffff', '#ffc0d8', 1.35));
const sun = new THREE.DirectionalLight('#fff4e0', 1.75);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0006;
Object.assign(sun.shadow.camera, { left: -26, right: 26, top: 26, bottom: -26, near: 1, far: 120 });
scene.add(sun, sun.target);

const world = buildWorld(floors);
scene.add(world.group);

// A soft-edged hexagon prism, darker down its sides, the way the footage shows it.
function hexGeometry(radius: number, thickness: number) {
  const shape = new THREE.Shape();
  for (let i = 0; i <= 6; i++) {
    const angle = (i * Math.PI) / 3;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    if (i) shape.lineTo(x, y);
    else shape.moveTo(x, y);
  }
  const bevel = 0.07;
  const geometry = new THREE.ExtrudeGeometry(shape, { depth: thickness - bevel * 2, bevelEnabled: true, bevelSize: bevel, bevelOffset: -bevel, bevelThickness: bevel, bevelSegments: 2 }).rotateX(-Math.PI / 2);
  geometry.translate(0, -thickness + bevel, 0);
  geometry.computeVertexNormals();
  const normal = geometry.getAttribute('normal');
  const shades = new Float32Array(normal.count * 3);
  for (let i = 0; i < normal.count; i++) {
    const shade = normal.getY(i) > 0.5 ? 1 : 0.68;
    shades.set([shade, shade, shade], i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(shades, 3));
  return geometry;
}

const tileMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.48, metalness: 0, envMapIntensity: 0.4 });
const geometry = hexGeometry(floors.hexRadius * 0.975, 0.42);
const white = new THREE.Color('#fff8e2');
const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
const matrix = new THREE.Matrix4();
const mixed = new THREE.Color();
const times = tileTimes(meta);
const shade = (q: number, r: number, layer: number) => Math.abs(Math.sin(q * 12.9898 + r * 78.233 + layer * 37.7) * 43758.5453) % 1;
const tileSets = floors.tiles.map((tiles, layer) => {
  const mesh = new THREE.InstancedMesh(geometry, tileMaterial, tiles.length);
  mesh.receiveShadow = mesh.castShadow = true;
  mesh.frustumCulled = false;
  const palette = FLOOR_COLORS[layer % FLOOR_COLORS.length];
  const base = tiles.map(([q, r]) => new THREE.Color(palette[shade(q, r, layer) % 1 < 0.17 ? 1 : 0]));
  const rest = tiles.map(([q, r]) => {
    const [x, y] = hexCenter(q, r, floors.hexRadius);
    return new THREE.Matrix4().makeTranslation(y, -layer * floors.layerHeight, x);
  });
  const entries = tiles.map(([q, r]) => times.get(hexKey(layer, q, r)));
  tiles.forEach((_, i) => { mesh.setMatrixAt(i, rest[i]); mesh.setColorAt(i, base[i]); });
  scene.add(mesh);
  return { mesh, base, rest, entries, drawn: new Float32Array(tiles.length).fill(0) };
});

const pedestalMaterial = new THREE.MeshStandardMaterial({ color: '#f0ac2c', roughness: 0.5 });
const pedestalLeft = new Map<number, number>();
for (const event of meta.events) if (event.type === 'pedestal_gone') pedestalLeft.set(event.runner as number, event.t);
const pedestals = floors.pedestals.map(({ x, y }) => {
  const mesh = new THREE.Mesh(hexGeometry(floors.hexRadius, 0.5), pedestalMaterial.clone());
  mesh.position.set(y, floors.pedestalHeight, x);
  mesh.castShadow = true;
  scene.add(mesh);
  return mesh;
});

function drawTiles() {
  for (const set of tileSets) {
    let changedMatrix = false;
    let changedColor = false;
    for (let i = 0; i < set.base.length; i++) {
      const whiteness = tileWhiteness(time, set.entries[i], floors.touchSeconds);
      if (whiteness === set.drawn[i]) continue;
      if ((whiteness === -1) !== (set.drawn[i] === -1)) {
        set.mesh.setMatrixAt(i, whiteness === -1 ? hidden : set.rest[i]);
        changedMatrix = true;
      }
      if (whiteness !== -1) {
        set.mesh.setColorAt(i, mixed.copy(set.base[i]).lerp(white, whiteness));
        changedColor = true;
      }
      set.drawn[i] = whiteness;
    }
    if (changedMatrix) set.mesh.instanceMatrix.needsUpdate = true;
    if (changedColor) set.mesh.instanceColor!.needsUpdate = true;
  }
  pedestals.forEach((mesh, runner) => {
    const left = pedestalLeft.get(runner) ?? Infinity;
    const age = time - left;
    mesh.visible = age < floors.touchSeconds;
    (mesh.material as THREE.MeshStandardMaterial).color.set('#f0ac2c').lerp(white, Math.min(Math.max(age / floors.touchSeconds - 0.47, 0) / 0.4, 1));
  });
}

// Short white rings where feet touch, like the footage, and pink drops where a robot hits the slime.
const dotGeometry = new THREE.SphereGeometry(0.055, 8, 6);
const dots = new THREE.InstancedMesh(dotGeometry, new THREE.MeshBasicMaterial({ color: '#ffffff' }), 8 * 40);
dots.frustumCulled = false;
scene.add(dots);
const touches = meta.events.filter(event => event.type === 'tile_touch');
const drops = new THREE.InstancedMesh(new THREE.SphereGeometry(0.2, 8, 6), new THREE.MeshStandardMaterial({ color: '#ff8fbf', roughness: 0.3 }), 24 * 8);
drops.frustumCulled = false;
scene.add(drops);
const splashes = meta.events.filter(event => event.type === 'splash');
const slimeY = -(floors.layers - 1) * floors.layerHeight - floors.slimeDepth;
const rippleMaterial = new THREE.MeshBasicMaterial({ color: '#ffe3ef', transparent: true, side: THREE.DoubleSide, depthWrite: false });
const ripples = splashes.map(() => {
  const ripple = new THREE.Mesh(new THREE.RingGeometry(0.85, 1, 40).rotateX(-Math.PI / 2), rippleMaterial.clone());
  scene.add(ripple);
  return ripple;
});
const dot = new THREE.Object3D();

function drawEffects() {
  let used = 0;
  for (const touch of touches) {
    const age = time - touch.t;
    if (age < 0) break;
    if (age > 0.35 || used >= dots.count - 8) continue;
    const radius = 0.14 + age * 1.1;
    for (let k = 0; k < 8; k++) {
      const angle = (k / 8) * Math.PI * 2;
      dot.position.set((touch.y as number) + Math.cos(angle) * radius, -(touch.layer as number) * floors.layerHeight + 0.06 + age * 0.3, (touch.x as number) + Math.sin(angle) * radius);
      dot.scale.setScalar(1 - age / 0.4);
      dot.updateMatrix();
      dots.setMatrixAt(used++, dot.matrix);
    }
  }
  dots.count = used;
  dots.instanceMatrix.needsUpdate = true;
  let dropCount = 0;
  splashes.forEach((splash, s) => {
    const age = time - splash.t;
    const ripple = ripples[s];
    ripple.visible = age >= 0 && age < 1.6;
    if (ripple.visible) {
      ripple.position.set(splash.y as number, slimeY + 0.08, splash.x as number);
      ripple.scale.setScalar(1 + age * 4);
      (ripple.material as THREE.MeshBasicMaterial).opacity = 1 - age / 1.6;
    }
    if (age < 0 || age > 1) return;
    for (let k = 0; k < 24; k++) {
      const angle = (k * 2.4) + s;
      const speed = 1.6 + (k % 5) * 0.8;
      dot.position.set((splash.y as number) + Math.cos(angle) * speed * age, slimeY + (5 + (k % 7)) * age - 4.9 * age * age * 1.6, (splash.x as number) + Math.sin(angle) * speed * age);
      dot.scale.setScalar(1 - age * 0.6);
      dot.updateMatrix();
      drops.setMatrixAt(dropCount++, dot.matrix);
    }
  });
  drops.count = dropCount;
  drops.instanceMatrix.needsUpdate = true;
}

// Where bodies hit something: a ring of stars for a knockdown, a soft puff for a landing.
const starShape = new THREE.Shape();
for (let i = 0; i < 10; i++) {
  const angle = (i / 10) * Math.PI * 2;
  const reach = i % 2 ? 0.05 : 0.13;
  if (i) starShape.lineTo(Math.cos(angle) * reach, Math.sin(angle) * reach);
  else starShape.moveTo(Math.cos(angle) * reach, Math.sin(angle) * reach);
}
const knocks = meta.events.filter(event => event.type === 'knock');
const stars = new THREE.InstancedMesh(new THREE.ShapeGeometry(starShape), new THREE.MeshBasicMaterial({ color: '#ffe14d', side: THREE.DoubleSide }), 8 * 8);
stars.frustumCulled = false;
scene.add(stars);
// A bouncing body lands several times. One puff per bounce sequence.
const lastLand = new Map<number, number>();
const lands = meta.events.filter(event => {
  if (event.type !== 'land') return false;
  const fresh = event.t - (lastLand.get(event.runner as number) ?? -9) > 0.8;
  if (fresh) lastLand.set(event.runner as number, event.t);
  return fresh;
});
const puffs = new THREE.InstancedMesh(new THREE.SphereGeometry(0.1, 10, 8), new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 1, transparent: true, opacity: 0.55, depthWrite: false }), 6 * 16);
puffs.frustumCulled = false;
scene.add(puffs);

const skins = await Promise.all(ROSTER.map(skin => loadSkin(skin)));
const robots = ROSTER.map((skin, i) => new Robot(scene, recording, i, skin, skins[i]));

function makeTag(text: string, color: string) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const draw = canvas.getContext('2d')!;
  draw.font = '34px "Titan One", sans-serif';
  draw.textAlign = 'center';
  draw.lineJoin = 'round';
  draw.lineWidth = 8;
  draw.strokeStyle = color;
  draw.fillStyle = '#fff';
  draw.strokeText(text, 128, 44);
  draw.fillText(text, 128, 44);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false, transparent: true }));
  sprite.scale.set(1.5, 0.375, 1);
  sprite.renderOrder = 5;
  scene.add(sprite);
  return sprite;
}
const tags = ROSTER.map(skin => makeTag(skin.name, '#2a1a4f'));

// The crown and confetti for whoever is last.
const winEvent = meta.events.find(event => event.type === 'winner');
const crown = new THREE.Group();
{
  const gltf = await new GLTFLoader().loadAsync('robots/crown.glb');
  crown.add(gltf.scene);
  crown.scale.setScalar(0.3);
  crown.visible = false;
  crown.traverse(child => { child.frustumCulled = false; });
  scene.add(crown);
}
const confettiColors = ['#ff4fa3', '#ffd21f', '#22b5e8', '#7b4ff0', '#52d66b', '#ff7a1a'];
const confetti = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.16, 0.09), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), 240);
confetti.frustumCulled = false;
for (let i = 0; i < confetti.count; i++) confetti.setColorAt(i, new THREE.Color(confettiColors[i % confettiColors.length]));
scene.add(confetti);
const hipsNow: Vec3 = [0, 0, 0];

function drawImpacts() {
  let used = 0;
  for (const knock of knocks) {
    const age = time - knock.t;
    if (age < 0) break;
    if (age > 0.9 || used >= 7) continue;
    const robot = robots[knock.runner as number];
    for (let k = 0; k < 8; k++) {
      const angle = (k / 8) * Math.PI * 2 + age * 5;
      dot.position.set(robot.head.x + Math.cos(angle) * (0.35 + age * 0.4), robot.head.y + 0.15 + Math.sin(age * 4 + k) * 0.08, robot.head.z + Math.sin(angle) * (0.35 + age * 0.4));
      dot.lookAt(camera.position);
      dot.scale.setScalar(1.6 * (1 - age / 0.9));
      dot.updateMatrix();
      stars.setMatrixAt(used * 8 + k, dot.matrix);
    }
    used++;
  }
  dot.rotation.set(0, 0, 0);
  stars.count = used * 8;
  stars.instanceMatrix.needsUpdate = true;
  let puffCount = 0;
  for (const land of lands) {
    const age = time - land.t;
    if (age < 0) break;
    if (age > 0.5 || puffCount >= 15) continue;
    for (let k = 0; k < 6; k++) {
      const angle = k * 1.05 + land.t;
      dot.position.set((land.y as number) + Math.cos(angle) * (0.25 + age * 2.2), -(land.layer as number) * floors.layerHeight + 0.08 + age * 0.3, (land.x as number) + Math.sin(angle) * (0.25 + age * 2.2));
      dot.scale.setScalar(1.3 - age * 2);
      dot.updateMatrix();
      puffs.setMatrixAt(puffCount * 6 + k, dot.matrix);
    }
    puffCount++;
  }
  puffs.count = puffCount * 6;
  puffs.instanceMatrix.needsUpdate = true;
}

function drawWin() {
  if (!winEvent) { confetti.visible = false; return; }
  const age = time - winEvent.t;
  const winner = robots[winEvent.runner as number];
  crown.visible = confetti.visible = age >= 0;
  if (age < 0) return;
  // Dropped onto the head, then resting there.
  crown.position.copy(winner.head).setY(winner.head.y - 0.12 + Math.max(0.5 - age, 0) * 1.6);
  crown.rotation.y = time * 0.25;
  for (let i = 0; i < confetti.count; i++) {
    const a = i * 2.399963;
    const r = 0.3 + ((i * 7) % 23) * 0.3;
    const fall = ((age * (1.2 + (i % 5) * 0.2) + i * 0.13) % 8);
    dot.position.set(winner.head.x + Math.cos(a) * r + Math.sin(age * 2 + i) * 0.25, winner.head.y + 6 - fall, winner.head.z + Math.sin(a) * r);
    dot.rotation.set(age * 3 + i, age * 2.3 + i * 2, i);
    dot.scale.setScalar(age < 0.1 ? 0 : 1);
    dot.updateMatrix();
    confetti.setMatrixAt(i, dot.matrix);
  }
  confetti.instanceMatrix.needsUpdate = true;
  dot.rotation.set(0, 0, 0);
}

const shots = actionShots(meta, spectatorShots(spleefMeta, recording.duration, 7, 1.6));
const eliminations = meta.events.filter(event => event.type === 'splash');
const hud = {
  go: document.querySelector<HTMLElement>('#big')!,
  out: document.querySelector<HTMLElement>('#out')!,
  chips: document.querySelector<HTMLElement>('#chips')!,
  feed: document.querySelector<HTMLOListElement>('#feed')!,
  banner: document.querySelector<HTMLElement>('#banner')!,
  win: document.querySelector<HTMLElement>('#win')!,
  winName: document.querySelector<HTMLElement>('#win-name')!,
  stamp: document.querySelector<HTMLElement>('#stamp')!,
  stampName: document.querySelector<HTMLElement>('#stamp-name')!,
};
let time = 0;
let paused = false;
let follow: number | null = params.has('follow') ? Number(params.get('follow')) : null;
const closeness = params.has('close') ? 0.5 : 1;
let wide = params.get('cam') === 'wide';
const speed = Number(params.get('speed') ?? 1);
const chips = meta.runners.map((runner, i) => {
  const button = document.createElement('button');
  button.innerHTML = `<i style="background:${runner.color}"></i>${runner.name}`;
  button.onclick = () => { follow = follow === i ? null : i; wide = false; };
  hud.chips.append(button);
  return button;
});
const toggleWide = document.querySelector<HTMLButtonElement>('#wide')!;
toggleWide.onclick = () => { wide = !wide; follow = null; };
const togglePause = document.querySelector<HTMLButtonElement>('#pause')!;
togglePause.onclick = () => { paused = !paused; togglePause.textContent = paused ? 'Resume' : 'Pause'; };
document.querySelector<HTMLButtonElement>('#again')!.onclick = () => { time = 0; paused = false; togglePause.textContent = 'Pause'; };

new ResizeObserver(() => {
  const { width, height } = stage.getBoundingClientRect();
  renderer.setSize(width, height);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}).observe(stage);

const panelRows = meta.runners.map((runner, i) => {
  const row = document.createElement('li');
  row.innerHTML = `<i style="background:${runner.color}"></i><span>${runner.name}</span><em></em><b></b>`;
  document.querySelector('#tile-list')!.append(row);
  return { row, gait: row.querySelector('em')!, tiles: row.querySelector('b')!, index: i };
});
const rowHeight = 26;

function drawPanel(state: ReturnType<typeof roundStateAt>) {
  const { tiles, gait } = brainStateAt(meta, time);
  const order = panelRows.slice().sort((a, b) => Number(state.out[a.index] !== null) - Number(state.out[b.index] !== null) || tiles[a.index] - tiles[b.index]);
  order.forEach((entry, place) => {
    entry.row.style.transform = `translateY(${place * rowHeight}px)`;
    entry.row.classList.toggle('gone', state.out[entry.index] !== null);
    entry.row.classList.toggle('watched', watchedRunner(state) === entry.index);
    const name = gait[entry.index] === null ? '' : GAIT_NAMES[gait[entry.index]!];
    if (entry.gait.textContent !== name) { entry.gait.textContent = name; entry.gait.dataset.gait = String(gait[entry.index]); }
    entry.tiles.textContent = String(tiles[entry.index]);
  });
}

// Where the watched robot just went, drawn on the floor, so the route it chose can be read.
const trail = new THREE.InstancedMesh(new THREE.CircleGeometry(0.1, 16).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.85, depthWrite: false }), 30);
trail.frustumCulled = false;
scene.add(trail);

function drawTrail(state: ReturnType<typeof roundStateAt>) {
  const runner = watchedRunner(state);
  let used = 0;
  if (runner >= 0) {
    trail.material.color.set(meta.runners[runner].color);
    const layerNow = recording.joint(time, runner, 0, hipsNow) ? hipsNow[2] : null;
    for (let k = 1; layerNow !== null && k <= trail.count; k++) {
      const at = time - k * 0.12;
      // Only the path on this floor. A fall ends the line.
      if (at < floors.countdownSeconds || !recording.joint(at, runner, 0, hipsNow) || Math.abs(hipsNow[2] - layerNow) > 1.2) break;
      dot.position.set(hipsNow[1], Math.round((hipsNow[2] - 0.9) / floors.layerHeight) * floors.layerHeight + 0.03, hipsNow[0]);
      dot.scale.setScalar(1 - k / (trail.count + 6));
      dot.updateMatrix();
      trail.setMatrixAt(used++, dot.matrix);
    }
  }
  trail.count = used;
  trail.instanceMatrix.needsUpdate = true;
}

let lastHud = -1;
function drawHud(state: ReturnType<typeof roundStateAt>) {
  const countdown = floors.countdownSeconds - time;
  hud.banner.classList.toggle('show', time < floors.countdownSeconds - 0.3);
  const big = countdown > 0 ? String(Math.ceil(countdown)) : countdown > -0.8 ? 'GO!' : '';
  if (hud.go.textContent !== big) {
    hud.go.textContent = big;
    hud.go.classList.remove('pop');
    void hud.go.offsetWidth;
    if (big) hud.go.classList.add('pop');
  }
  // The stamp every eliminated robot gets. Not for the last one in, who won anyway.
  const stamped = state.winner === null ? eliminations.filter(event => time >= event.t && time < event.t + 1.5).pop() : undefined;
  hud.stamp.classList.toggle('show', !!stamped);
  if (stamped) hud.stampName.textContent = names[stamped.runner as number];
  const tenth = Math.floor(time * 10);
  if (tenth === lastHud) return;
  lastHud = tenth;
  const eliminated = state.out.filter(out => out !== null).length;
  hud.out.innerHTML = `ELIMINATED<b>${eliminated}/${meta.runners.length - 1}</b>`;
  chips.forEach((chip, i) => {
    chip.classList.toggle('gone', state.out[i] !== null);
    chip.classList.toggle('watched', watchedRunner(state) === i);
  });
  hud.feed.innerHTML = meta.events.filter(event => event.t <= time).map(event => describeHex(event, names)).filter(Boolean).slice(-3).reverse().map(line => `<li>${line}</li>`).join('');
  const won = state.winner !== null;
  hud.win.classList.toggle('show', won);
  drawPanel(state);
  hud.feed.style.display = won ? 'none' : '';
  if (won) hud.winName.textContent = names[state.winner!];
}

function watchedRunner(state: ReturnType<typeof roundStateAt>) {
  if (wide) return -1;
  if (follow !== null && state.out[follow] === null) return follow;
  return shotAt(shots, time).runner;
}

const lookAt = new THREE.Vector3();
const smooth = new THREE.Vector3();
const heading = new THREE.Vector3();
const past = new THREE.Vector3();

/** A time-averaged spot, so the camera does not copy every stride or turn. */
function averageHips(runner: number, at: number, seconds: number, out: THREE.Vector3) {
  out.set(0, 0, 0);
  let count = 0;
  for (let k = 0; k < 7; k++) {
    if (recording.joint(Math.max(at - (k / 6) * seconds, 0), runner, 0, hipsNow)) {
      out.x += hipsNow[1];
      out.y += hipsNow[2];
      out.z += hipsNow[0];
      count++;
    }
  }
  return count ? out.divideScalar(count) : null;
}

function drawCamera(state: ReturnType<typeof roundStateAt>) {
  const intro = time < floors.countdownSeconds && follow === null;
  if (wide || intro) {
    const orbit = (intro ? 0.5 : 0.1) * time + 0.9;
    const reach = intro ? 26 - time * 2.2 : 44;
    camera.position.set(Math.cos(orbit) * reach, intro ? 9 - time : 26, Math.sin(orbit) * reach);
    lookAt.set(0, intro ? 3 : -12, 0);
    camera.lookAt(lookAt);
    sun.target.position.set(0, -6, 0);
    sun.position.set(-14, 60, 10);
    return;
  }
  const runner = watchedRunner(state);
  if (winEvent && time >= winEvent.t + 0.6 && follow === null) {
    // The last shot circles the winner.
    const age = time - winEvent.t;
    const hips = robots[winEvent.runner as number].head;
    const orbit = age * 0.55 + 2;
    camera.position.set(hips.x + Math.cos(orbit) * 3.6, hips.y + 0.1, hips.z + Math.sin(orbit) * 3.6);
    lookAt.set(hips.x, hips.y + 0.05, hips.z);
    camera.lookAt(lookAt);
    sun.target.position.copy(lookAt);
    sun.position.set(hips.x - 14, hips.y + 40, hips.z + 10);
    return;
  }
  let t = time;
  while (t > 0 && !recording.joint(t, runner, 0, hipsNow)) t -= 0.1;
  if (!averageHips(runner, t, 0.45, smooth)) return;
  averageHips(runner, Math.max(t - 1.1, 0), 0.5, past);
  heading.copy(smooth).sub(past).setY(0);
  if (heading.lengthSq() < 0.05) heading.set(-smooth.x, 0, -smooth.z);
  if (heading.lengthSq() < 0.01) heading.set(0, 0, 1);
  heading.normalize();
  const aim = Math.max(smooth.y, slimeY + 0.8);
  // Far above and behind, looking down across the honeycomb like the footage. Below the next floor up.
  lookAt.set(smooth.x + heading.x * 1.2 * closeness, aim - 0.25, smooth.z + heading.z * 1.2 * closeness);
  camera.position.set(smooth.x - heading.x * 4.8 * closeness, aim + 3.5 * closeness, smooth.z - heading.z * 4.8 * closeness);
  camera.lookAt(lookAt);
  sun.target.position.set(smooth.x, aim, smooth.z);
  sun.position.set(smooth.x - 14, aim + 40, smooth.z + 10);
}

let previous = performance.now();
function draw() {
  drawTiles();
  drawEffects();
  world.update(time);
  const state = roundStateAt(spleefMeta, time);
  robots.forEach((robot, i) => {
    const present = robot.draw(time);
    const tag = tags[i];
    tag.visible = present && state.out[i] === null && state.winner === null;
    if (!present) return;
    // Nothing plays while a robot waits on its hexagon or after it has won, so it rocks on its feet.
    robot.wiggle(time, time < floors.countdownSeconds || state.winner === i ? 1 : 0);
    tag.position.copy(robot.head).setY(robot.head.y + 0.42);
  });
  drawImpacts();
  drawTrail(state);
  drawWin();
  drawHud(state);
  drawCamera(state);
  // A floor above the camera would only block the view of the one below.
  tileSets.forEach((set, layer) => { set.mesh.visible = -layer * floors.layerHeight < camera.position.y - 1.6; });
  pedestals.forEach(mesh => { mesh.visible &&= floors.pedestalHeight < camera.position.y - 1.6; });
  renderer.render(scene, camera);
}
function frame(now: number) {
  const elapsed = Math.min((now - previous) / 1000, 0.1);
  previous = now;
  if (!paused) time = Math.min(time + elapsed * speed, recording.duration);
  draw();
  requestAnimationFrame(frame);
}
Object.assign(window, {
  seekRound: (seconds: number) => { time = seconds; paused = true; draw(); },
  roundInfo: { duration: recording.duration, ranking: () => ranking(spleefMeta, roundStateAt(spleefMeta, time)).map(r => r.name), renderer },
});
requestAnimationFrame(frame);
document.body.dataset.ready = '1';
