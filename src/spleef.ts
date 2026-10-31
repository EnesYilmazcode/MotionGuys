import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Figure } from './figure.ts';
import { Recording, type RaceMeta, type Vec3 } from './game/playback.ts';
import { describeRound, ranking, roundStateAt, shotAt, spectatorShots, tileKey, type SpleefMeta } from './game/spleef.ts';
import './style.css';

const seed = new URLSearchParams(location.search).get('seed') ?? '1';
const base = `spleef/seed_${seed}`;
const [meta, buffer] = await Promise.all([
  fetch(`${base}.json`).then(r => r.json() as Promise<SpleefMeta>),
  fetch(`${base}.bin`).then(r => r.arrayBuffer()),
]);
const recording = new Recording(meta as unknown as RaceMeta, buffer);
const { arena: floors } = meta;

const arena = document.querySelector<HTMLElement>('#arena')!;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#1b2a4a');
scene.fog = new THREE.Fog('#1b2a4a', 28, 70);
const camera = new THREE.PerspectiveCamera(44, 1, 0.1, 200);
camera.position.set(17, 17, 20);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
arena.append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.maxDistance = 70;
controls.minDistance = 3;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445577, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2);
sun.position.set(-10, 28, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16, far: 70 });
scene.add(sun);

const palettes = [['#ffb547', '#ffc978'], ['#57c7a8', '#83dcc2'], ['#ee6c9a', '#f59dbb']];
const shakeMaterial = new THREE.MeshStandardMaterial({ color: '#fff4d6', roughness: 0.5, emissive: '#ff5a36', emissiveIntensity: 0.35 });
const tileGeometry = new THREE.BoxGeometry(floors.tile - 0.05, 0.35, floors.tile - 0.05);
const tiles = new Map<string, { mesh: THREE.Mesh; base: THREE.Material; y: number }>();
for (let layer = 0; layer < floors.layers; layer++) {
  const materials = palettes[layer % palettes.length].map(color => new THREE.MeshStandardMaterial({ color, roughness: 0.6 }));
  const y = -layer * floors.layerHeight - 0.175;
  for (let column = 0; column < floors.tiles; column++) for (let row = 0; row < floors.tiles; row++) {
    const material = materials[(column + row) % 2];
    const mesh = new THREE.Mesh(tileGeometry, material);
    // The round records X and Y across the floor and Z up. Three.js is Y up.
    mesh.position.set(-floors.half + (row + 0.5) * floors.tile, y, -floors.half + (column + 0.5) * floors.tile);
    mesh.receiveShadow = mesh.castShadow = true;
    scene.add(mesh);
    tiles.set(tileKey(layer, column, row), { mesh, base: material, y });
  }
}

const figures = meta.runners.map((runner, index) => new Figure(scene, recording, index, runner));
const shots = spectatorShots(meta, recording.duration);
const watching = document.querySelector<HTMLElement>('#watching')!;
let spectating = true;
renderer.domElement.addEventListener('pointerdown', () => { spectating = false; });
renderer.domElement.addEventListener('dblclick', () => { spectating = true; });
const standingsList = document.querySelector<HTMLOListElement>('#standings')!;
const clockLabel = document.querySelector<HTMLElement>('#clock')!;
const feed = document.querySelector<HTMLOListElement>('#feed')!;
document.querySelector('#seed')!.textContent = seed;
const pauseButton = document.querySelector<HTMLButtonElement>('#pause')!;
let time = 0;
let paused = false;
pauseButton.onclick = () => { paused = !paused; pauseButton.textContent = paused ? 'Resume' : 'Pause'; };
document.querySelector<HTMLButtonElement>('#restart')!.onclick = () => { time = 0; paused = false; pauseButton.textContent = 'Pause'; };
new ResizeObserver(() => {
  const { width, height } = arena.getBoundingClientRect();
  renderer.setSize(width, height);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}).observe(arena);

function highestActiveFloor(state: ReturnType<typeof roundStateAt>) {
  const active = state.layer.filter((_, runner) => state.out[runner] === null);
  return active.length ? Math.min(...active) : floors.layers - 1;
}

function drawFloors(state: ReturnType<typeof roundStateAt>) {
  // A floor above the action would only block the view of the one below.
  const topFloor = spectating ? state.layer[shotAt(shots, time).runner] : highestActiveFloor(state);
  for (const [key, tile] of tiles) {
    const dropped = state.gone.get(key);
    const shaking = state.shaking.has(key);
    tile.mesh.material = shaking ? shakeMaterial : tile.base;
    tile.mesh.visible = Number(key.split(':')[0]) >= topFloor && (dropped === undefined || time - dropped < 1);
    tile.mesh.position.y = tile.y - (dropped === undefined ? 0 : 4.9 * (time - dropped) ** 2) + (shaking ? Math.sin(time * 60) * 0.03 : 0);
  }
}

let lastHud = -1;
function drawHud(state: ReturnType<typeof roundStateAt>) {
  const tenth = Math.floor(time * 10);
  if (tenth === lastHud) return;
  lastHud = tenth;
  clockLabel.textContent = `${time.toFixed(1)}s`;
  standingsList.innerHTML = ranking(meta, state).map((runner, place) => {
    const status = runner.index === state.winner ? 'winner' : runner.out !== null ? `out ${runner.out.toFixed(1)}s` : `floor ${runner.layer + 1}`;
    return `<li><span class="place">${place + 1}</span><span class="swatch" style="background:${runner.color}"></span>${runner.name}<span class="status">${status}</span></li>`;
  }).join('');
  feed.innerHTML = meta.events.filter(event => event.t <= time).map(event => describeRound(event, meta)).filter(Boolean).slice(-5).reverse().map(line => `<li>${line}</li>`).join('');
}

scene.add(sun.target);
const hips: Vec3 = [0, 0, 0];
const past: Vec3 = [0, 0, 0];
const lookAt = new THREE.Vector3();
const behind = new THREE.Vector3();

// A chase camera over the watched runner. It is a pure function of time, so seeking and replay match.
function spectate() {
  const { runner } = shotAt(shots, time);
  const name = meta.runners[runner].name;
  if (watching.textContent !== name) watching.textContent = name;
  let t = time;
  while (t > 0 && !recording.joint(t, runner, 0, hips)) t -= 0.1;
  if (!recording.joint(t, runner, 0, hips)) return;
  behind.set(0, 0, 0);
  for (const seconds of [0.8, 1.6, 2.4]) {
    recording.joint(Math.max(t - seconds, 0), runner, 0, past);
    behind.set(past[1] - hips[1], 0, past[0] - hips[0]);
    if (behind.length() > 0.8) break;
  }
  if (behind.length() < 0.01) behind.set(hips[1], 0, hips[0]);
  if (behind.length() < 0.01) behind.set(0, 0, -1);
  behind.normalize();
  lookAt.set(hips[1] - behind.x * 1.2, Math.max(hips[2], -floors.layers * floors.layerHeight) + 0.1, hips[0] - behind.z * 1.2);
  camera.position.set(hips[1] + behind.x * 4.2, lookAt.y + 2.2, hips[0] + behind.z * 4.2);
  camera.lookAt(lookAt);
  controls.target.copy(lookAt);
  sun.target.position.set(hips[1], lookAt.y, hips[0]);
  sun.position.set(hips[1] - 8, lookAt.y + 24, hips[0] + 6);
}

let previous = performance.now();
function frame(now: number) {
  const elapsed = Math.min((now - previous) / 1000, 0.1);
  previous = now;
  if (!paused) time = Math.min(time + elapsed, recording.duration);
  const state = roundStateAt(meta, time);
  drawFloors(state);
  drawHud(state);
  figures.forEach(figure => figure.draw(time));
  if (spectating) spectate();
  else controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
Object.assign(window, { seekRace: (seconds: number) => { time = seconds; paused = true; } });
requestAnimationFrame(frame);
