import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Figure } from './figure.ts';
import { Recording, courseStateAt, describe, standings, type RaceMeta, type Vec3 } from './game/playback.ts';
import './style.css';

const seed = new URLSearchParams(location.search).get('seed') ?? '1';
const base = `races/seed_${seed}`;
const [meta, buffer] = await Promise.all([
  fetch(`${base}.json`).then(r => r.json() as Promise<RaceMeta>),
  fetch(`${base}.bin`).then(r => r.arrayBuffer()),
]);
const recording = new Recording(meta, buffer);
const { course } = meta;

const arena = document.querySelector<HTMLElement>('#arena')!;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#9fd6ee');
scene.fog = new THREE.Fog('#9fd6ee', 70, 170);
const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 300);
camera.position.set(17, 13, -13);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
arena.append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.maxPolarAngle = Math.PI / 2.1;
controls.minDistance = 5;
controls.maxDistance = 90;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.9);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -24, right: 24, top: 24, bottom: -24, far: 90 });
scene.add(sun, sun.target);

// The race records X forward, Y across, Z up. Three.js is Y up.
const place = (object: THREE.Object3D, x: number, y: number, z: number) => object.position.set(y, z, x);

function box(w: number, h: number, d: number, color: string, x: number, y: number, z: number, parent: THREE.Object3D = scene) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({ color, roughness: 0.65 }));
  place(mesh, x, y, z);
  mesh.castShadow = mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

const width = course.halfWidth * 2;
const { tiles } = course;
const tileStart = tiles.x;
const tileEnd = tiles.x + tiles.rows * tiles.size;
const lanes = ['#d9d2bf', '#e6dfcc'];
for (let x = -8, i = 0; x < tileStart; x += 4, i++) box(width, 0.6, Math.min(4, tileStart - x), lanes[i % 2], x + Math.min(4, tileStart - x) / 2, 0, -0.3);
const runoff = 18;
box(width, 0.6, course.finishX + runoff - tileEnd, lanes[0], (tileEnd + course.finishX + runoff) / 2, 0, -0.3);
box(width, 0.04, 0.8, '#b8f153', 0, 0, 0.02);
for (let i = 0; i < width; i++) for (let j = 0; j < 2; j++) {
  box(1, 0.04, 0.6, (i + j) % 2 ? '#25354a' : '#ffffff', course.finishX + j * 0.6, i - course.halfWidth + 0.5, 0.02);
}
for (const y of [-course.halfWidth, course.halfWidth]) box(0.4, 5, 0.4, '#25354a', course.finishX, y, 2.5);
box(width + 0.4, 0.8, 0.4, '#b8f153', course.finishX, 0, 5);

const doorWidth = width / course.doorCount;
const doorMeshes = new Map<string, THREE.Mesh>();
course.doorRows.forEach((row, rowIndex) => {
  for (let door = 0; door < course.doorCount; door++) {
    const y = -course.halfWidth + (door + 0.5) * doorWidth;
    const mesh = box(doorWidth - 0.1, 2.6, 0.22, door % 2 ? '#ff7a59' : '#ffc145', row.x, y, 1.3);
    doorMeshes.set(`${rowIndex}:${door}`, mesh);
  }
  box(width + 0.3, 0.5, 0.4, '#7a5cd6', row.x, 0, 2.85);
});

const arms = course.sweepers.map(sweeper => {
  const post = new THREE.Mesh(
    new THREE.CylinderGeometry(course.postRadius, course.postRadius, Math.max(sweeper.z + 0.5, 1.5), 28),
    new THREE.MeshStandardMaterial({ color: '#4c78c9', roughness: 0.5 }),
  );
  place(post, sweeper.x, 0, post.geometry.parameters.height / 2);
  post.castShadow = true;
  scene.add(post);
  const pivot = new THREE.Group();
  place(pivot, sweeper.x, 0, sweeper.z);
  const arm = new THREE.Mesh(
    new THREE.CapsuleGeometry(sweeper.radius, sweeper.length * 2, 8, 20),
    new THREE.MeshStandardMaterial({ color: sweeper.action === 'jump' ? '#e8504f' : '#37b3a4', roughness: 0.4 }),
  );
  arm.rotation.z = Math.PI / 2;
  arm.castShadow = true;
  pivot.add(arm);
  scene.add(pivot);
  return { sweeper, pivot };
});

const tileMeshes = new Map<string, THREE.Mesh>();
for (let row = 0; row < tiles.rows; row++) for (let column = 0; column < tiles.columns; column++) {
  const mesh = box(tiles.size - 0.08, 0.6, tiles.size - 0.08, (row + column) % 2 ? '#7fd1ae' : '#a4e3c5',
    tiles.x + (row + 0.5) * tiles.size, -course.halfWidth + (column + 0.5) * tiles.size, -0.3);
  tileMeshes.set(`${row}:${column}`, mesh);
}

const figures = meta.runners.map((runner, index) => new Figure(scene, recording, index, runner));
const a: Vec3 = [0, 0, 0];

const standingsList = document.querySelector<HTMLOListElement>('#standings')!;
const clockLabel = document.querySelector<HTMLElement>('#clock')!;
const feed = document.querySelector<HTMLOListElement>('#feed')!;
document.querySelector('#seed')!.textContent = seed;
const pauseButton = document.querySelector<HTMLButtonElement>('#pause')!;
let time = 0;
let paused = false;
let follow = true;
pauseButton.onclick = () => { paused = !paused; pauseButton.textContent = paused ? 'Resume' : 'Pause'; };
document.querySelector<HTMLButtonElement>('#restart')!.onclick = () => { time = 0; paused = false; pauseButton.textContent = 'Pause'; };
renderer.domElement.addEventListener('pointerdown', () => { follow = false; });
renderer.domElement.addEventListener('dblclick', () => { follow = true; });
new ResizeObserver(() => {
  const { width: w, height: h } = arena.getBoundingClientRect();
  renderer.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}).observe(arena);

function drawCourse() {
  const state = courseStateAt(meta, time);
  for (const [key, mesh] of doorMeshes) {
    mesh.visible = !state.broken.has(key);
    (mesh.material as THREE.MeshStandardMaterial).color.set(state.solid.has(key) ? '#4d5666' : key.split(':')[1] === undefined || Number(key.split(':')[1]) % 2 ? '#ff7a59' : '#ffc145');
  }
  for (const { sweeper, pivot } of arms) pivot.rotation.y = -(sweeper.angle + sweeper.speed * time) + Math.PI / 2;
  for (const [key, mesh] of tileMeshes) {
    const dropped = state.gone.get(key);
    const shaking = state.shaking.has(key);
    mesh.position.y = dropped === undefined ? -0.3 : -0.3 - 4.9 * (time - dropped) ** 2;
    mesh.visible = dropped === undefined || time - dropped < 1.6;
    mesh.position.x = (mesh.userData.baseX ??= mesh.position.x) + (shaking ? Math.sin(time * 70) * 0.05 : 0);
  }
  return state;
}

let lastHud = -1;
function drawHud(state: ReturnType<typeof courseStateAt>) {
  const tenth = Math.floor(time * 10);
  if (tenth === lastHud) return;
  lastHud = tenth;
  clockLabel.textContent = `${time.toFixed(1)}s`;
  standingsList.innerHTML = standings(recording, time, state).map((runner, place) => {
    const status = runner.finished !== null ? `${runner.finished.toFixed(1)}s`
      : state.status[runner.index] === 'running' ? `${state.knockdowns[runner.index]} down` : state.status[runner.index];
    return `<li><span class="place">${place + 1}</span><span class="swatch" style="background:${runner.color}"></span>${runner.name}<span class="status">${status}</span></li>`;
  }).join('');
  feed.innerHTML = meta.events.filter(event => event.t <= time).map(event => describe(event, meta)).filter(Boolean).slice(-5).reverse().map(line => `<li>${line}</li>`).join('');
}

const focus = new THREE.Vector3(0, 1, 0);
const target = new THREE.Vector3();
function followPack() {
  const positions: number[] = [];
  for (let runner = 0; runner < meta.runners.length; runner++) {
    const hips = recording.joint(time, runner, 0, a);
    if (hips && hips[2] > -1) positions.push(hips[0]);
  }
  if (!positions.length) return;
  positions.sort((p, q) => q - p);
  const x = positions[Math.floor(positions.length / 3)];
  target.set(0, 1, x);
  const move = target.clone().sub(focus).multiplyScalar(0.05);
  focus.add(move);
  sun.position.set(-12, 34, focus.z - 8);
  sun.target.position.copy(focus);
  if (follow) {
    camera.position.add(move);
    controls.target.copy(focus);
  }
}

let previous = performance.now();
function frame(now: number) {
  const elapsed = Math.min((now - previous) / 1000, 0.1);
  previous = now;
  if (!paused) time = Math.min(time + elapsed, recording.duration);
  drawHud(drawCourse());
  figures.forEach(figure => figure.draw(time));
  followPack();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
Object.assign(window, { seekRace: (seconds: number) => { time = seconds; paused = true; for (let i = 0; i < 400; i++) followPack(); } });
requestAnimationFrame(frame);
