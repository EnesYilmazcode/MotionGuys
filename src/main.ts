import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { course } from './game/course.ts';
import { Race, STEP } from './game/race.ts';
import './style.css';

const params = new URLSearchParams(location.search);
const rawSeed = Number(params.get('seed') ?? 42);
const seed = Number.isFinite(rawSeed) ? rawSeed >>> 0 : 42;
let race = new Race(seed);
let paused = false;
let accumulated = 0;
const arena = document.querySelector<HTMLElement>('#arena')!;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#b4d6e4');
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 250);
camera.position.set(30, 37, -26);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
arena.append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 32);
controls.maxPolarAngle = Math.PI / 2.1;
controls.minDistance = 8;
controls.maxDistance = 120;
controls.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0x66778c, 2.5));
const sun = new THREE.DirectionalLight(0xffffff, 3);
sun.position.set(-20, 45, 10);
scene.add(sun);

function box(w: number, h: number, d: number, color: string, x: number, y: number, z: number, parent: THREE.Object3D = scene) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({ color, roughness: 0.8 }));
  mesh.position.set(x, y, z);
  mesh.castShadow = mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

// Leave a real visual gap in the final section of the course.
box(14, 1, 52, '#ecf0e0', 0, -0.5, 26);
box(14, 1, 19, '#ecf0e0', 0, -0.5, 66.5);
for (const x of [-7.35, 7.35]) {
  box(0.5, 0.6, 52, '#708b99', x, 0, 26);
  box(0.5, 0.6, 19, '#708b99', x, 0, 66.5);
}
box(14, 0.04, 2, '#b8f153', 0, 0.03, 3);
box(14, 1.1, 0.8, '#fb7ca8', 0, 0.55, 19);
box(14, 0.7, 7, '#8faaff', 0, 1.7, 36.5);
for (const x of [-6.5, 6.5]) box(0.5, 2.1, 7, '#687dd4', x, 1.05, 36.5);
for (let i = 0; i < 14; i++) for (let j = 0; j < 2; j++) {
  box(1, 0.05, 0.65, (i + j) % 2 ? '#25354a' : '#ffffff', i - 6.5, 0.04, course.finish + j * 0.65);
}
for (const x of [-7, 7]) box(0.3, 4, 0.3, '#25354a', x, 2, course.finish);
box(14.3, 0.6, 0.4, '#b8f153', 0, 4, course.finish);

function makeRunner(color: string) {
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);
  box(0.55, 0.7, 0.36, color, 0, 1.15, 0, body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.25, 16, 12), new THREE.MeshStandardMaterial({ color: '#f5d5b8' }));
  head.position.y = 1.78;
  body.add(head);
  const limbs = [-1, 1].map(side => {
    const leg = new THREE.Group(); leg.position.set(side * 0.17, 0.85, 0); body.add(leg);
    box(0.2, 0.72, 0.23, '#334762', 0, -0.36, 0, leg);
    const arm = new THREE.Group(); arm.position.set(side * 0.39, 1.45, 0); body.add(arm);
    box(0.17, 0.65, 0.2, color, 0, -0.32, 0, arm);
    return { leg, arm };
  });
  scene.add(root);
  return { root, body, limbs };
}
const characters = race.runners.map(r => makeRunner(r.color));
const standings = document.querySelector<HTMLOListElement>('#standings')!;
const clockLabel = document.querySelector<HTMLElement>('#clock')!;
document.querySelector('#seed')!.textContent = String(seed);
const pauseButton = document.querySelector<HTMLButtonElement>('#pause')!;
pauseButton.onclick = () => { paused = !paused; pauseButton.textContent = paused ? 'Resume' : 'Pause'; };
document.querySelector<HTMLButtonElement>('#restart')!.onclick = () => {
  race = new Race(seed); accumulated = 0; paused = false; pauseButton.textContent = 'Pause';
};
const resize = new ResizeObserver(() => {
  const { width, height } = arena.getBoundingClientRect();
  renderer.setSize(width, height);
  camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix();
});
resize.observe(arena);

let previous = performance.now();
let lastHud = -1;
function frame(now: number) {
  const elapsed = Math.min((now - previous) / 1000, 0.1); previous = now;
  if (!paused) {
    accumulated += elapsed;
    while (accumulated >= STEP) { race.step(); accumulated -= STEP; }
  }
  for (const runner of race.runners) {
    const actor = characters[runner.id];
    const obstacle = course.obstacles.find(o => o.action === runner.action);
    const phase = obstacle ? (runner.z - obstacle.start) / (obstacle.end - obstacle.start) : 0;
    const lift = runner.action === 'jump' ? Math.sin(phase * Math.PI) * 2.2 : runner.action === 'vault' ? Math.sin(phase * Math.PI) * 1.5 : 0;
    actor.root.position.set(runner.x, lift, runner.z);
    actor.body.scale.y = runner.action === 'duck' ? 0.55 : 1;
    const swing = runner.finishedAt === null ? Math.sin(race.time * runner.speed * 3) * 0.65 : 0;
    actor.limbs.forEach(({ leg, arm }, i) => { leg.rotation.x = swing * (i ? 1 : -1); arm.rotation.x = -leg.rotation.x; });
  }
  if (Math.floor(race.time * 10) !== lastHud) {
    lastHud = Math.floor(race.time * 10);
    clockLabel.textContent = race.complete ? 'Finished' : `${race.time.toFixed(1)}s`;
    standings.innerHTML = race.standings.map((r, i) => `<li><span class="place">${i + 1}</span><span class="swatch" style="background:${r.color}"></span>${r.name}<span class="status">${r.finishedAt !== null ? `${r.finishedAt.toFixed(1)}s` : r.action}</span></li>`).join('');
  }
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
