import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Recording, courseStateAt, describe, type RaceMeta, type Vec3 } from '../src/game/playback.ts';

const meta: RaceMeta = {
  seed: 1, fps: 10, frames: 2, joints: ['Hips', 'Head'],
  course: {
    halfWidth: 6, finishX: 100, doorRows: [], doorCount: 6, sweepers: [], postRadius: 0.5,
    tiles: { x: 70, size: 3, rows: 8, columns: 4 },
  },
  runners: [{ name: 'Ada', color: '#fff', finishedAt: null }, { name: 'Bo', color: '#000', finishedAt: null }],
  events: [
    { t: 1, type: 'door_break', runner: 0, row: 0, door: 2 },
    { t: 2, type: 'knockdown', runner: 1, cause: 'jump' },
    { t: 2, type: 'arm_hit', runner: 1, sweeper: 'low' },
    { t: 3, type: 'tile_shake', runner: 0, row: 1, column: 1 },
    { t: 4, type: 'tile_drop', runner: null, row: 1, column: 1 },
    { t: 5, type: 'recover', runner: 1 },
    { t: 6, type: 'finish', runner: 0, place: 1 },
  ],
};

function buffer() {
  const hips = new Float32Array([1, 2, 3, NaN, NaN, NaN, 2, 2, 3, NaN, NaN, NaN]);
  const joints = new Int16Array([0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0]);
  const bytes = new Uint8Array(hips.byteLength + joints.byteLength);
  bytes.set(new Uint8Array(hips.buffer), 0);
  bytes.set(new Uint8Array(joints.buffer), hips.byteLength);
  return bytes.buffer;
}

test('joints blend between frames, relative to the hips', () => {
  const recording = new Recording(meta, buffer());
  const out: Vec3 = [0, 0, 0];
  assert.deepEqual(recording.joint(0.05, 0, 1, out), [1.5, 2, 4.5]);
});

test('a runner who has left the race has no joints', () => {
  assert.equal(new Recording(meta, buffer()).joint(0, 1, 0, [0, 0, 0]), null);
});

test('a truncated recording is rejected', () => {
  assert.throws(() => new Recording(meta, buffer().slice(0, 40)), /expected/);
});

test('course state comes from the events up to that time', () => {
  const early = courseStateAt(meta, 3.5);
  assert.ok(early.broken.has('0:2'));
  assert.ok(early.shaking.has('1:1'));
  assert.equal(early.status[1], 'down');
  assert.equal(early.knockdowns[1], 1);
  const late = courseStateAt(meta, 6);
  assert.equal(late.shaking.size, 0);
  assert.equal(late.gone.get('1:1'), 4);
  assert.equal(late.status[1], 'running');
  assert.equal(late.finished[0], 6);
});

test('events read as plain sentences', () => {
  assert.equal(describe(meta.events[2], meta), 'Bo was hit by the low sweeper');
  assert.equal(describe(meta.events[1], meta), null);
});
